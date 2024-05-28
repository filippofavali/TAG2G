import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import pdb
import os
from pathlib import Path

# edited Favali, 03-01-2024

speaker_id_dict = {
    2: 0,
    10: 1,
}


def serial_printer(**kwargs):

    for key, dict in kwargs.items():
        for person, value in dict.items():
            print(f"{key}-{person}: {value.shape}")


class SpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, audio_dim, style_dim, sequence_length=5, npy_root="../../process",
                 dataset="trn", version='v0', speakers=('main-agent', 'interloctr'), motion_window_length=18):

        self.motion_window_length = motion_window_length
        self.motion_dim = motion_dim
        self.audio_dim = audio_dim
        self.dataset = dataset
        self.style_dim = style_dim
        if '_dyadic' in version:
            # dyadic info is not usefull in the scope of h5_dataloader so won't store this
            self.version = version.replace("_dyadic", "")
        else:
            self.version = version
        self.speakers = speakers
        self.stat_dir = os.path.dirname(h5file)

        with h5py.File(h5file, "r") as h5:
            self.len = len(h5.keys())

            # loading features stats to further standardization
            assert os.path.isdir(self.stat_dir), 'stats_dir is not a dir'
            # nb file name: {speaker}-gesture_trn_{stat}_v0.npy
            self.stats = {}
            statistics = ('mean', 'std')

            for speaker in self.speakers:
                for stat in statistics:
                    stats_pointer = os.path.join(self.stat_dir, f'{speaker}-gesture_{self.dataset}_{stat}_{self.version}.npy')
                    assert os.path.isfile(stats_pointer), f"Provided stats_pointer '{stats_pointer}' is not a file"
                    self.stats[f"{speaker}_{stat}"] = np.load(stats_pointer)

            # load features data from h5 file
            self.text = {}
            self.audio = {}
            self.gesture = {}               # gesture featurization (n_features, 256)
            self.motion = {}                # ground truth gesture to compute model loss (sample_frames, 74)
            self.gesture_vel = {}
            self.gesture_acc = {}
            self.id = {}

            for speaker in self.speakers:

                self.text[f"{speaker}"] = [h5[str(i)][f"{speaker}_text"][:] for i in range(len(h5.keys()))]
                self.audio[f"{speaker}"] = [h5[str(i)][f"{speaker}_audio"][:] for i in range(len(h5.keys()))]
                self.gesture[f"{speaker}"] = [(h5[str(i)][f"{speaker}_gesture"][:] - self.stats[f'{speaker}_mean']) / self.stats[f'{speaker}_std'] for i in range(len(h5.keys()))]
                # adding groundTruth gesture to compare at the end of TAG2G pipeline
                self.motion[f"{speaker}"] = [h5[str(i)][f"{speaker}_GT"][:] for i in range(len(h5.keys()))]
                if "v0" in self.version:
                    self.gesture_vel[f"{speaker}"] = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in
                                        self.gesture[f"{speaker}"]]
                    self.gesture_acc[f"{speaker}"] = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in
                                        self.gesture_vel[f"{speaker}"]]
                self.id[f"{speaker}"] = [int(np.array(h5[str(i)[0]][f"{speaker}_id"])) for i in range(len(h5.keys()))]

        self.sequence_length = sequence_length    # number of windows computed per each iteration 'n_poses' in config

        print(f"Total clips: {(self.len)}")
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.gesture[self.speakers[0]])            # return length of 'main-agent' features vector

    def __getitem__(self, idx):
        """
        :param idx: an int between 0 and len(gesture['main-agent']), dataset length
        :return: 4 dicts (text&audio as conditions, gesture, motion, speakers ID)
        """
        total_windows_len = self.gesture[f"{self.speakers[0]}"][idx].shape[0]
        start_window = np.random.randint(0, total_windows_len - self.segment_length)
        end_window = start_window + self.segment_length

        # return 4 dicts, with data of both the speakers into them
        text_audio = {}
        gesture = {}
        motion = {}
        speaker_id = {}

        for speaker in self.speakers:

            # Load existing data or zero-padding if idx==0 and speaker==interloctrù
            if speaker == 'main-agent':
                audio = self.audio[f"{speaker}"][idx][start_window:end_window]
                text = self.text[f"{speaker}"][idx][start_window:end_window]
                textaudio = np.concatenate((audio, text), axis=-1)
                text_audio[f"{speaker}"] = torch.FloatTensor(textaudio)
                pos = self.gesture[f"{speaker}"][idx][start_window:end_window]
                # returning ground truth motion to be used in loss over real samples
                shift = self.motion_window_length
                motion[f"{speaker}"] = torch.FloatTensor(self.motion[f"{speaker}"][idx][start_window * shift:end_window * shift])
                # if v0 return also cinematic values
                if "v0" in self.version:
                    vel = self.gesture_vel[f"{speaker}"][idx][start_window:end_window]
                    acc = self.gesture_acc[f"{speaker}"][idx][start_window:end_window]
                    gesture[f"{speaker}"] = torch.FloatTensor(np.concatenate((pos, vel, acc), axis=-1))
                else:
                    gesture[f"{speaker}"] = torch.FloatTensor(pos)

            elif speaker == 'interloctr' and start_window < self.segment_length:
                # zero-padding of inputs --> -1 data doesnt exits
                # data exists from 0 to end frame-18 .. so i
                # TODO: this implementation is to be checked
                text_audio[f"{speaker}"] = torch.FloatTensor(np.zeros((self.segment_length, self.motion_window_length, self.audio_dim)))
                if "v0" in self.version:
                    # DONE: return a Float tensor with correct shape (3x motion dim)
                    # TODO: Review this because is not all to be zero padded - only first code has to be a zero then from 1 to 7 code are good...
                    gesture[f"{speaker}"] = torch.FloatTensor(np.zeros((self.segment_length, 3*self.motion_dim)))
                else:
                    raise NotImplementedError

            elif speaker == 'interloctr' and start_window >= self.segment_length:
                # DONE: the same ad main-agent but with start window and end window at -1
                # TODO: Here need to see if till -8 codes or not ... same problem as above .. in inference Do not have that amount of data
                audio = self.audio[f"{speaker}"][idx][start_window-self.segment_length:end_window-self.segment_length]
                text = self.text[f"{speaker}"][idx][start_window-self.segment_length:end_window-self.segment_length]
                textaudio = np.concatenate((audio, text), axis=-1)
                text_audio[f"{speaker}"] = torch.FloatTensor(textaudio)
                pos = self.gesture[f"{speaker}"][idx][start_window-self.segment_length:end_window-self.segment_length]
                # if v0 return also cinematic values
                if "v0" in self.version:
                    vel = self.gesture_vel[f"{speaker}"][idx][start_window-self.segment_length:end_window-self.segment_length]
                    acc = self.gesture_acc[f"{speaker}"][idx][start_window-self.segment_length:end_window-self.segment_length]
                    gesture[f"{speaker}"] = torch.FloatTensor(np.concatenate((pos, vel, acc), axis=-1))
                else:
                    raise NotImplementedError

            # returning ID
            id = np.zeros([self.style_dim])
            id[self.id[f"{speaker}"][idx]] = 1
            speaker_id[f"{speaker}"] = torch.FloatTensor(id)

        return text_audio, gesture, motion, speaker_id


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id

    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id

    def __iter__(self):
        return iter(range(self.min_id, self.max_id))


if __name__ == '__main__':
    '''
    cd ./BEAT-main/mydiffusion_beat/data_loader
    python h5_data_loader.py
    '''

    # Some test to assess quality of gesture dataset and dataloader

    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset(h5file="../../../data/trn_features_with_wavlm/TWH-trn_with_wavlm.h5",
                                       motion_dim=256, audio_dim=1434,
                                       style_dim=17,
                                       sequence_length=8)

    train_loader = DataLoader(trn_dataset, num_workers=4,
                              sampler=SequentialSampler(0, len(trn_dataset)),
                              batch_size=350,
                              pin_memory=True,
                              drop_last=False)

    n_seed = 1
    version = "v0_dyadic"
    cond_labels = {
        'main-agent': 'y',
        'interloctr': 'y_inter1'
    }

    key1 = False
    if key1:

        for batch_i, batch in enumerate(train_loader, 0):
            print(batch_i)
            if batch_i == 0:
                print("Inside batch 1")
                # testing what happens during training_loop.run_loop(self) called from end2end
                device = "cuda"
                text_audio, gesture, gesture_gt, style = batch
                # textaudio, gesture, speaker = batch     # [b, 8, 18, 302+108], [b, 8, 256], [b, 8*18, 74]
                cond_ = {}

                for person in trn_dataset.speakers:
                    if (not 'dyadic' in version) and (person != 'main-agent'):
                        # if not in dyadic and this person is not the speaker: break the cycle of building
                        # then output will be cond_ = {'y': {...}}}
                        break

                    # load person data from batch
                    motion = gesture[person].permute(0, 2, 1).unsqueeze(2).to(device, non_blocking=True)
                    wavlm = text_audio[person]
                    print(f"Accessing '{person}'")
                    id = style[person]

                    # pack person data into cond_
                    label = cond_labels[person]
                    cond_[label] = {}
                    cond_[label]['seed'] = motion[..., 0:n_seed]
                    cond_[label]['gesture'] = motion
                    cond_[label]['style'] = id.to(device, non_blocking=True)
                    cond_[label]['audio'] = wavlm.to(torch.float32)[:, n_seed:].to(device,
                                                                                        non_blocking=True)  # attention4
                    if person == 'main-agent':
                        # only main-agent is interested in training loss
                        motion_gt = gesture_gt[person].permute(0, 2, 1).unsqueeze(2).to(device, non_blocking=True)
                        cond_[label]['motion_gt'] = motion_gt  # train diffusion with GT motion (and not features) # attention4

                    print(f"gesture_{person}: {cond_['y']['gesture'].shape}")
                    print(f"Seed_{person}: {cond_['y']['seed'].shape}")
                    print(f"Style_id_{person}: {cond_['y']['style'].shape}")
                    print(f"Text_audio_{person}: {cond_['y']['audio'].shape}")

            print(f"Batch keys ...")
            print(cond_.keys())


            """
                Every batch is a collection of #batch_size sample index, from everyone these index is taken a slice of #sequence_length
                features
                eg. if you have 370 video in the h5file, using 128 as batch size you'll have max 3 batch for each epoch, using
                    32 as batch_size you'll have 11 batches for each epoch.
            """

        # NEED TO FURTHER INVESTIGATE IF ALL THE SHAPES ARE GOOD

    key2 = True
    if key2:
        for i in range(len(trn_dataset)):
            print(f"Sample {i}")
            text_audio, gesture, motion, speaker_id = trn_dataset[i]
            serial_printer(text_audio=text_audio,
                           gesture=gesture,
                           motion=motion,
                           speaker_id=speaker_id)
            print('\n')
