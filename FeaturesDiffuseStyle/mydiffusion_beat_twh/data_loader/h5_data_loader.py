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


class SpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, style_dim, sequence_length=5, npy_root="../../process",
                 dataset="trn", version='v0', speakers=('main-agent', 'interloctr'), motion_window_length=18):

        self.motion_window_length = motion_window_length
        self.motion_dim = motion_dim
        self.dataset = dataset
        self.style_dim = style_dim
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
                    stat_pointer = os.path.join(self.stat_dir, f'{speaker}-gesture_{self.dataset}_{stat}_{self.version}.npy')
                    assert os.path.isfile(stat_pointer), f"Provided stat_pointer '{stat_pointer}' is not a file"
                    self.stats[f"{speaker}_{stat}"] = np.load(stat_pointer)

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

        self.sequence_length = sequence_length    # <-- need to reason about this ...

        print(f"Total clips: {(self.len)}")
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.gesture[self.speakers[0]])            # return length of 'main-agent' features vector

    def __getitem__(self, idx):
        """
        :param idx: an int between 0 and len(gesture['main-agent']), dataset length
        :return: 4 dicts (text&audio as conditions, gesture, motion, speakers ID)
        """
        total_frame_len = self.gesture[f"{self.speakers[0]}"][idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length

        # return 4 dicts, with data of both the speakers into them
        text_audio = {}
        gesture = {}
        motion = {}
        speaker_id = {}

        for speaker in self.speakers:
            # returning text and audio features
            audio = self.audio[f"{speaker}"][idx][start_frame:end_frame]
            text = self.text[f"{speaker}"][idx][start_frame:end_frame]
            textaudio = np.concatenate((audio, text), axis=-1)
            text_audio[f"{speaker}"] = torch.FloatTensor(textaudio)
            # returning gesture features
            pos = self.gesture[f"{speaker}"][idx][start_frame:end_frame]
            if "v0" in self.version:
                vel = self.gesture_vel[f"{speaker}"][idx][start_frame:end_frame]
                acc = self.gesture_acc[f"{speaker}"][idx][start_frame:end_frame]
                gesture[f"{speaker}"] = torch.FloatTensor(np.concatenate((pos, vel, acc), axis=-1))
            else:
                gesture[f"{speaker}"] = torch.FloatTensor(pos)
            # returning ID
            id = np.zeros([self.style_dim])
            id[self.id[f"{speaker}"][idx]] = 1
            speaker_id[f"{speaker}"] = torch.FloatTensor(id)
            # returning ground truth motion that is a window of 18 frames for each
            shift = self.motion_window_length
            motion[f"{speaker}"] = torch.FloatTensor(self.motion[f"{speaker}"][idx][start_frame*shift:end_frame*shift])

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
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset(h5file="../../../data/trn_features/TWH-trn_v0_nocinematic.h5",
                                       motion_dim=256,
                                       style_dim=17,
                                       sequence_length=8)

    train_loader = DataLoader(trn_dataset, num_workers=4,
                              sampler=SequentialSampler(0, len(trn_dataset)),
                              batch_size=32,
                              pin_memory=True,
                              drop_last=False)

    for batch_i, batch in enumerate(train_loader, 0):
        # textaudio, gesture, speaker = batch     # (128, 8, 18, 302+108), (128, 8, 256), (128, 8*18, 74)
        print(batch_i)
        # pdb.set_trace()

        if batch_i == 0:

            # testing what happens during training_loop.run_loop(self) called from end2end

            n_seed = 1
            device = "cuda"

            cond_ = {'y': {}}

            text_audio, gesture, gesture_gt, style = batch
            motion = gesture["main-agent"]
            wavlm = text_audio["main-agent"]
            style = style["main-agent"]
            motion_gt = gesture_gt["main-agent"]

            cond_['y']['seed'] = motion[:, 0:n_seed, :]
            cond_['y']['style'] = style.to(device, non_blocking=True)
            cond_['y']['audio'] = wavlm.to(torch.float32)[:, n_seed:].to(device, non_blocking=True)  # attention4

            print(f"Gesture_features: {motion.shape}")
            print(f"Gesture ground_truth: {motion_gt.shape}")
            print(f"Seed cond_: {cond_['y']['seed'].shape}")
            print(f"Style cond_: {cond_['y']['style'].shape}")
            print(f"Text_audio cond_: {cond_['y']['audio'].shape}")


        """
            Every batch is a collection of #batch_size sample index, from everyone these index is taken a slice of #sequence_length
            features
            eg. if you have 370 video in the h5file, using 128 as batch size you'll have max 3 batch for each epoch, using
                32 as batch_size you'll have 11 batches for each epoch.
        """

        # NEED TO FURTHER INVESTIGATE IF ALL THE SHAPES ARE GOOD