import sys
import os
import h5py
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(script_dir, '..'))
[sys.path.append(dir) for dir in [main_dir, script_dir]]
# print(sys.path)                                                           # Debug line
from Featurization.ProcessAudioData import AudioProcessor
from Featurization.ProcessTextData import TextProcessor
from Featurization.ProcessMotionData import GestureProcessor
from Featurization.Config.FeaturesConfig import FeaturesConfig
import numpy as np
import torch
from argparse import ArgumentParser
from easydict import EasyDict
from tqdm import tqdm
import glob


def define_device(force_cpu=False):

    dev = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print(f"Running on '{dev}'")
    return dev


def load_metadata(metadata, speakers):

    """
    This version is been taken from Genea code and adapted to produce a dyadic version of metadata usage
    # was (metadata, participant)
    """

    assert ("main-agent" in speakers)&("interloctr" in speakers), "`speakers` must be 'main-agent' and 'interloctr'"

    metadict_byfname = {}
    metadict_byindex = {}
    speaker_ids = []
    finger_info = []

    with open(metadata, "r") as f:
        # NOTE: The first line contains the csv header so we skip it
        for i, line in enumerate(f.readlines()[1:]):
            (
                fname,
                main_speaker_id,
                main_has_finger,
                ilocutor_speaker_id,
                ilocutor_has_finger,
            ) = line.strip().split(",")

            """
            if participant == "main-agent":
                has_finger = (main_has_finger == "finger_incl")
                speaker_id = int(main_speaker_id) - 1
            else:
                has_finger = (ilocutor_has_finger == "finger_incl")
                speaker_id = int(ilocutor_speaker_id) - 1
            """
            has_finger = (main_has_finger == "finger_incl", ilocutor_has_finger == "finger_incl")
            speaker_id = (int(main_speaker_id) - 1, int(ilocutor_speaker_id) - 1)

            finger_info.append(has_finger)
            speaker_ids.append(speaker_id)

            metadict_byindex[i] = has_finger, speaker_id
            metadict_byfname[fname] = has_finger, speaker_id

    speaker_ids = np.array(speaker_ids)
    finger_info = np.array(finger_info)
    # num_speakers = np.unique(speaker_ids).shape[0]            # not used in code, not implemented in dyadic version
    # assert num_speakers == spks.max(), "Error speaker info!"
    # print("Number of speakers: ", num_speakers)
    # print("Has Finger Ratio:", np.mean(finger_info))

    return metadict_byfname, metadict_byindex                   # num_speaker not returned


def TAG_featuring(fparams):

    speaker_dict = {
        "main-agent": 0,
        "interloctr": 1,
    }

    assert os.path.isdir(fparams.data_root), f"Source data directory does not exist"

    # retrieving featurization parameters and checking valid conditions
    speakers = fparams.speakers
    assert speakers == ["main-agent", "interloctr"], "Only main-agent and interloctr"
    dataset_root = fparams.data_root
    dataset_type = next((type for type in ["trn", "val", "tst"] if type in dataset_root), "not_defined")
    print(f"Processing {dataset_type} dataset")
    assert dataset_type != "not_defined", f"Provided a non valid dataset_root: trn, val, tst"
    win_len = fparams.vqvae_params["motion_window_length"]


    # TODO: Relative reference in this path need to be related to os.getcwd()
    # TODO: Do I really need this? is it preload that usefull?
    # TODO: If one of the paths is not a dir --> then preload = false because we have to

    # Define output directories to store files
    cwd = os.getcwd()
    features_root = os.path.join(cwd, fparams.features_root, f"{dataset_type}_features_{fparams.description}")
    if not os.path.isdir(features_root):
        os.makedirs(features_root)

    text_save_path = os.path.join(cwd, features_root, "text")
    audio_save_path = os.path.join(cwd, features_root, "audio")
    gesture_save_path = os.path.join(features_root, "gesture")
    motion_save_path = os.path.join(features_root, "gestureGT")

    if fparams.store_npy:
        for path in [features_root, text_save_path, audio_save_path, gesture_save_path, motion_save_path]:
            if not os.path.exists(path):
                # make the dir but also set preload to false - if dir does not exist no file to load ^_^
                os.makedirs(path)
                fparams.preload = False

    metadata_path = os.path.join(dataset_root, "metadata.csv")
    metadict_byfname, metadict_byindex = load_metadata(metadata_path, speakers)
    filenames = sorted(metadict_byfname.keys())

    if fparams.debug:
        all_filenames = ["trn_2023_v0_000", "trn_2023_v0_001", "trn_2023_v0_002"]
    else:
        all_filenames = filenames

    if not fparams.preload:
        # Multimodal processing instantiation
        text_processor = TextProcessor(fparams=fparams)
        audio_processor = AudioProcessor(fparams=fparams)
        gesture_processor = GestureProcessor(fparams=fparams)

    h5_dump_pointer = os.path.join(features_root, f"TWH-{dataset_type}_{fparams.description}.h5")
    with h5py.File(h5_dump_pointer, "w") as h5:

        progress_bar = tqdm(all_filenames, desc="Extracting features")
        for i, filename in enumerate(progress_bar):

            progress_bar.set_description(f"Processing {i+1}/{len(all_filenames)}: '{filename}'")
            g_data = h5.create_group(str(i))
            has_finger, speaker_id = metadict_byfname[filename]

            # For every file a tuple with both main-agent and interloctr data in it --> used in h5 group.create_dataset
            gesture_features = []
            gesture = []
            audio_features = []
            text_features = []

            """
            if preload --> h5_build over the text, audio, gesture npy features
            if not preload --> from tsv, wav, bvh to text, audio, gesture features
            """

            if not fparams.preload:
                # have to do both for "main-agent" and "interloctr"
                for participant in speakers:

                    bvh_path = os.path.join(dataset_root, participant, "bvh", f"{filename}_{participant}.bvh")
                    wav_path = os.path.join(dataset_root, participant, "wav", f"{filename}_{participant}.wav")
                    tsv_path = os.path.join(dataset_root, participant, "tsv", f"{filename}_{participant}.tsv")

                    # process gesture
                    # TODO:  Throws error with interloctr - tst dataset (no main-agent tst)
                    if dataset_type == "trn" or dataset_type == "val":
                        # DONE: if features already exist and preload = True
                        if os.path.isfile(os.path.join(gesture_save_path,f"{filename}_{participant}.npy")) and fparams.preload:
                            print(
                                f"Gesture features '{filename}_{participant}' already exist - Preload:{fparams.preload}")
                            gesture_features_npy = np.load(os.path.join(gesture_save_path, f"{participant}_{filename}.npy"))
                            gesture_npy = np.load(os.path.join(motion_save_path, f"{participant}_{filename}.npy"))
                        else:
                            dump_pipeline = (filename == 'trn_2023_v0_002')
                            gesture_features_npy, gesture_npy, _ = gesture_processor(bvh_path, dump_pipeline=dump_pipeline)
                            if fparams.store_npy:
                                np.save(os.path.join(gesture_save_path, f"{filename}_{participant}.npy"), gesture_features_npy)
                                np.save(os.path.join(motion_save_path, f"{filename}_{participant}.npy"), gesture_npy)

                        gesture_features.append(gesture_features_npy)
                        gesture.append(gesture_npy)

                    # process audio
                    if os.path.exists(os.path.join(audio_save_path, f"{filename}_{participant}.npy")) and fparams.preload:
                        print(f"Audio features '{filename}_{participant}' already exist - Preload:{fparams.preload}")
                        audio_features_npy = np.load(os.path.join(audio_save_path, f"{participant}_{filename}.npy"))
                    else:
                        audio_features_npy = audio_processor(wav_path)
                        if fparams.store_npy:
                            np.save(os.path.join(audio_save_path, f"{participant}_{filename}.npy"), audio_features_npy)
                    audio_features.append(audio_features_npy)

                    # process text
                    if os.path.exists(os.path.join(text_save_path, f"{filename}_{participant}.npy")) and fparams.preload:
                        print(f"Text fetures '{filename}_{participant}' already exist - Preload:{fparams.preload}")
                        text_features_npy = np.load(os.path.join(text_save_path, f"{filename}_{participant}.npy"))
                    else:
                        clip_len = min(audio_features_npy.shape[0], gesture_npy.shape[0])
                        print(f"Clipping text length at {clip_len}")
                        text_features_npy = text_processor(tsv_path, crop_length=clip_len)
                        if fparams.store_npy:
                            np.save(os.path.join(text_save_path, f"{filename}_{participant}.npy"), text_features_npy)
                    text_features.append(text_features_npy)

            elif fparams.preload:

                for participant in speakers:

                    if dataset_type == "trn" or dataset_type == "val":
                        # process gesture only in train or validation (inference time in tst doesn't require gesture)
                        gesture_features_npy = np.load(os.path.join(gesture_save_path, f"{filename}_{participant}.npy"))
                        gesture_npy = np.load(os.path.join(motion_save_path, f"{filename}_{participant}.npy"))
                        gesture_features.append(gesture_features_npy)
                        gesture.append(gesture_npy)

                    audio_features_npy = np.load(os.path.join(audio_save_path, f"{filename}_{participant}.npy"))
                    audio_features.append(audio_features_npy)
                    text_features_npy = np.load(os.path.join(text_save_path, f"{filename}_{participant}.npy"))
                    text_features.append(text_features_npy)

            # TODO: Need to see how to do if someone wants to run the tst dataset -- online pipeline ??
            # then build the h5 file to run diffusion model on, only for trn and val data
            if dataset_type == "trn" or dataset_type == "val":

                # preprocess the length of each data for each participant
                min_clip_len = []
                for participant, id in speaker_dict.items():

                    clip_len = min(gesture_features[id].shape[0] * win_len,
                                   gesture[id].shape[0],
                                   audio_features[id].shape[0],
                                   text_features[id].shape[0])
                    min_clip_len.append(clip_len)

                    # clipping all the data to the shortest signal
                    gesture_features[id] = gesture_features[id][:(clip_len//win_len), ...]
                    gesture[id] = gesture[id][:clip_len, ...]
                    audio_features[id] = audio_features[id][:clip_len, ...]
                    text_features[id] = text_features[id][:clip_len, ...]

                assert min_clip_len[0] == min_clip_len[1], \
                    f"Found discrepancy between main-agent and interloctr data length at '{filename}'"

                # g_data.create_dataset("has_finger", data=[has_finger])            # fingers not used
                for speaker in speakers:
                    g_data.create_dataset(f"{speaker}_id", data=[speaker_id[speaker_dict[f'{speaker}']]])

                # from here text and audio are going to be a stack of t, t+18 samples to compare to gesture features
                for participant, id in speaker_dict.items():
                    print(f"Participant id {id+1}/{len(speaker_dict.keys())}: '{filename}'")

                    print(f"Processing text, audio in total windows: {gesture_features[id].shape[0]}")

                    audio_data = [np.array([audio_features[id][feat_idx * win_len: feat_idx * win_len + win_len, ...]]) \
                                  for feat_idx in range(gesture_features[id].shape[0])]
                    audio_data = np.concatenate(audio_data)

                    text_data = [np.array([text_features[id][feat_idx * win_len: feat_idx * win_len + win_len, ...]]) \
                                 for feat_idx in range(gesture_features[id].shape[0])]
                    text_data = np.concatenate(text_data)

                    g_data.create_dataset(f"{participant}_gesture", data=gesture_features[id], dtype=np.float32)
                    g_data.create_dataset(f"{participant}_GT", data=gesture[id], dtype=np.float32)
                    g_data.create_dataset(f"{participant}_audio", data=audio_data, dtype=np.float32)
                    g_data.create_dataset(f"{participant}_text", data=text_data, dtype=np.float32)


if __name__ == "__main__":

    from pprint import pprint

    # DONE: Clean current ***t implementation
    # DONE: Add WavLM when encoding audio
    # DONE: Argument parser - add also 'force_cpu' and mix with fparams
    # TODO: decide about generation of tst dataset - does it has to happen online (so discard usage with tst)

    parser = ArgumentParser()
    parser.add_argument("--data_source", type=str, help="train root with text, audio, gesture dir")
    parser.add_argument("--force_cpu", action='store_true')
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--store_npy', action='store_true')
    args = parser.parse_args()
    fparams = FeaturesConfig()                     # parameters to be used in features embedding

    # Update featurization parameters with arg parser inputs
    fparams['device'] = define_device(force_cpu=args.force_cpu)
    for key, value in vars(args).items():
        fparams[key] = value

    print("Printing featurization parameters ...")
    pprint(fparams)

    fparams = EasyDict(fparams)

    # DONE: test functionality of the above procedure -Working 17/05/24
    TAG_featuring(fparams)