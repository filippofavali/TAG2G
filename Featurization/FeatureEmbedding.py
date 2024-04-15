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
from Featurization.Config.FeaturesConfig import features_config

import numpy as np
import torch

from argparse import ArgumentParser
import glob


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


def TAG_test_featuring(fparams, device, debug):

    speaker_dict = {
        "main-agent": 0,
        "interloctr": 1,
    }

    speakers = fparams.speakers
    dataset_root = fparams.dataset_root
    win_len = fparams.vqvae_params["motion_window_length"]

    """
    dataset_type = "not_defined"
    for type in ["trn", "val", "tst"]:
        if type in dataset_root:
            dataset_type = type
            break
    """

    dataset_type = next((type for type in ["trn", "val", "tst"] if type in dataset_root), "not_defined")
    print(f"Processing {dataset_type} dataset")

    # define features dir and make it if it doesn't exist
    assert speakers == ["main-agent", "interloctr"], "Only main-agent and interloctr"

    features_root = os.path.join(fparams.features_root, f"{dataset_type}_features")
    text_save_path = os.path.join(features_root, "text")
    audio_save_path = os.path.join(features_root, "audio")
    gesture_save_path = os.path.join(features_root, "gesture")


    for path in [features_root, text_save_path, audio_save_path, gesture_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    metadata_path = os.path.join(dataset_root, "metadata.csv")
    metadict_byfname, metadict_byindex = load_metadata(metadata_path, speakers)
    filenames = sorted(metadict_byfname.keys())

    if debug:
        all_filenames = [f"{dataset_type}_2023_v0_000", f"{dataset_type}_2023_v0_001", f"{dataset_type}_2023_v0_002"]
    else:
        all_filenames = filenames

    # loading method processors to process and featurize data from dataset
    text_processor = TextProcessor(word2vec_dir=fparams.word2vec_dir)
    audio_processor = AudioProcessor(audio_parameters=fparams.audio_parameters)
    gesture_processor = GestureProcessor(fparams=fparams, device=device)

    for i, filename in enumerate(all_filenames):
        print(f"Processing {i + 1}/{len(filenames)}: '{filename}'", end="\r")

        if not preload:

            # have to do both for "main-agent" and "interloctr"
            for participant in speakers:

                bvh_path = os.path.join(dataset_root, participant, "bvh", f"{filename}_{participant}.bvh")
                wav_path = os.path.join(dataset_root, participant, "wav", f"{filename}_{participant}.wav")
                tsv_path = os.path.join(dataset_root, participant, "tsv", f"{filename}_{participant}.tsv")

                # process gesture
                if dataset_type == "trn" or dataset_type == "val":

                    if os.path.isfile(os.path.join(gesture_save_path, f"{filename}_{participant}.npy")):
                        print(f"'{filename}' gesture already exist")
                        gesture_features_npy = np.load(os.path.join(gesture_save_path, f"{filename}_{participant}.npy"))
                        print(f"Loaded gesture: {gesture_features_npy.shape}")
                    else:
                        dump_pipeline = (filename == 'trn_2023_v0_002')
                        gesture_features_npy, gesture_npy, _ = gesture_processor(bvh_path, dump_pipeline=dump_pipeline)
                        np.save(os.path.join(gesture_save_path, f"{filename}_{participant}.npy"), gesture_features_npy)

                # process audio
                if os.path.exists(os.path.join(audio_save_path, f"{filename}_{participant}.npy")):
                    print(f"'{filename}' audio already exist")
                    # load audio as needed for text clip length
                    audio_features_npy = np.load(os.path.join(audio_save_path, f"{filename}_{participant}.npy"))
                    print(f"Loaded audio: {audio_features_npy.shape}")
                else:
                    audio_features_npy = audio_processor(wav_path)
                    np.save(os.path.join(audio_save_path, f"{filename}_{participant}.npy"), audio_features_npy)

                # process text
                if os.path.exists(os.path.join(text_save_path, f"{filename}_{participant}.npy")):
                    print(f"{filename} text already exist")
                    text_features_npy = np.load(os.path.join(text_save_path, f"{filename}_{participant}.npy"))
                    print(f"Loaded audio: {text_features_npy.shape}")
                else:
                    clip_len = min(audio_features_npy.shape[0], gesture_npy.shape[0])
                    print(f"Clipping text length at {clip_len}")
                    text_features_npy = text_processor(tsv_path, crop_length=clip_len)
                    np.save(os.path.join(text_save_path, f"{filename}_{participant}.npy"), text_features_npy)


def TAG_featuring(fparams, device, preload, debug):

    speaker_dict = {
        "main-agent": 0,
        "interloctr": 1,
    }

    speakers = fparams.speakers
    dataset_root = fparams.dataset_root
    win_len = fparams.vqvae_params["motion_window_length"]

    """
    dataset_type = "not_defined"
    for type in ["trn", "val", "tst"]:
        if type in dataset_root:
            dataset_type = type
            break
    """

    dataset_type = next((type for type in ["trn", "val", "tst"] if type in dataset_root), "not_defined")
    print(f"Processing {dataset_type} dataset")

    # define features dir and make it if it doesn't exist
    # assert ("trn" or "val" or "tst") in dataset_root, "Only trn, val, tst root permitted"
    assert speakers == ["main-agent", "interloctr"], "Only main-agent and interloctr"

    features_root = os.path.join(fparams.features_root, f"{dataset_type}_features")
    text_save_path = os.path.join(features_root, "text")
    audio_save_path = os.path.join(features_root, "audio")
    gesture_save_path = os.path.join(features_root, "gesture")
    motion_save_path = os.path.join(features_root, "gestureGT")

    for path in [features_root, text_save_path, audio_save_path, gesture_save_path, motion_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    metadata_path = os.path.join(dataset_root, "metadata.csv")
    metadict_byfname, metadict_byindex = load_metadata(metadata_path, speakers)
    filenames = sorted(metadict_byfname.keys())

    if debug:
        all_filenames = ["trn_2023_v0_000", "trn_2023_v0_001", "trn_2023_v0_002"]
    else:
        all_filenames = filenames

    # loading method processors to process and featurize data from dataset
    if not preload:
        text_processor = TextProcessor(word2vec_dir=fparams.word2vec_dir)
        audio_processor = AudioProcessor(audio_parameters=fparams.audio_parameters)
        gesture_processor = GestureProcessor(fparams=fparams, device=device)

    h5_dump_pointer = os.path.join(features_root, f"TWH-{dataset_type}_v0_.h5")
    with h5py.File(h5_dump_pointer, "w") as h5:

        for i, filename in enumerate(all_filenames):
            print(f"Processing {i+1}/{len(filenames)}: '{filename}'", end="\r")
            g_data = h5.create_group(str(i))
            has_finger, speaker_id = metadict_byfname[filename]

            """
            if preload --> h5_build over the text, audio, gesture npy features
            if not preload --> from tsv, wav, bvh to text, audio, gesture features
            """

            # For every file a tuple with both main-agent and interloctr data in it --> used in h5 group.create_dataset
            gesture_features = []
            gesture = []
            audio_features = []
            text_features = []

            if not preload:

                # have to do both for "main-agent" and "interloctr"
                for participant in speakers:

                    bvh_path = os.path.join(dataset_root, participant, "bvh", f"{filename}_{participant}.bvh")
                    wav_path = os.path.join(dataset_root, participant, "wav", f"{filename}_{participant}.wav")
                    tsv_path = os.path.join(dataset_root, participant, "tsv", f"{filename}_{participant}.tsv")

                    # process gesture
                    if dataset_type == "trn" or dataset_type == "val":
                        if not os.path.isfile(os.path.join(gesture_save_path,f"{participant}_{filename}.npy")):
                            dump_pipeline = (filename == 'trn_2023_v0_002')
                            gesture_features_npy, gesture_npy, _ = gesture_processor(bvh_path, dump_pipeline=dump_pipeline)
                            np.save(os.path.join(gesture_save_path, f"{participant}_{filename}.npy"), gesture_features_npy)
                            np.save(os.path.join(motion_save_path, f"{participant}_{filename}.npy"), gesture_npy)
                        else:
                            gesture_features_npy = np.load(os.path.join(gesture_save_path, f"{participant}_{filename}.npy"))
                            gesture_npy = np.load(os.path.join(motion_save_path, f"{participant}_{filename}.npy"))

                        gesture_features.append(gesture_features_npy)
                        gesture.append(gesture_npy)

                    # process audio
                    if os.path.exists(os.path.join(audio_save_path, f"{participant}_{filename}.npy")):
                        print(f"'{filename}' audio already exist")
                        audio_features_npy = np.load(os.path.join(audio_save_path, f"{participant}_{filename}.npy"))
                    else:
                        audio_features_npy = audio_processor(wav_path)
                        if dataset_type == "trn" or dataset_type == "val":
                            np.save(os.path.join(audio_save_path, f"{participant}_{filename}.npy"), audio_features_npy)
                        else:
                            np.save(os.path.join(audio_save_path,
                                                 f"{participant}_{filename}_{speaker_id[speaker_dict[participant]]}.npy"),
                                    audio_features_npy)

                    audio_features.append(audio_features_npy)

                    # process text
                    if os.path.exists(os.path.join(text_save_path, f"{participant}_{filename}.npy")):
                        print(f"{filename} text already exist")
                        text_features_npy = np.load(os.path.join(text_save_path, f"{participant}_{filename}.npy"))
                    else:
                        clip_len = min(audio_features_npy.shape[0], gesture_npy.shape[0])
                        print(f"Clipping text length at {clip_len}")
                        text_features_npy = text_processor(tsv_path, crop_length=clip_len)
                        np.save(os.path.join(text_save_path, f"{participant}_{filename}.npy"), text_features_npy)

                    text_features.append(text_features_npy)

            else:

                for participant in speakers:

                    if dataset_type == "trn" or dataset_type == "val":
                        # process gesture only in train or validation (inference time in tst doesn't require gesture)
                        gesture_features_npy = np.load(os.path.join(gesture_save_path, f"{participant}_{filename}.npy"))
                        gesture_npy = np.load(os.path.join(motion_save_path, f"{participant}_{filename}.npy"))
                        gesture_features.append(gesture_features_npy)
                        gesture.append(gesture_npy)

                    audio_features_npy = np.load(os.path.join(audio_save_path, f"{participant}_{filename}.npy"))
                    audio_features.append(audio_features_npy)
                    text_features_npy = np.load(os.path.join(text_save_path, f"{participant}_{filename}.npy"))
                    text_features.append(text_features_npy)

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

    """
    parser = ArgumentParser()
    parser.add_argument("--train_root", type=str, help="train root with text, audio, gesture dir")
    parser.add_argument("--vqvae_dir", type=str)
    parser.add_argument("--crawl_dir", type=str)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    train_root = args.train_root
    vqvae_dir = args.vqvae_dir
    word2vec_dir = args.crawl_dir
    """

    # settings to run the code from pycharm -->
    preload = False
    debug = False
    test_data = False

    ## then cancel until here <--

    fparams = features_config()                                     # parameters to be used in features embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    if test_data:
        TAG_test_featuring(fparams, device, debug=debug)
    else:
        TAG_featuring(fparams, device, preload=preload, debug=debug)

    # then I need something that start to window with window length of 30 frames or so
    # probably motion has to master the embedding and then the other two coming