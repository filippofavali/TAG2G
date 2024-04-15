import pdb
import argparse
import h5py
import os
import numpy as np
from pathlib import Path
from pprint import pprint


def main(dataset, version, speakers=('main-agent', 'interloctr')):

    h5_pointer = Path("C:/Users/faval/PycharmProjects/TAG2G/data/trn_features")
    assert os.path.isdir(h5_pointer), "Provided h5_pointer is not a dir"
    h5_names = list(h5_pointer.rglob('*.h5'))
    h5_pointer = os.path.join(h5_pointer, h5_names[-1])
    print(f"Loading {h5_pointer}")

    # open h5 file and save gesture into gesture_trn dict
    h5 = h5py.File(h5_pointer, "r")
    gesture_trn = {}
    for speaker in speakers:
        # for key in h5.keys(): print(h5[key][f'{speaker}_gesture'][:].shape)       # debug line
        gesture_trn[f"{speaker}"] = [h5[key][f'{speaker}_gesture'][:] for key in h5.keys()]
    h5.close()

    # print total found clips in H5 file from featurization step
    print("Total trn clips:", len(gesture_trn[speakers[0]]))     # Total trn clips: 27

    # compute mean and std for each
    for speaker in speakers:
        gesture = np.concatenate(gesture_trn[f"{speaker}"])
        print(f"Gesture trn shape: {gesture.shape}")
        mean_pointer = os.path.join(os.path.dirname(h5_pointer), f"{speaker}-gesture_{dataset}_mean_{version}.npy")
        std_pointer = os.path.join(os.path.dirname(h5_pointer), f"{speaker}-gesture_{dataset}_std_{version}.npy")
        np.save(mean_pointer, np.mean(gesture, axis=0))
        np.save(std_pointer,np.std(gesture, axis=0) + 1e-6)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate_gesture_statistics.py')
    parser.add_argument('--dataset', type=str, default='trn')
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--speakers', type=tuple, default=('main-agent', 'interloctr'))
    args = parser.parse_args()
    main(args.dataset, args.version)
