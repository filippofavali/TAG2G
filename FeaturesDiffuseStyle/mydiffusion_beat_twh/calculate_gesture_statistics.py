import pdb
import argparse
import h5py
import os
import numpy as np
from pathlib import Path
from pprint import pprint


def main(args):

    dataset = args.split

    if 'dyadic' in args.version:
        version = args.version.replace('_dyadic', '')
    else:
        version = args.version

    h5_pointer = args.h5_file
    assert os.path.isfile(h5_pointer), "Provided h5_pointer is not a dir"
    print(f"Loading {h5_pointer}")

    # open h5 file and save gesture into gesture_trn dict
    h5 = h5py.File(h5_pointer, "r")
    gesture_trn = {}
    for speaker in args.speakers:
        # for key in h5.keys(): print(h5[key][f'{speaker}_gesture'][:].shape)       # debug line
        gesture_trn[f"{speaker}"] = [h5[key][f'{speaker}_gesture'][:] for key in h5.keys()]
    h5.close()

    # print total found clips in H5 file from featurization step
    print("Total trn clips:", len(gesture_trn[args.speakers[0]]))     # Total trn clips: 27

    # compute mean and std for each
    for speaker in args.speakers:
        gesture = np.concatenate(gesture_trn[f"{speaker}"])
        print(f"Gesture trn shape - {speaker}: {gesture.shape}")
        mean_pointer = os.path.join(os.path.dirname(h5_pointer), f"{speaker}-gesture_{dataset}_mean_{version}.npy")
        std_pointer = os.path.join(os.path.dirname(h5_pointer), f"{speaker}-gesture_{dataset}_std_{version}.npy")
        np.save(mean_pointer, np.mean(gesture, axis=0))
        np.save(std_pointer,np.std(gesture, axis=0) + 1e-6)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate_gesture_statistics.py')
    parser.add_argument('--h5_file', type=str, default='')
    parser.add_argument('--split', type=str, default='trn')
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--speakers', type=tuple, default=('main-agent', 'interloctr'))
    args = parser.parse_args()
    main(args)
