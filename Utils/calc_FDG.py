import pdb
import argparse
import h5py
import os
import numpy as np
from pathlib import Path
from pprint import pprint
from scipy import linalg



def main(dataset, version, speakers=('main-agent', 'interloctr')):

    h5_pointer = Path("/home/vik/Desktop/TAG2G/data/val_features")
    assert os.path.isdir(h5_pointer), "Provided h5_pointer is not a dir"
    h5_names = list(h5_pointer.rglob('*.h5'))
    h5_pointer = os.path.join(h5_pointer, h5_names[-1])
    print(f"Loading {h5_pointer}")

    h5_pointer_gen = Path("/home/vik/Desktop/TAG2G/data/gen_features")
    assert os.path.isdir(h5_pointer_gen), "Provided h5_pointer_gen is not a dir"
    h5_names_gen = list(h5_pointer_gen.rglob('*.h5'))
    h5_pointer_gen = os.path.join(h5_pointer_gen, h5_names_gen[-1])
    print(f"Loading {h5_pointer_gen}")


    speaker = 'main-agent'

    # open h5 file and save gesture into gesture_trn dict
    h5 = h5py.File(h5_pointer, "r")
    gesture_trn = {}
    # for key in h5.keys(): print(h5[key][f'{speaker}_gesture'][:].shape)       # debug line
    gesture_trn[f"{speaker}"] = [h5[key][f'{speaker}_gesture'][:] for key in h5.keys()]
    h5.close()
    # print total found clips in H5 file from featurization step
    print("Total trn clips:", len(gesture_trn[speakers[0]]))     # Total trn clips: 27
    gesture_gt = np.concatenate(gesture_trn[f"{speaker}"])
    print(f"Gesture trn shape: {gesture_gt.shape}")

    # open h5 file and save gesture into gesture_trn dict
    h5_gen = h5py.File(h5_pointer_gen, "r")
    gesture_gen = {}
    # for key in h5.keys(): print(h5[key][f'{speaker}_gesture'][:].shape)       # debug line
    gesture_gen[f"{speaker}"] = [h5_gen[key][f'{speaker}_gesture'][:] for key in h5_gen.keys()]
    h5_gen.close()
    # print total found clips in H5 file from featurization step
    print("Total trn clips:", len(gesture_gen[speakers[0]]))     # Total trn clips: 27
    gesture_pred = np.concatenate(gesture_gen[f"{speaker}"])
    print(f"Gesture gen shape: {gesture_pred.shape}")

    fgd_score = get_scores(gesture_pred, gesture_gt)
    print('FGD SCORE:', fgd_score)
    # gesture = np.concatenate(gesture_trn[f"{speaker}"])
    # print(f"Gesture trn shape: {gesture.shape}")
    # # mean_pointer = os.path.join(os.path.dirname(h5_pointer), f"{speaker}-gesture_{dataset}_mean_{version}.npy")
    # # std_pointer = os.path.join(os.path.dirname(h5_pointer), f"{speaker}-gesture_{dataset}_std_{version}.npy")
    # np.save(mean_pointer, np.mean(gesture, axis=0))
    # np.save(std_pointer,np.std(gesture, axis=0) + 1e-6)

def get_scores(generated_feats, real_feats):

    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist

    ####################################################################
    # frechet distance
    frechet_dist = frechet_distance(generated_feats, real_feats)

    ####################################################################
    # # distance between real and generated samples on the latent feature space1    dists = []
    # for i in range(real_feats.shape[0]):
    #     d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
    #     dists.append(d)
    # feat_dist = np.mean(dists)

    # return frechet_dist, feat_dist
    return frechet_dist


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    sing_threeshold = 100
    # if not np.isfinite(covmean).all():
    if np.max(covmean) > sing_threeshold:
        print(f'COVmean max element: {np.max(covmean)}')
        print(f"ADDING OFFSET TO MATRICES COVMEAN THAT IS SINGULAR")
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        print(f'COVmean max element: {np.max(covmean)}')
    # else:
    #     print(f'COVmean max element: {np.max(covmean)}')

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate_gesture_statistics.py')
    parser.add_argument('--dataset', type=str, default='trn')
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--speakers', type=tuple, default=('main-agent', 'interloctr'))
    args = parser.parse_args()
    main(args.dataset, args.version)
