import numpy as np
import scipy
from scipy import linalg
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import joblib as jl
from VQVAE.vqvae_utils.motion_utils import load_bvh_file
import warnings


class GestureEvaluator:

    def __init__(self):
        pass

    def __call__(self, gesture, gt_gesture=None, mode='GHL'):

        if mode == 'GHL':
            assert gesture.shape[0] == 74, 'Generated gesture is not a 74,N signal '
            acc, jerk = self.acceleration_jerk(gesture)
            return {'acc': acc,
                    'jerk': jerk}

        elif mode == 'MAA':
            assert gt_gesture.shape[0] == 74, 'Gesture gt is not a 74,N signal'
            assert gesture.shape[0] == 74, 'Generated gesture is not a 74,N signal '
            # Main-agent appropriateness
            ape = self.APE(gt_gesture=gt_gesture, gesture=gesture)
            fdg = self.frechet_distance(gesture=gt_gesture, gen_gesture=gesture)
            s_dtw = self.DTW_distance(gt_gesture=gt_gesture, gesture=gesture)
            s_cov = self.cov_similarity(gt_gesture=gt_gesture, gesture=gesture)
            return{
                'APE': ape,
                'FDG': fdg,
                'S_DTW': s_dtw,
                'S_COV': s_cov
            }

        elif mode == 'INA':
            assert gt_gesture.shape[0] == 74, 'Gesture gt is not a 74,N signal'
            assert gesture.shape[0] == 74, 'Generated gesture is not a 74,N signal '
            # Interlocutor appropriateness
            fdg = self.frechet_distance(gesture=gt_gesture, gen_gesture=gesture)
            s_dtw = self.DTW_distance(gt_gesture=gt_gesture, gesture=gesture)
            s_cov = self.cov_similarity(gt_gesture=gt_gesture, gesture=gesture)
            return {
                'FDG': fdg,
                'S_DTW': s_dtw,
                'S_COV': s_cov
            }

        else:
            raise NotImplementedError(f'{mode} evaluation not implemented yet')

        # print(f"APE distance: {self.APE(gt_gesture, gesture)}")
        # print(f"Frechet distance: {self.frechet_distance(gt_gesture, gesture)}")
        # # print(f"FDG: {get_scores(gesture, gt_gesture)}")
        # print(f"DTW distance: {self.DTW_distance(gt_gesture, gesture)}")
        # self.cov_similarity(gt_gesture, gesture)
        # acc, jerk = self.acceleration_jerk(gt_gesture)
        # print(f'gt acc: {acc}, gt jerk: {jerk}')

    def frechet_distance(self, gesture, gen_gesture, eps=1e-6):

        try:

            mu1 = np.mean(gesture, axis=0)
            mu2 = np.mean(gen_gesture, axis=0)
            sigma1 = np.cov(gesture, rowvar=False)
            sigma2 = np.cov(gen_gesture, rowvar=False)

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
            # providing this with a finite tollerance as np.isfinite does not work out
            if np.max(covmean) > 10:
                msg = ('fid calculation produces singular product; '
                       'adding %s to diagonal of cov estimates') % eps
                # print(msg)
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        except ValueError:

            print('Encountered singularity immaginary component {}'.format(m))
            return 1e+10

    def DTW_distance(self, gt_gesture, gesture):

        # Dynamic Time Warping measurement on similarity between two signals
        gt_gesture, gesture = gt_gesture.transpose(0, 1), gesture.transpose(0, 1)
        distance, _ = fastdtw(gt_gesture, gesture, dist=euclidean)

        return distance

    def cov_similarity(self, gt_gesture, gesture):
        # get covariance matrix from signals
        gt_mean = np.mean(gt_gesture, axis=1, keepdims=True)
        gen_mean = np.mean(gesture, axis=1, keepdims=True)
        T = gt_gesture.shape[1]
        gt_cov = (np.dot(gt_gesture-gt_mean, (gt_gesture-gt_mean).T)/(T-1)).flatten()
        gen_cov = (np.dot(gesture - gen_mean, (gesture - gen_mean).T) / (T - 1)).flatten()
        # compute similarity via covariance descriptors
        c_score = np.dot(gt_cov, gen_cov.T) / (np.linalg.norm(gt_cov)*np.linalg.norm(gen_cov))

        return c_score

    def APE(self, gt_gesture, gesture):
        # MAE implementation
        # distance = np.mean(np.sqrt(np.square(gesture - gen_gesture)), axis=1)
        # distance = np.mean(distance, axis=0)

        # MSE implementation
        distance = np.sum(np.square(gt_gesture - gesture)) / (gesture.shape[0]*gesture.shape[1])

        return distance

    def acceleration_jerk(self, gesture):

        # from full skeleton to sticky-skeleton
        gesture = self.filter_joints(gesture)

        space = gesture[..., 1:] - gesture[..., :-1]
        space = np.sum(np.sqrt(space**2))

        speed = gesture[..., 1:] - gesture[..., :-1]
        acc = speed[..., 1:] - speed[..., :-1]
        jerk = acc[..., 1:] - acc[..., :-1]
        acc = np.sum(np.sqrt(acc ** 2))
        jerk = np.sum(np.sqrt(jerk ** 2))

        return acc/space, jerk/space

    def filter_joints(self, gesture):

        joint_idx = [0, 1, 2, 18, 19, 20, 33, 34, 35, 39, 40, 41, 60, 61, 62, 63, 64, 65]

        mask = np.ones(gesture.shape[0], dtype=bool)
        mask[joint_idx] = False
        gesture[mask, :] = 0

        return gesture


if __name__ == '__main__':

    J, N = 74, 1800

    test1 = False
    if test1:
        # Test if the metrics computation works
        gt_gesture = np.random.rand(J, N)
        gen_gesture = np.random.rand(J, N)
        evaluation = GestureEvaluator()
        # evaluation(gt_gesture, gen_gesture)
        FDG = get_scores(gt_gesture, gen_gesture)

        print(f'This is FDG: {FDG}')

    test2 = False
    if test2:

        angulars = np.linspace(start=0, stop=1.75*np.pi, num=N)

        SHIFT_PARAM = 0.5   # use this to add a phase distance between the two signals

        # Test over time shifted sinusoidal signals
        gt_gesture = np.sin(angulars)
        # gen_gesture = np.sin(angulars-SHIFT_PARAM)
        gen_gesture = np.random.randn(N)

        # # For example, let's consider a polynomial of degree 3: y = ax^3 + bx^2 + cx + d
        # # Generate polynomial signals
        # a, b, c, d = 0.0024, -0.005, 0.002, 0.001  # Example coefficients
        # gt_gesture = a * angulars ** 3 + b * angulars ** 2 + c * angulars + d
        # gen_gesture = a * (angulars - 1) ** 3 + b * (angulars - 1) ** 2 + c * (angulars - 1) + d

        gt_gesture = np.tile(gt_gesture, (J, 1))
        gen_gesture = np.tile(gen_gesture, (J, 1))

        plt.figure(figsize=(10, 6))
        plt.plot(angulars, gt_gesture[0, :], color='b')
        plt.plot(angulars, gen_gesture[0, :], color='r')
        plt.title('Plotting a single joint example')
        # plt.show()

        evaluator = GestureEvaluator()
        evaluator(gt_gesture, gen_gesture)

    test3 = True
    if test3:

        warnings.filterwarnings('ignore')
        # DONE: test with two real samples from the dataset
        pipeline = jl.load(r'C:\Users\faval\PycharmProjects\TAG2G\Utils\pipeline_expmap_74joints.sav')

        gen_gesture = r'C:\Users\faval\PycharmProjects\TAG2G\data\generated_samples\test\dyadic_lean\val_2023_v0_000_main-agent_natural.bvh'
        # gen_gesture = r'C:\Users\faval\genea2023_dataset\val\interloctr\bvh\val_2023_v0_000_interloctr.bvh'
        gen_gesture = load_bvh_file(gen_gesture)
        gen_gesture = pipeline.transform([gen_gesture])
        gen_gesture = np.squeeze(gen_gesture, axis=0).T

        gt_gesture = r'C:\Users\faval\genea2023_dataset\val\main-agent\bvh\val_2023_v0_000_main-agent.bvh'
        gt_gesture = load_bvh_file(gt_gesture)
        gt_gesture = pipeline.transform([gt_gesture])
        gt_gesture = np.squeeze(gt_gesture, axis=0).T

        assert gt_gesture.shape == gen_gesture.shape, print('Gt and Gen have different shapes')
        print(f'gt shape: {gt_gesture.shape} - gen shape: {gen_gesture.shape}')

        evaluator = GestureEvaluator()
        evaluator(gesture=gen_gesture, gt_gesture=gt_gesture)

        # FDG = get_scores(gt_gesture, gen_gesture)
        # print(f'This is FDG: {FDG}')






