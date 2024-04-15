from VQVAE.vqvae_utils.utils import *
from VQVAE.system import VQVAESystem
from Config.FeaturesConfig import features_config
import os
import torch
import numpy as np
import joblib as jl

"""
Should be usefull in two case scenario:
    1) bvh to npy using a given pipeline (need to be defined which one)
    2) npy to trained VQVAE latent feature gestures 
"""


class GestureProcessor:

    def __init__(self, fparams, device):

        self.fparams = fparams
        self.device = device
        self.expmap_pipeline = create_TAG2G_pipeline()
        self.vqvae = self.load_vqvae()
        self.codebook = self.vqvae.vqvae.vq.embedding.weight
        self.win_len = self.fparams.vqvae_params["motion_window_length"]

    def __call__(self, sample, dump_pipeline=False):

        print(f"Motion working '{sample}'")

        # loading bvh and processing it through the pipeline --> ndarray [T, 74]
        assert os.path.isfile(sample), "Provided __call__ with sample that is not a file"
        gesture_npy = process_pipeline(load_bvh_file(sample), pipeline=self.expmap_pipeline)

        if dump_pipeline:
            # save the pipeline in the given directory
            assert os.path.isdir(self.fparams.dump_pipeline_dir), "Provided dump_pipe_dir is not a dir"
            dump_pipeline_dir = os.path.join(self.fparams.dump_pipeline_dir, f"expmap-74j_TAG2G_{os.path.basename(sample).replace('.bvh', '.sav')}")
            jl.dump(self.expmap_pipeline, dump_pipeline_dir)

        # sorting data in a torchDouble batch
        total_windows = gesture_npy.shape[0] // self.win_len                  # max number of windows to encode the sample
        gesture_features_npy = [gesture_npy[n_win*self.win_len : n_win*self.win_len + self.win_len, ...].reshape(1, self.win_len, gesture_npy.shape[1])
                                for n_win in range(total_windows)]
        gesture_features_npy = np.concatenate(gesture_features_npy, axis=0)         # total_windows, 18, 54
        # test_batch = gesture_features_npy                                   # batch test
        gesture_features_npy = torch.tensor(gesture_features_npy, dtype=torch.double).to(self.device)
        gesture_features_npy = gesture_features_npy.permute(0, 2, 1)    # needs to be total_window, 54, 18 otherwise won't fit the model

        # retrieving closer codebook entry for each gesture window from learnt codebook
        with torch.no_grad():

            gesture_features_npy = self.vqvae .vqvae.encoder(gesture_features_npy)
            vq_vector = self.vqvae.vqvae.vq(gesture_features_npy, training=False)
            distances = torch.sum(gesture_features_npy ** 2, dim=1, keepdim=True) + \
                        torch.sum(self.codebook ** 2, dim=1) - 2 * \
                        torch.matmul(gesture_features_npy, self.codebook.t())
            codebook_ind = torch.argmin(distances, dim=-1).cpu().numpy()
            gesture_features_npy = self.codebook[codebook_ind].cpu().numpy()

        # return gesture_features_npy, gesture_npy, vq_vector[2], test_batch  # batch test
        return gesture_features_npy, gesture_npy, vq_vector[2]

    def load_vqvae(self):

        # need to implement something to load trained model from VQVAE directory

        checkpoint = self.fparams.vqvae_checkpoint
        assert os.path.isdir(checkpoint), f"Invalid checkpoint: '{checkpoint}' is not a dir"

        # load last VQVAE checkpoint "last."
        checkpoint = os.path.join(checkpoint, os.listdir(checkpoint)[-1])

        vqvae_params = self.fparams.vqvae_params
        model = VQVAESystem(num_embeddings=vqvae_params["num_emb"],
                            embedding_dim=vqvae_params["emb_dim"],
                            input_dim=vqvae_params["input_dim"],
                            hidden_dim=vqvae_params["hidden_dim"],
                            max_frames=vqvae_params["motion_window_length"])
        print(f"Loading vqvae_cinematic parameters from '{checkpoint}'")
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        model.to(self.device)

        return model


if __name__ == "__main__":

    key1 = True  # use this to test load hparams, load selected model and so on from feature config
    if key1:
        fparams = features_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        bvh_dir = r"C:\Users\faval\genea2023_dataset\trn\main-agent\bvh"
        bvh_file = os.path.join(bvh_dir, os.listdir(bvh_dir)[1])

        gesture_processor = GestureProcessor(device=device,
                                             fparams=fparams)

        # features, ground_truth, list_of_codebooks, test_batch = gesture_processor(bvh_file, dump_pipeline=True)  # batch_test
        features, ground_truth, list_of_codebooks = gesture_processor(bvh_file, dump_pipeline=True)

        print(f"Shape of input: {ground_truth.shape}")
        print(f"Shape of features: {features.shape}\n type {type(features)}")

        """
        All test passed on 15-12-23 -- you have to uncomment lines with 'batch_test' comment to test coherence of batch
        
        random_number = 3
        ground_truth = ground_truth[18*random_number : 18*random_number+18, ...]
        print(ground_truth.shape)
        test_batch = test_batch[random_number].reshape(test_batch.shape[1], test_batch.shape[2])
        print(test_batch.shape)
        assert ground_truth.all() == test_batch.all(), "not the same slice of data"  
        
        """




