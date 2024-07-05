"""
    This provides an object to be called in h5sincro.py to sincronize all input features and
    to generate a features wide data before diffusion model
"""


class FeaturesConfig(dict):

    def __init__(self, **kwargs):
        # Initialize the base dictionary
        super().__init__()

        # Define the default parameters
        self.default_parameters = {
            # general parameters
            "description": "",
            "data_root": r"../../../genea2023_dataset/val",
            "features_root": r"../data",
            "speakers": ["main-agent", "interloctr"],

            # text specs
            "word2vec_dir": r"../crawl-300d-2M.vec",

            # Audio specs
            "audio_parameters": {
                "NFFT": 4096,
                "MFCC_INPUTS": 40,  # how many parameters will store for each MFCC vector
                "HOP_LENGTH": 1 / 30,
                "DIM": 64,
                "WavLM": r'../WavLM/WavLM-Large.pt'
            },

            # Gesture specs (from VQVAE trained model)
            "vqvae_checkpoint": r"../VQVAE/results/vqvae_cinematic",
            "vqvae_params": {
                "num_emb": 2048,
                "emb_dim": 256,
                "input_dim": 74,
                "hidden_dim": 1024,
                "motion_window_length": 18
            },
            "dump_pipeline_dir": r"../Utils",
        }

        # Update the default parameters with any provided in kwargs
        self.default_parameters.update(kwargs)

        # Set the dictionary with the updated parameters
        self.update(self.default_parameters)

    def __call__(self):
        return self


if __name__ == '__main__':

    # Testing usage and operator access with EasyDict - working 17-05-24

    from pprint import pprint
    from easydict import EasyDict
    # Usage test
    config = FeaturesConfig(dataset_root=r"new\path\to\dataset")
    config = EasyDict(config)
    # pprint(config())

    print(config.audio_parameters)



