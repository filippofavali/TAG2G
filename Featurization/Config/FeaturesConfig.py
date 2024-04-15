"""
    This provides an object to be called in h5sincro.py to sincronize all input features and
    to generate a features wide data before diffusion model
"""


class features_config(object):

    def __init__(self, **kwargs):

        # INSERT HERE PREVIOUS KNOW CONSTANT PARAMETERS TO THE NET
        self.__dict__ = self.features_config(**kwargs)

    def __call__(self):
        return self.__dict__

    def features_config(self, **kwargs):

        parameters_dict = {

            # general parameters
            "dataset_root": r"C:\Users\faval\genea2023_dataset\val",
            "features_root": r"C:\Users\faval\PycharmProjects\TAG2G\data",
            "speakers": ["main-agent", "interloctr"],

            # text specs
            "word2vec_dir": r"C:\Users\faval\PycharmProjects\TAG2G\crawl-300d-2M.vec",          # repo word2vec dir

            # Audio specs
            "audio_parameters": {
                "NFFT": 4096,
                "MFCC_INPUTS": 40,                              # how many parameters will store for each MFCC vector
                "HOP_LENGTH": 1/30,
                "DIM": 64,
            },

            # Gesture specs (from VQVAE trained model)
            "vqvae_checkpoint": r"C:\Users\faval\PycharmProjects\TAG2G\VQVAE\results\vqvae",
            "vqvae_params": {
                "num_emb": 2048,
                "emb_dim": 256,
                "input_dim": 74,
                "hidden_dim": 1024,
                "motion_window_length": 18
            },
            "dump_pipeline_dir" : r"C:\Users\faval\PycharmProjects\TAG2G\Utils",
        }

        # updating parameters from call method
        for parameter, value in kwargs.items():
            if parameter in parameters_dict:
                parameters_dict[parameter] = value

        return parameters_dict
