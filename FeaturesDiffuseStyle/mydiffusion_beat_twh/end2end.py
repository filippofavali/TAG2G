import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from data_loader.h5_data_loader import SpeechGestureDataset, RandomSampler, SequentialSampler
import torch
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FeaturesDiffuseStyle.utils.model_util import create_gaussian_diffusion
from FeaturesDiffuseStyle.model.mdm import MDM
from FeaturesDiffuseStyle.train.training_loop import TrainLoop
from VQVAE.system import VQVAESystem

"""
    TODO: data dimensions into this file from h5 need to be changed according to the actual dimension of
    features encodings
"""

# edited Favali, 03-01-2024


def create_model_and_diffusion(args,vqvae):

    model = MDM(modeltype='', njoints=args.njoints, nfeats=1, cond_mode=args.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.style_dim, source_audio_dim=args.audio_feature_dim, 
                audio_feat_dim_latent=args.audio_feat_dim_latent, version=args.version, labels=args.agent_labels)
    print(f"Model MDM correctly created")
    diffusion = create_gaussian_diffusion(vqvae)
    return model, diffusion


def load_vqvae(args, device):

    # loading VQVAE trained instance to perform features deconding into diffusion model

    model = VQVAESystem(num_embeddings=2048,
                        embedding_dim=256,
                        input_dim=74,
                        hidden_dim=1024,
                        max_frames=18)

    print(f"Loading parameters from '{args.vqvae_checkpoint}'")
    checkpoint = torch.load(args.vqvae_checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    print(f"Using {device} as device")
    model.to(device)
    model.eval()

    return model


def main(args, mydevice):

    # Get data, data loaders and collate function ready
    print(f"Loading '{args.dataset}' dataset")
    trn_dataset = SpeechGestureDataset(h5file=args.h5file, motion_dim=args.motion_dim, audio_dim=args.audio_feature_dim,
                                       style_dim=args.style_dim,
                                       sequence_length=args.n_poses, npy_root="../process", 
                                       version=args.version, dataset=args.dataset)        # debug
    print("Dataset correctly loaded ...")
    print(f"USING BATCHSIZE: {args.batch_size}")
    print(f"Dataset correctly loaded with {len(trn_dataset)} samples")
    train_loader = DataLoader(trn_dataset, num_workers=args.num_workers,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False)

    if args.use_vqvae_decoder:
        print("Using vqvae decoder in training loss ...")
        vqvae = load_vqvae(args, mydevice)
    else:
        vqvae = None

    model, diffusion = create_model_and_diffusion(args, vqvae)
    model.to(mydevice)

    TrainLoop(args=args, model=model, diffusion=diffusion, device=mydevice, data=train_loader).run_loop()


if __name__ == '__main__':
    '''
    cd ./BEAT-main/mydiffusion_beat/
    python end2end.py --config=./configs/DiffuseStyleGesture.yml --gpu 0
    '''

    # first look into args from parser
    args = parse_args()
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device(device_name)
    print(f"Using '{mydevice}' as device")
    torch.cuda.set_device(int(args.gpu))
    # torch.cuda.empty_cache()                                            # empty cache memory of cuda, dangerous to run this on the server
    args.no_cuda = args.gpu

    # then look into config file settings
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        # add arg parser inputs to parameters
        config[k] = v
    config = EasyDict(config)

    print(f"Starting model: {config.name}")
    assert config.name in ['FeaturesDiffuseStyle', 'DiffuseStyleGesture', 'DiffuseStyleGesture+'], f"'{config.name}' not implemented"
    if config.name == 'FeaturesDiffuseStyle':
        config.cond_mode = 'cross_local_attention4_style1'
    elif config.name == 'DiffuseStyleGesture' or 'DiffuseStyleGesture+':
        print(f"'{config.name}' model version not implemented yet")
    print(f"Using loss over vqvae_decoder: {config.use_vqvae_decoder}")
    if config.dyadic:
        config.version = 'dyadic'
        print(f"Dyadic mode: {config.dyadic}")

    # model parameter configs
    if 'v0' == config.version:
        config.motion_dim = 256
        config.njoints = 768                                                # 256*3
        config.latent_dim = 512
        config.audio_feat_dim_latent = 128
        config.style_dim = 17
        config.audio_feature_dim = 410
    elif 'dyadic' in config.version:
        config.motion_dim = 256
        config.njoints = 768                                                # 256*3
        config.latent_dim = 512
        config.audio_feat_dim_latent = 128
        config.style_dim = 17
        config.audio_feature_dim = 1434
    else:
        raise NotImplementedError

    # directories config
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(f"Project path: {project_path}")
    # set save directory
    config.save_dir = os.path.join(config.save_dir, f"TAG2Gmodel_{config.version}_{config.description}")
    if not os.path.exists(config.save_dir):
        print("Saving directory already exist ...")
        os.makedirs(name=config.save_dir)
    print('model save path: ', config.save_dir, '   version:', config.version)
    # define h5features pointer
    config.h5file = os.path.join(project_path, config.h5file)
    assert os.path.isfile(config.h5file), f"Provided h5_pointer '{config.h5file}' is not a file"
    # define vqvae checkpoint
    config.vqvae_checkpoint = os.path.join(project_path, config.vqvae_checkpoint)
    config.vqvae_checkpoint = os.path.join(config.vqvae_checkpoint, os.listdir(config.vqvae_checkpoint)[-1])
    assert os.path.isfile(config.vqvae_checkpoint), f"Provided vqvae checkpoint {config.vqvae_checkpoint} is not a file"
    # set speakers
    config.speakers = (config.speaker1, config.speaker2)
    config.agent_labels = {
        config.speakers[0]:config.label1,
        config.speakers[1]:config.label2
    }

    # store train hyperparameters into a textfile in the save dir
    # DONE: Favali - check if the underneath procedure works
    with open(os.path.join(config.save_dir, "hyperparameters.yml"), 'w') as file:
        yaml.dump(dict(config), file)


    print(f"Using batch_size: {args.batch_size}")

    main(config, mydevice)
