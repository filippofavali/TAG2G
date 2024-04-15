import pdb
import sys
import os
# [sys.path.append(i) for i in ['.', '..', '../process', '../model']]
script_path = os.path.dirname(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_path)
from FeaturesDiffuseStyle.model.mdm import MDM
from FeaturesDiffuseStyle.utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from VQVAE.system import  VQVAESystem
from VQVAE.vqvae_utils.motion_utils import save_gesture_as_bvh, gesture_smoothing, inverse_process_pipeline
import subprocess
from datetime import datetime
import copy
import librosa
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from Featurization.FeatureEmbedding import load_metadata
import joblib as jl
import time

# from process_BEAT_bvh import wav2wavlm, pose2bvh, pose2bvh_bugfix
# from process_TWH_bvh import pose2bvh as pose2bvh_twh
# from process_TWH_bvh import wavlm_init, load_metadata
import argparse


speaker_id_dict = {
    2: 0,
    10: 1,
}

id_speaker_dict = {
    0: 2,
    1: 10,
}


def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=args.njoints, nfeats=1, cond_mode=config.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.style_dim, source_audio_dim=args.audio_feature_dim,
                audio_feat_dim_latent=args.audio_feat_dim_latent)
    diffusion = create_gaussian_diffusion(vqvae=None)      # no need to use vqvae_cinematic: not computing any loss at inference
    return model, diffusion


def text_audio_windowing(text_data, audio_data, frames_per_feature=18):

    assert text_data.shape[0]==audio_data.shape[0], "Lengths of text and audio are not coherent: not equal frames"

    tot_frames = text_data.shape[0]                             # actual total frames of this sample (to be retrieved)
    n_features = text_data.shape[0] // frames_per_feature
    print(f"n_features: {n_features}")

    text_data = np.stack([text_data[frames_per_feature*win:frames_per_feature*win+frames_per_feature, ...] for win in range(n_features)], axis=0)
    audio_data = np.stack([audio_data[frames_per_feature*win:frames_per_feature*win+frames_per_feature, ...] for win in range(n_features)], axis=0)

    print(text_data.shape, audio_data.shape)

    return text_data, audio_data, tot_frames


def load_VQVAE(args, project_dir, device):

    # define hyperparams
    emb_num = args.vq_emb_num           # 2048
    emb_dim = args.vq_emb_dim           # 256
    hidden_dim = args.vq_hidden_dim     # 1024
    input_dim = args.vq_input_dim          # 74
    checkpoint = os.path.join(project_dir,  args.vqvae_checkpoint)
    checkpoint = os.path.join(checkpoint, os.listdir(checkpoint)[-1])

    assert os.path.isfile(checkpoint), "Provided checkpoint is not a file"

    vqvae = VQVAESystem(num_embeddings=emb_num,
                        embedding_dim=emb_dim,
                        hidden_dim=hidden_dim,
                        input_dim=input_dim,
                        max_frames=18)

    print(f"Loading parameters from '{checkpoint}'")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    vqvae.load_state_dict(state_dict)
    print(f"Using {device} as device")
    vqvae.to(device)
    vqvae.eval()

    return vqvae


def pad_with_lastframe(gesture, last_frame):

    if last_frame > gesture.shape[0]:
        pad_length = last_frame - gesture.shape[0]
        pad = np.concatenate([gesture[-1:, ...] for _ in range(pad_length)], axis=0)
        gesture = np.concatenate([gesture, pad], axis=0)

    # else gesture is already at former length and good as it is

    return gesture


def inference(args, save_dir, tst_path, prefix, textaudio, sample_fn, model, n_features=0, smoothing=False, skip_timesteps=0,
              style=None, seed=123456, dataset='BEAT', tot_frames=0):

    """
    TODO: need to zero pad the end - so pass also real length of the sample and implement padding at last frame
    :param args:
    :param save_dir:
    :param prefix:
    :param textaudio:
    :param sample_fn:
    :param model:
    :param n_frames:
    :param smoothing:
    :param skip_timesteps:
    :param style:
    :param seed:
    :param dataset:
    :return:
    """
    # set seed for reproducibility
    torch.manual_seed(seed)
    # retrieve speaker ID from one hot encoding
    speaker = np.where(style == np.max(style))[0][0]

    # find out how many runs of diffusion model are needed for the sample
    if n_features == 0:
        n_features = textaudio.shape[0]
    else:
        textaudio = textaudio[:n_features]
    real_n_features = copy.deepcopy(n_features)     # 1830
    stride_poses = args.n_poses - args.n_seed
    if n_features < stride_poses:
        # if n_frames are less then a single length of diff model -> just a single run
        num_subdivision = 1
        n_features = stride_poses
    else:
        # otherwise find out how many different runs are needed
        num_subdivision = math.ceil(n_features / stride_poses)  # takes one more frame than needed
        n_features = num_subdivision * stride_poses
        print('real_n_features: {}, num_subdivision: {}, stride_poses: {}, n_features: {}, speaker_id: {}'.format(real_n_features, num_subdivision, stride_poses, n_features, speaker))

    model_kwargs_ = {'y': {}}
    # don't need to adapt masks dimension because first thing text_audio is linearly pushed to normal size
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    # padding text_audio context features to match num_subdivision
    textaudio_pad = torch.zeros([n_features - real_n_features, args.motion_window_length, args.audio_feature_dim]).to(mydevice)
    textaudio = torch.cat((textaudio, textaudio_pad), 0)
    audio_reshape = textaudio.reshape(num_subdivision, stride_poses, args.motion_window_length, args.audio_feature_dim).transpose(0, 1)

    # loading stats from festures embedding latent space
    feat_dir, _ = os.path.split(tst_path)
    data_mean_ = np.load(os.path.join(feat_dir, "trn_features", "main-agent-gesture_trn_mean_v0.npy"))
    data_std_ = np.load(os.path.join(feat_dir, "trn_features", "main-agent-gesture_trn_std_v0.npy"))
    data_mean = np.array(data_mean_)
    data_std = np.array(data_std_)
    # std = np.clip(data_std, a_min=0.01, a_max=None)

    """
    # DiffuseStyleGesture++ not implemented in FeaturesDiffuseStyle
    if args.name == 'DiffuseStyleGesture++':
        gesture_flag1 = np.load("../../BEAT_dataset/processed/" + 'gesture_BEAT' + "/2_scott_0_1_1.npy")[:args.n_seed + 2]
        gesture_flag1 = (gesture_flag1 - data_mean) / data_std
        gesture_flag1_vel = gesture_flag1[1:] - gesture_flag1[:-1]
        gesture_flag1_acc = gesture_flag1_vel[1:] - gesture_flag1_vel[:-1]
        gesture_flag1_ = np.concatenate((gesture_flag1[2:], gesture_flag1_vel[1:], gesture_flag1_acc), axis=1)  # (args.n_seed, args.njoints)
        gesture_flag1_ = torch.from_numpy(gesture_flag1_).float().transpose(0, 1).unsqueeze(0).to(mydevice)
        gesture_flag1_ = gesture_flag1_.unsqueeze(2)
        model_kwargs_['y']['seed_last'] = gesture_flag1_
    """
    
    shape_ = (1, model.njoints, model.nfeats, args.n_poses)         # 1, 256, 1, math.ceil(n_frames // 18)
    out_list = []
    for i in range(0, num_subdivision):
        print(f"Need to sample {num_subdivision} times")
        print(f"{i}/{num_subdivision}")
        model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]
        if i == 0:

            if args.name == 'FeaturesDiffuseStyle':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)       # attention 4
            elif args.name == 'DiffuseStyleGesture' or 'DiffuseStyleGesture+' or 'DiffuseStyleGesture++':
                # pad_zeros = torch.zeros([args.n_seed, 1, args.audio_feature_dim]).to(mydevice)
                # model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0).transpose(0,1)  # attention 3
                # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)  # attention 4
                # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][:-args.n_seed, ...].transpose(0, 1)       # attention 5
                raise NotImplementedError
                
            # model_kwargs_['y']['seed'] = torch.zeros([1, args.njoints, 1, args.n_seed]).to(mydevice)

            # seed gesture loading, standardization and stacking
            # TODO: one day i will need to take both the people data :')
            seed_gesture = np.load(os.path.join(tst_path, "gesture", f"main-agent_{prefix}.npy"))[:args.n_seed + 2]
            # takes 2 more frames so that to compute vel and acc in syncronized way
            seed_gesture = (seed_gesture - data_mean) / data_std
            seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
            seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
            seed_gesture_ = np.concatenate((seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1)      # (args.n_seed, args.njoints)
            seed_gesture_ = torch.from_numpy(seed_gesture_).float().transpose(0, 1).unsqueeze(0).to(mydevice)
            model_kwargs_['y']['seed'] = seed_gesture_.unsqueeze(2)

        else:

            if args.name == 'FeaturesDiffuseStyle':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)  # attention 4
            elif args.name == 'DiffuseStyleGesture' or 'DiffuseStyleGesture+' or 'DiffuseStyleGesture++':
                # pad_audio = audio_reshape[-args.n_seed:, i - 1:i]  # attention 3
                # model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0).transpose(0,1)  # attention 3
                # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)  # attention 4
                # model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][:-args.n_seed, ...].transpose(0, 1)  # attention 5
                raise NotImplementedError

            # if is not first subdivision uses as seed the last n.seed features from last generation
            model_kwargs_['y']['seed'] = out_list[-1][..., -args.n_seed:].to(mydevice)

        # generating a sample of n.features for this subdivision
        sample = sample_fn(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # None, torch.randn(*shape_, device=mydevice)
            const_noise=False,
        )

        # smoothing motion transition
        if len(out_list) > 0 and args.n_seed != 0:
            last_poses = out_list[-1][..., -args.n_seed:]        # # (1, model.njoints, 1, args.n_seed)
            out_list[-1] = out_list[-1][..., :-args.n_seed]  # delete last 4 frames
            # if smoothing:
            #     # Extract predictions
            #     last_poses_root_pos = last_poses[:, :12]        # (1, 3, 1, 8)
            #     next_poses_root_pos = sample[:, :12]        # (1, 3, 1, 88)
            #     root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
            #     predict_pos = next_poses_root_pos[..., 0]
            #     delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
            #     sample[:, :12] = sample[:, :12] - delta_pos
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[..., j]
                next = sample[..., j]
                sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(sample)

    if "v0" in args.version:
        motion_feature_division = 3
    elif "v2" in args.version:
        motion_feature_division = 1
    else:
        raise ValueError("wrong version name")

    # discarding velocity and acceleration features
    out_list = [i.detach().data.cpu().numpy()[:, :args.njoints // motion_feature_division] for i in out_list]
    if len(out_list) > 1:
        out_dir_vec_1 = np.vstack(out_list[:-1])
        sampled_seq_1 = out_dir_vec_1.squeeze(2).transpose(0, 2, 1).reshape(batch_size, -1, model.njoints // motion_feature_division)
        out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
    sampled_seq = sampled_seq[:, args.n_seed:]

    out_poses = np.multiply(sampled_seq[0], data_std) + data_mean
    print(out_poses.shape, real_n_features)
    out_poses = out_poses[:real_n_features]

    """
    if dataset == 'BEAT':
        if "v0" in args.version:
            pose2bvh_bugfix(save_dir, prefix, out_poses, pipeline='../process/resource/data_pipe_30fps' + '_speaker' + str(speaker) + '.sav')
        elif "v2" in args.version:
            pose2bvh(save_dir, prefix, out_poses)
        else:
            raise ValueError("wrong version name")
    elif dataset == 'TWH':
        pose2bvh_twh(out_poses, save_dir, prefix, pipeline_path="../process/resource/pipeline_rotmat_62.sav")
    """

    return out_poses


def main(args, save_dir, model_path, tst_path=None, max_len=0, skip_timesteps=0, tst_prefix=None, dataset='BEAT', 
         wav_path=None, txt_path=None, wavlm_path=None, word2vector_path=None):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # defining models: Diffusion and VQVAE
    print("Creating models and diffusion ...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from '{model_path}'...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop  # predict x_start

    vqvae = load_VQVAE(args, project_dir=project_path, device=mydevice)

    # load pipe
    print("Loading pipe ...")
    expmap_pipeline = jl.load(os.path.join(project_path, args.expmap_74j_pipe))


    if tst_path is not None:

        # check files and data to be loaded
        tst_audio_dir = os.path.join(tst_path, 'audio')
        tst_text_dir = os.path.join(tst_path, 'text')

        metadata_path = os.path.join(tst_path, "metadata.csv")
        assert os.path.isfile(metadata_path), f"Provided metadata is not a file  '{metadata_path}'"
        metadict_byfname, metadict_byindex = load_metadata(metadata_path, speakers=("main-agent", "interloctr"))
        filenames = sorted(metadict_byfname.keys())

        # if a prefix it's been specified goes on only with that one
        if tst_prefix is not None:
            filenames = [tst_prefix]
        else:
            filenames = filenames
            print(f"Generating a batch of {len(filenames)} samples:")
            pprint(filenames)

        # TODO: one day i will have to asses again the prolem but with two persons :')
        for i, filename in enumerate(filenames):
            tic = time.perf_counter()
            print(f"Processing: {filename}")
            speaker_id = metadict_byfname[filename.replace("_main-agent", "")][1][0]        # discard 'has_fingers' and pick only 'main-agent' ID
            speaker = np.zeros([17])
            speaker[speaker_id] = 1
                
            audio_path = os.path.join(tst_audio_dir, f"main-agent_{filename}.npy")
            audio = np.load(audio_path)
            text_path = os.path.join(tst_text_dir, f"main-agent_{filename}.npy")
            text = np.load(text_path)
            text, audio, last_frame = text_audio_windowing(text_data=text, audio_data=audio,
                                                           frames_per_feature=args.motion_window_length)
            textaudio = np.concatenate((audio, text), axis=-1)
            textaudio = torch.FloatTensor(textaudio)
            textaudio = textaudio.to(mydevice)

            print(f"textaudio shape: {textaudio.shape}, speaker_id: {speaker_id}")

            # inference for sample gesture, decode, pad to real last frame and smooth with filtering
            gesture = inference(args=args, save_dir=save_dir, tst_path=tst_path, prefix=filename, textaudio=textaudio,
                                sample_fn=sample_fn, model=model, n_features=max_len, smoothing=True,skip_timesteps=skip_timesteps,
                                style=speaker, seed=123456, dataset=dataset, tot_frames=last_frame)
            print(f"Shape after inference: {gesture.shape}")
            gesture = vqvae.vqvae.decoder(torch.tensor(gesture, dtype=torch.float64).to(mydevice))
            print(f"Shape after decoder: {gesture.shape}")
            gesture = gesture.detach().permute(0, 2, 1).cpu().numpy()
            gesture = np.concatenate([gesture[i, ...] for i in range(gesture.shape[0])])
            gesture = pad_with_lastframe(gesture=gesture, last_frame=last_frame)
            print(f"Shape after padding: {gesture.shape}")
            gesture = gesture_smoothing(sample=gesture)
            print(f"gesture shape: {gesture.shape}")

            # process gesture through the pipe and save as a bvh
            mocap_gesture = inverse_process_pipeline(gesture_data=gesture, pipeline=expmap_pipeline)
            # TODO: review of this point if a another person needs to be saved
            save_gesture_as_bvh(mocap_data=mocap_gesture, file_path=save_dir,
                                file_name=f"main-agent_{filename}")

            toc = time.perf_counter()
            print(f"Generated {filename} in {round(toc-tic, 1)} seconds")

    # TODO: Needs to be implemented as a single sample generation
    """
        The scripts goes to .wav,.tsv folder and calling audio processor and text processor process data instead of loading
    else:       
        # 20230805 update: generate audiowavlm..., sample from single one
        if dataset == 'TWH':
            from process_TWH_bvh import load_wordvectors, load_audio, load_tsv
        elif dataset == 'BEAT':
            from process_BEAT_bvh import load_wordvectors, load_audio, load_tsv

        wavlm_model, cfg = wavlm_init(wavlm_path, mydevice)
        word2vector = load_wordvectors(fname=word2vector_path)
    
        wav = load_audio(wav_path, wavlm_model, cfg)
        clip_len = wav.shape[0]
        tsv = load_tsv(txt_path, word2vector, clip_len)
        textaudio = np.concatenate((wav, tsv), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)
        speaker = np.zeros([17])
        speaker[0] = 1      # random choice will be great
        filename = 'tts'
        inference(args, save_dir, filename, textaudio, sample_fn, model, n_frames=max_len, smoothing=True,
                  skip_timesteps=skip_timesteps, style=speaker, seed=123456, dataset=dataset)
    """


if __name__ == '__main__':

    ''' 
    command to be called: 
    
    _diffuse style gesture version
    python sample.py --config=configs/DiffuseStyleGesture.yml --gpu 0 --model_path 'TWH_mymodel4_512_v0/model001200000.pt' --max_len 0 --tst_prefix 'val_2023_v0_014_main-agent'
    
    _non cinematic version
    python sample.py --config=configs/DiffuseStyleGesture.yml --tst_path=data/val_features --gpu 0 --model_path 'TWH_mymodel4_512_v0/model001200000.pt' --max_len 0 --tst_prefix 'val_2023_v0_000_main-agent'
    python sample.py --config=configs/DiffuseStyleGesture.yml --tst_path=data/val_features --gpu 0 --model_path 'TWH_mymodel4_512_v0/model003825000.pt' --max_len 0 --tst_prefix 'val_2023_v0_000_main-agent'
    
    -cinematic version
    python sample.py --config=configs/DiffuseStyleGesture.yml --tst_path=data/val_features --gpu 0 --model_path 'TWH_mymodel4_512_cinematic/model005379000.pt' --max_len 0 --tst_prefix 'val_2023_v0_023'
    
    Ways of use:
    1. only tst_path --> data is alredy been featurized (text and audio) as .npy files, load everything from that dir
    2. tst_path & tst_prefix --> load only that sample's features from tst_path directory
    3. if only tst_prefix --> script to be built to avoid any kind of previous featurization of the script
    '''

    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', type=str, default=None)
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--model_path', type=str, default='./model000450000.pt')
    parser.add_argument('--tst_path', type=str, default=None)
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='BEAT')  # not implemented in TAG2G
    parser.add_argument('--wavlm_path', type=str, default='./WavLM/WavLM-Large.pt')  # not implemented in TAG2G so far
    parser.add_argument('--word2vector_path', type=str, default='././crawl-300d-2M.vec')

    # define model's args and script's parameters
    args = parser.parse_args()
    args.config = os.path.join(script_path, args.config)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)
    config = EasyDict(config)

    assert config.name in ['FeaturesDiffuseStyle', 'DiffuseStyleGesture', 'DiffuseStyleGesture+', 'DiffuseStyleGesture++']
    if config.name == 'FeaturesDiffuseStyle':
        config.cond_mode = 'cross_local_attention4_style1_sample'
    elif config.name == 'DiffuseStyleGesture' or 'DiffuseStyleGesture+' or 'DiffuseStyleGesture++':
        raise NotImplementedError

    if 'v0' in config.version:
        config.motion_dim = 256
        config.njoints = 768
        config.latent_dim = 512
        config.audio_feat_dim_latent = 128
        config.style_dim = 17
        config.audio_feature_dim = 410
    else:
        raise NotImplementedError

    # define device
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    # define batch_size to sample
    batch_size = 1

    # define save_dir
    model_root = config.model_path.split('/')[1]
    model_spicific = config.model_path.split('/')[-1].split('.')[0]
    # config.save_dir = "./" + "./" + model_root + '/' + 'sample_dir_' + model_spicific + '/'
    config.save_dir = os.path.join(project_path, "data", "generated_samples", model_spicific)

    # set model path
    config.model_path = os.path.join(script_path, config.model_path)

    # define if starting from preprocessed data or from pure dataset data (.tsv and .wav files)
    if config.tst_path is not None:
        config.tst_path = os.path.join(project_path, config.tst_path)
    else:
        # should look into user/dataset/val or another dir where I have the sample etc. pass this to main function
        # then main function should do embeddings, calculate mean and variance and then use features
        raise NotImplementedError
    # print('model_root', model_root, 'tst_path', config.tst_path, 'save_dir', config.save_dir)

    # TODO: Need to introduce a dyadic version input sample generation (thus input loanding etcc..)
    main(config, config.save_dir, config.model_path, tst_path=config.tst_path, max_len=config.max_len,
         skip_timesteps=config.skip_timesteps, tst_prefix=config.tst_prefix, dataset=config.dataset, 
         wav_path=config.wav_path, txt_path=config.txt_path, wavlm_path=config.wavlm_path, word2vector_path=config.word2vector_path)
