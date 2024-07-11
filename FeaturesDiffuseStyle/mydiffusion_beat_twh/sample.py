import pdb
import sys
import os
# [sys.path.append(i) for i in ['.', '..', '../process', '../model']]
script_path = os.path.dirname(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))      # /TAG2G folder position
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
from glob import glob
from tqdm import tqdm
import logging
from scipy import linalg
import csv


speaker_id_dict = {
    2: 0,
    10: 1,
}

id_speaker_dict = {
    0: 2,
    1: 10,
}


def frechet_distance_latent(args, filename, gesture, gen_gesture, eps=1e-6):

    assert gesture.shape == gen_gesture.shape, f'Provided different shapes between gt gesture and gen gesture'

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
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
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


def serial_printer(data):
    for key1, dict in data.items():
        for key2, value in dict.items():
            print(f"{key1}: {key2} shape {value.shape}")


def create_model_and_diffusion(args):

    model = MDM(modeltype='', njoints=args.njoints, nfeats=1, cond_mode=config.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.style_dim, source_audio_dim=args.audio_feature_dim,
                audio_feat_dim_latent=args.audio_feat_dim_latent, version=args.version)
    diffusion = create_gaussian_diffusion(vqvae=None)      # no need to use vqvae_cinematic: not computing any loss at inference
    return model, diffusion


def text_audio_windowing(text_data, audio_data, frames_per_feature=18):

    assert text_data.shape[0]==audio_data.shape[0], "Lengths of text and audio are not coherent: not equal frames"

    tot_frames = text_data.shape[0]                             # actual total frames of this sample (to be retrieved)
    n_features = text_data.shape[0] // frames_per_feature

    text_data = np.stack([text_data[frames_per_feature*win:frames_per_feature*win+frames_per_feature, ...] for win in range(n_features)], axis=0)
    audio_data = np.stack([audio_data[frames_per_feature*win:frames_per_feature*win+frames_per_feature, ...] for win in range(n_features)], axis=0)

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


def inference(args, data, prefix, sample_fn, model, n_features=0, smoothing=False, skip_timesteps=0, sample_length=0,
             seed=123456):

    """

    :param args: EasyDict with callable config parameters
    :param data: dict with main-agent and interloctr text-audio and ID
    :param prefix: name of the sample to generate
    :param sample_fn:
    :param model:
    :param n_features:
    :param smoothing:
    :param skip_timesteps:
    :return: None, saves a generated festure in save dir
    """

    # set dirs
    tst_path = args.tst_path

    # set seed for reproducibility
    torch.manual_seed(seed)

    # retrieve length of sample to compute length of diffusion process
    """
    NOTE:
    Here audio is transformed to in windows of length n_poses - n_seed because then the model will output 8 codes, 
    per each new subdivision the last code will be used next time as seed and only 7 codes of audio will be used, so    
    discarding last code of gestures let us have the same amount of code (7) both for textaudio and gesture 
    """
    if n_features == 0:
        n_features = data['main-agent']['textaudio'].shape[0]         # sample length in code-domain
    real_n_features = copy.deepcopy(n_features)                       # eg 108 - sample true length in code-domain
    stride_poses = args.n_poses - args.n_seed                         # 7codes = 8 - 1 ==> 7*18=126frames
    if n_features < stride_poses:
        # if n_frames are less then a single length of diff model -> just a single run
        num_subdivision = 1
        n_features = stride_poses         # old !!
        # n_features = args.n_poses
    else:
        # otherwise find out how many different runs are needed to generate all required codes
        num_subdivision = math.ceil(n_features / stride_poses)  # takes one more frame than needed, old !!
        # num_subdivision = math.ceil(n_features / args.n_poses)  # Why not like this?
        n_features = num_subdivision * stride_poses  # old
        # n_features = num_subdivision * args.n_poses
        print('real_n_features: {}, num_subdivision: {}, stride_poses: {}, n_features: {}'.format(real_n_features, num_subdivision, stride_poses, n_features))

    # Loading stats to normalize gestures
    # if tst path is something like local_path/data_folder/val_features/... --> stat path is ../trn_folder/..
    stat_dir = tst_path.replace("val", "trn")               # only works with val used as tst path
    gesture_stats = {
        'main-agent': {},
        'interloctr': {},
    }
    for speaker in args.speakers:
        for stat in ['mean', 'std']:
            stat_pointer = os.path.join(stat_dir, f"{speaker}-gesture_trn_{stat}_v0.npy")
            if stat == 'std':
                gesture_stats[speaker][stat] = np.clip(np.load(stat_pointer), a_min=0.01, a_max=None)
            else:
                gesture_stats[speaker][stat] = np.load(stat_pointer)

    # Starting inference cycle
    shape_ = (1, model.njoints, model.nfeats, args.n_poses)         # 1, 256, 1, math.ceil(n_frames // 18)
    out_list = []
    for i in range(0, num_subdivision):
        try:
            # Build data for inference mask, mask_local, style, seed(ma), textaudio, gesture(inter)
            model_kwargs_ = {
                'y' : {},
                'y_inter1' : {}
            }
            for speaker in args.speakers:

                if args.inference_mode == "monadic" and speaker == 'interloctr':
                    # If monadic: when entering interloctr should break
                    break

                # Retrieve speaker label and istanciate speaker data dict
                label = args.agent_labels[speaker]
                # model_kwargs_[label] = {}

                # Preparing mask, mask-local, style
                model_kwargs_[label]['mask'] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
                model_kwargs_[label]['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)
                style = data[speaker]['id']
                model_kwargs_[label]['style'] = torch.as_tensor([style]).float().to(mydevice)

                # Preparing textaudio for further slicing - zero-padding if needed
                if real_n_features < n_features:
                    # zero-padding textaudio signal
                    textaudio = data[speaker]['textaudio']
                    textaudio_pad = torch.zeros([n_features - real_n_features,
                                                 args.motion_window_length,
                                                 args.audio_feature_dim]).to(mydevice)
                    textaudio = torch.cat((textaudio, textaudio_pad), dim=0)
                else:
                    textaudio = data[speaker]['textaudio']

                # from [#frames, 18, 410] to [#subdivisions, 7, 18, 410]

                textaudio = textaudio.reshape(num_subdivision, stride_poses, args.motion_window_length,
                                              args.audio_feature_dim).transpose(0, 1)

                # If main agent textaudio from current subdivision, seed is first n_seed gesture from known gesture (file)
                if speaker == 'main-agent':
                    model_kwargs_[label]['audio'] = textaudio[-stride_poses:, i:i+1, ...].transpose(0, 1)   # may raise an error
                    if i == 0:
                        ### trying to assess first 5 seconds freezed
                        model_kwargs_['y_inter1']['audio'] = textaudio[-stride_poses:, i:i+1, ...].transpose(0, 1)

                        seed_gesture = data[speaker]['gesture'][:args.n_seed + 2]
                        # Takes 2 more frames so that to compute vel and acc in syncronized way
                        seed_gesture = (seed_gesture - gesture_stats[speaker]['mean']) / gesture_stats[speaker]['std']
                        seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
                        seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
                        seed_gesture_ = np.concatenate((seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc),
                                                       axis=1)  # (args.n_seed, args.njoints)
                        seed_gesture_ = torch.from_numpy(seed_gesture_).float().transpose(0, 1).unsqueeze(0).to(mydevice)
                        model_kwargs_[label]['seed'] = seed_gesture_.unsqueeze(2)
                        del seed_gesture_vel
                        del seed_gesture_acc
                    else:
                        # Last generated n_seed is a 256*3 dimensional code (pos, vel, acc)
                        model_kwargs_[label]['seed'] = out_list[-1][..., -args.n_seed:].to(mydevice)

                # If interloctr textaudio from previous sub, gesture (load data) last subdivision, seed last code seen
                if speaker == 'interloctr':
                    if i == 0:

                        # at start of conv interloctr has zero conv and zero gesture - -n_codes:0 sample doesn't exist
                        # model_kwargs_[label]['audio'] = torch.zeros((args.bs, args.n_poses-args.n_seed, args.motion_window_length, args.audio_feature_dim)).to(mydevice)

                        model_kwargs_[label]['gesture'] = torch.zeros((1, args.n_poses, args.njoints)).to(mydevice)
                        seed = torch.zeros((args.bs, args.njoints, args.n_seed, 1)).to(mydevice)
                        model_kwargs_[label]['seed'] = seed
                    else:
                        # Load gesture from sample and standardize
                        model_kwargs_[label]['audio'] = textaudio[-stride_poses:, i-1:i, ...].transpose(0, 1)
                        gesture = data[speaker]['gesture']
                        gesture = (gesture - gesture_stats[speaker]['mean']) / gesture_stats[speaker]['std']
                        # Need to pad interloctr gesture either at start or end of sample (if 8 codes are not available)
                        # gesture from (i-1)*n_poses-2 to (i-1)n_poses+n_poses  (-2 to compute vel and acc)
                        if (i-1)*args.n_poses -2 < 0:
                            # pad initial signal with two zeros
                            gesture = np.concatenate((np.zeros((2, gesture.shape[1])),
                                                     gesture[(i-1)*args.n_poses:(i-1)*args.n_poses+args.n_poses]),
                                                     axis=0)
                        gesture = gesture[(i-1)*args.n_poses-2:(i-1)*args.n_poses+args.n_poses, ...]   # 10, 256
                        # check - last subdivision may take not enough codes at the end
                        if gesture.shape[0] <= args.n_poses+2:
                            gesture_pad = np.zeros((args.n_poses+2, gesture.shape[1]))
                            gesture = np.concatenate((gesture, gesture_pad), axis=0)
                        vel = gesture[1:, ...] - gesture[:-1, ...]
                        acc = vel[1:, ...] - vel[:-1, ...]
                        gesture = np.concatenate((gesture[-args.n_poses:], vel[-args.n_poses:], acc[-args.n_poses:]), axis=1)
                        # gesture shape [8, 768]
                        model_kwargs_[label]['gesture'] = torch.from_numpy(gesture).float().transpose(0, 1).unsqueeze(0).to(mydevice)
                        seed = torch.from_numpy(gesture[-args.n_seed:]).float().transpose(0, 1).unsqueeze(0).to(mydevice)
                        model_kwargs_[label]['seed'] = seed.unsqueeze(3)
                        del vel
                        del acc

                del textaudio

            # serial_printer(data=model_kwargs_)      # debug line

            # Generating a sample of args.n_poses for this subdivision
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
                last_poses = out_list[-1][..., -args.n_seed:]      # (1, model.njoints, 1, args.n_seed)
                out_list[-1] = out_list[-1][..., :-args.n_seed]    # delete last gesture code as it is been used as seed
                # if smoothing:
                #     # Extract predictions
                #     last_poses_root_pos = last_poses[:, :12]        # (1, 3, 1, 8)
                #     next_poses_root_pos = sample[:, :12]        # (1, 3, 1, 88)
                #     root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                #     predict_pos = next_poses_root_pos[..., 0]
                #     delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                #     sample[:, :12] = sample[:, :12] - delta_pos
                # for j in range(len(last_poses)):
                for j in range(last_poses.shape[-1]):
                    # n = len(last_poses)
                    n = last_poses.shape[-1]                                   # 7
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

            out_list.append(sample)

        except:
            print(f'Try failed at {i} subdivision')

    # After generation clean out the sequence and do other stuff ...

    if "v0" in args.version:
        motion_feature_division = 3
    elif "v2" in args.version:
        motion_feature_division = 1
    else:
        raise NotImplementedError(f"{args.version} not implemented")

    # discarding velocity and acceleration features
    out_list = [i.detach().data.cpu().numpy()[:, :args.njoints // motion_feature_division, ...] for i in out_list]
    if len(out_list) > 1:
        out_dir_vec_1 = np.vstack(out_list[:-1])
        sampled_seq_1 = out_dir_vec_1.squeeze(2).transpose(0, 2, 1).reshape(args.bs, -1, model.njoints // motion_feature_division)
        out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
    sampled_seq = sampled_seq[:, args.n_seed:]

    out_poses = np.multiply(sampled_seq[0], gesture_stats['main-agent']['std']) + gesture_stats['main-agent']['mean']
    out_poses = out_poses[:real_n_features]
    print(f'Generated a shape of: {out_poses.shape}')

    return out_poses


def main(args, max_len=0, skip_timesteps=0, tst_prefix=None,
         wavlm_path=None, word2vector_path=None):

    # set dirs
    tst_path = args.tst_path
    model_path = args.model_path
    save_dir = args.save_path
    print(f"Will save generated data in '{save_dir}'")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # defining models: Diffusion and VQVAE
    print("Creating models and diffusion ...")
    model, diffusion = create_model_and_diffusion(args)
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop                                         # predict x_start

    vqvae = load_VQVAE(args, project_dir=project_path, device=mydevice)

    # load pipe
    print("Loading pipe ...")
    expmap_pipeline = jl.load(os.path.join(project_path, args.expmap_74j_pipe))

    # Prepare to store FDG scores
    csv_path_fgd = os.path.join(save_dir, '000_FGD_scores.csv')
    csv_fieldnames = ['file_id', 'file_name', 'fgd_score']
    with open(csv_path_fgd, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fieldnames)
        writer.writeheader()

    if tst_path is not None:

        print(f"Looking at tst_path: {tst_path}")

        # check files and data to be loaded
        tst_audio_dir = os.path.join(tst_path, 'audio')
        tst_text_dir = os.path.join(tst_path, 'text')
        if args.use_inter_clusters:
            tst_gesture_dir = os.path.join(tst_path, 'gesture_clustered')
        else:
            tst_gesture_dir = os.path.join(tst_path, 'gesture')
        metadata_path = os.path.join(tst_path, "metadata.csv")
        assert os.path.isfile(metadata_path), f"Provided metadata is not a file  '{metadata_path}'"

        # check for samples in the provided source directory
        metadict_byfname, metadict_byindex = load_metadata(metadata_path, speakers=("main-agent", "interloctr"))
        filenames = sorted(metadict_byfname.keys())
        if tst_prefix is not None:
            # if a prefix it's been specified goes on only with that one
            filenames = [tst_prefix]
        else:
            # if a prefix is not given go with all the filenames in the examples directory
            filenames = filenames
        print(f"Generating a batch of {len(filenames)} samples:")
        pprint(filenames)

        progress_bar = tqdm(enumerate(filenames), total=len(filenames), desc='Generating samples')
        for i, filename in progress_bar:

            try:

                # Update the description dynamically
                progress_bar.set_description(f'Processing file {filename} (index {i})')

                tic = time.perf_counter()

                data = {
                    'main-agent': {},
                    'interloctr': {}
                }

                for speaker in args.speakers:

                    # prepare text-audio
                    audio_path = os.path.join(tst_audio_dir, f"{filename}_{speaker}.npy")
                    text_path = os.path.join(tst_text_dir, f"{filename}_{speaker}.npy")
                    gesture_path = os.path.join(tst_gesture_dir, f"{filename}_{speaker}.npy")
                    text, audio, last_frame = text_audio_windowing(text_data=np.load(text_path),
                                                                   audio_data=np.load(audio_path),
                                                                   frames_per_feature=args.motion_window_length)
                    data[speaker]['textaudio'] = torch.FloatTensor(np.concatenate((audio, text), axis=-1)).to(mydevice)
                    data[speaker]['last_frame'] = torch.FloatTensor(last_frame)
                    data[speaker]['gesture'] = np.load(gesture_path)
                    del text, audio

                    # prepare speaker ID
                    pos = 0 if speaker == 'main-agent' else 1        # it is a tuple (main_id, inter_id)
                    pos = metadict_byfname[filename.replace("_main-agent", "")][1][pos]
                    speaker_id = np.zeros((17))
                    speaker_id[pos] = 1
                    data[speaker]['id'] = speaker_id
                    del pos
                    del speaker_id

                    # TODO: load here also stats

                    # print(f"textaudio shape: {textaudio.shape}, speaker_id: {speaker_id}")  # debug line

                # Inference for sample gesture, decode, pad to real last frame and smooth with filtering

                gesture = inference(args=args, data=data, prefix=filename, sample_length= last_frame,
                                    sample_fn=sample_fn, model=model)

                fgd_score = frechet_distance_latent(args=args, gesture=data['main-agent']['gesture'], gen_gesture=gesture,
                                                    filename=filename)
                print(f'{filename} FDG score: {fgd_score}')
                csv_entry = {'file_id': i, 'file_name': filename, 'fgd_score': fgd_score}
                with open(csv_path_fgd, mode='a') as file:
                    writer = csv.DictWriter(file, fieldnames=csv_fieldnames)
                    # Write the header row
                    writer.writerow(csv_entry)

                # print(f"Shape after inference: {gesture.shape}")  # debug line
                gesture = vqvae.vqvae.decoder(torch.tensor(gesture, dtype=torch.float64).to(mydevice))
                # print(f"Shape after decoder: {gesture.shape}")    # debug line
                gesture = gesture.detach().permute(0, 2, 1).cpu().numpy()
                gesture = np.concatenate([gesture[i, ...] for i in range(gesture.shape[0])])
                gesture = pad_with_lastframe(gesture=gesture, last_frame=last_frame)
                # print(f"Shape after padding: {gesture.shape}")  # debug line
                gesture = gesture_smoothing(sample=gesture)
                # print(f"gesture shape: {gesture.shape}")  # debug line
    
                # process gesture through the pipe and save as a bvh
                mocap_gesture = inverse_process_pipeline(gesture_data=gesture, pipeline=expmap_pipeline)
                save_gesture_as_bvh(mocap_data=mocap_gesture, save_dir=save_dir,
                                    file_name=f"{filename}_main-agent")
                toc = time.perf_counter()
                print(f"Generated {filename} in {round(toc-tic, 1)} seconds")

            except FileNotFoundError as e:
                logging.error(f"'{filename}' not found: {e}")

    elif tst_prefix is not None:
        raise NotImplementedError(f"Loading from a single file is not already been implemented but will be soon :D")

        # TODO: Needs to be implemented as a single sample generation starting from wav, txt, bvh ( for inter only)
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
    _non cinematic version
    python sample.py --config=configs/DiffuseStyleGesture.yml --tst_path=data/val_features --gpu 0 --model_path 'TWH_mymodel4_512_v0/model003825000.pt' --max_len 0 --tst_prefix 'val_2023_v0_000_main-agent'
    _cinematic version
    python sample.py --config=configs/DiffuseStyleGesture.yml --tst_path=data/val_features --gpu 0 --model_path 'TWH_mymodel4_512_cinematic/model005379000.pt' --max_len 0 --tst_prefix 'val_2023_v0_023'
    
    Ways of use:
    1. only tst_path --> data is alredy been featurized (text and audio) as .npy files, load everything from that dir
    2. tst_path & tst_prefix --> load only that sample's features from tst_path directory
    3. if only tst_prefix --> script to be built to avoid any kind of previous featurization of the script
    
    Two path to MDM and VQVAE, path to val features (needs a 
    '''

    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--bs', type=int, default='1')
    parser.add_argument('--config', default='configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', type=str, default=None)
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--model_path', type=str, default='../model000450000.pt')
    parser.add_argument('--tst_path', type=str, default=None)
    # parser.add_argument('--wav_path', type=str, default=None)
    # parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    # parser.add_argument('--dataset', type=str, default='BEAT')  # not implemented in TAG2G
    parser.add_argument('--wavlm_path', type=str, default='../../WavLM/WavLM-Large.pt')
    parser.add_argument('--wavlm', action='store_true')
    parser.add_argument('--word2vec_path', type=str, default='../../crawl-300d-2M.vec')
    parser.add_argument('--use_inter_clusters', action='store_true', default=False)

    """
    --Hypotesys:
    a. there are some models that can generate only in the monadic version
    b. other models can generate in dyadic too 
    
    1. each model saved has, in its directory, the original config.yml which trained the model 
    2. tst-path contains either an h5 file with all data or an audio and a text folder
    3. the directory data/generated/**model/**sample_number is created to host all generated samples 
    """

    # Define model's args and script's parameters
    args = parser.parse_args()
    # load config
    assert os.path.isfile(args.model_path), f"Provided model_path is not a file"
    model_dir = os.path.dirname(args.model_path)
    yml_files = glob(os.path.join(model_dir, '*.yml'))
    if yml_files:
        args.config = yml_files[0]
    else:
        raise FileNotFoundError("No .yml files found in the directory")

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

    # TODO: Give a thought on this and resolve --maybe a --wavlm should fit better
    if config.version == 'v0':
        config.motion_dim = 256
        config.njoints = 768
        config.latent_dim = 512
        config.audio_feat_dim_latent = 128
        config.style_dim = 17
        # if config.wavlm:
        #     config.audio_feature_dim = 1434
        # else:
        #     config.audio_feature_dim = 410
        config.inference_mode = 'monadic'           # --monadic kwargs: 'y'
        print(f"Model using '{config.inference_mode}' inference mode, with {config.audio_feature_dim} audio features")
    elif 'dyadic' in config.version:
        config.motion_dim = 256
        config.njoints = 768
        config.latent_dim = 512
        config.audio_feat_dim_latent = 128
        config.style_dim = 17
        # if config.wavlm:
        #     config.audio_feature_dim = 1434
        # else:
        #     config.audio_feature_dim = 410
        config.inference_mode = 'dyadic'            # --dyadic kwargs: 'y' and 'y_inter1' using both interloct and m-ag
        print(f"Model using '{config.inference_mode}' inference mode, with {config.audio_feature_dim} audio features")
    else:
        raise NotImplementedError

    # define device
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    # define save_dir
    assert os.path.isfile(args.model_path), 'Provided model_path is not a file'
    model_spicific = config.model_path.split('/')[-1].split('.')[0]     # eg model000450000
    print(f"Model spicific: {model_spicific}")
    config.save_path = os.path.join(config.save_path, model_spicific)
    print(f"After model spicific: {config.save_path}")

    # set model path
    config.model_path = os.path.join(script_path, config.model_path)

    # set agents and labels
    config.speakers = (config.speaker1, config.speaker2)
    config.agent_labels = {
        config.speakers[0]:config.label1,
        config.speakers[1]:config.label2
    }

    # define if starting from preprocessed data or from pure dataset data (.tsv and .wav files)
    # if config.tst_path is not None:
    #     config.tst_path = os.path.join(project_path, config.tst_path)

    main(args=config, max_len=config.max_len, tst_prefix=config.tst_prefix,
         wavlm_path=config.wavlm_path, word2vector_path=config.word2vec_path)