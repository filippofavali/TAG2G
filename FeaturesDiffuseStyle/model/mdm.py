import pdb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(__file__))
# print(sys.path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FeaturesDiffuseStyle.model.local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from FeaturesDiffuseStyle.model.local_attention.local_attention import LocalAttention


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, motion_window_length=18,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='expmap_74j', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', device='cpu', 
                 style_dim=-1, source_audio_dim=-1, audio_feat_dim_latent=-1, version="v0", labels=['y', 'y_inter1'],
                 **kargs):
        super().__init__()

        print(f"MDM Receiving version {version}")

        # TODO: Need to activate a model with dyadic inputs if version = Dyadic
        if "dyadic" in version:
            self.model_version = "dyadic"   # forward path will be dyadic
        elif version=='v0':
            self.model_version = 'v0'       # forward path will be monadic
        print(f"MDM model version: '{self.model_version}'")

        self.agents_labels = labels
        print("From MDM init procedure")
        print(f"agents_labels: {[label for label in self.agents_labels]}")

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints                                                  # 256
        self.nfeats = nfeats                                                    # 1
        self.data_rep = data_rep                                                # expmap_74j
        self.dataset = dataset

        self.latent_dim = latent_dim                                            # 512 in FeatureDiffuseStyleGesture
        self.motion_window_length = motion_window_length                        # 18

        self.ff_size = ff_size                                                  # 1024
        self.num_layers = num_layers                                            # 8
        self.num_heads = num_heads                                              # 4
        self.dropout = dropout                                                  # 0.1

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        
        self.source_audio_dim = source_audio_dim
        self.audio_feat = audio_feat
        if self.audio_feat == 'wavlm':
            print('USE WAVLM')

        self.audio_feat_dim = audio_feat_dim_latent        # Linear 1024 -> 64
        self.WavEncoder = WavEncoder(self.source_audio_dim, self.motion_window_length, self.audio_feat_dim)

        # Time positional embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.model_version == 'dyadic':
            self.sequence_pos_encoder2 = PositionalEncoding(2 * self.latent_dim, self.dropout)
            self.embed_timestep2 = TimestepEmbedder(2 * self.latent_dim, self.sequence_pos_encoder2)
        self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode
        self.num_head = 8

        if 'style2' not in self.cond_mode:
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            if self.model_version == 'v0':
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                  nhead=self.num_heads,
                                                                  dim_feedforward=self.ff_size,
                                                                  dropout=self.dropout,
                                                                  activation=self.activation)
            elif self.model_version == 'dyadic':
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=2 * self.latent_dim,
                                                                  nhead=self.num_heads,
                                                                  dim_feedforward=self.ff_size,
                                                                  dropout=self.dropout,
                                                                  activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc]')

        self.n_seed = n_seed
        if 'style1' in self.cond_mode:
            print('EMBED STYLE BEGIN TOKEN')
            if 'cross_local_attention3' in self.cond_mode:
                self.style_dim = 64
                self.embed_style = nn.Linear(style_dim, self.style_dim)
                self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim - self.style_dim)
            elif 'cross_local_attention4' in self.cond_mode:
                self.style_dim = self.latent_dim
                if self.model_version == 'dyadic':
                    self.mixture_of_style = nn.Linear(2 * self.style_dim, 2 * self.style_dim)
                self.embed_style = nn.Linear(style_dim, self.style_dim)
                self.embed_dyadic_style = nn.Linear(2*self.style_dim, self.style_dim)
                self.embed_text = nn.Linear(self.njoints, self.audio_feat_dim)
            elif 'cross_local_attention5' in self.cond_mode:
                self.style_dim = self.latent_dim
                self.embed_style = nn.Linear(style_dim, self.style_dim)
                self.embed_text = nn.Linear(self.njoints, self.audio_feat_dim)
                self.embed_text_last = nn.Linear(self.njoints, self.audio_feat_dim)

        elif 'style2' in self.cond_mode:
            print('EMBED STYLE ALL FRAMES')
            self.style_dim = 64
            self.embed_style = nn.Linear(style_dim, self.style_dim)
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim + self.style_dim,
                                              self.latent_dim)
            if self.n_seed != 0:
                self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim)
        elif self.n_seed != 0:
            self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim)

        if self.model_version == 'v0':
            self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                                self.nfeats)
        elif self.model_version == 'dyadic':
            self.output_process = OutputProcess(self.data_rep, self.input_feats, 2 * self.latent_dim, self.njoints,
                                                self.nfeats)

        if 'cross_local_attention' in self.cond_mode:
            if self.model_version == 'v0':
                self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            elif self.model_version == 'dyadic':
                self.rel_pos = SinusoidalEmbeddings(2 * self.latent_dim // self.num_head)
            else:
                raise NotImplementedError
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
            if self.model_version == 'dyadic':
                self.inter_input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, 2 * self.latent_dim)
            self.cross_local_attention = LocalAttention(
                # Favali notes: dim is not used in this implementation
                dim=48,  # dimension of each head (you need to pass this in for relative positional encoding)
                # changed from 15 to 8 --> window_size = 8
                window_size=8,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2 = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        # bs, d = cond.shape                                    # old implementation
        bs, d = cond.shape[0], cond.shape[-1]                   # doing this works with every n agents provided
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            if len(cond.shape) == 3:
                mask = torch.cat((mask.unsqueeze(1), mask.unsqueeze(1)), axis=1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        """

        # print(f"Y keys: {[key for key in y.keys()]}")  # debug line for test keys ...

        # serial_printer(y)     # debug line

        if self.model_version == "v0":

            # monadic procedure
            bs, njoints, nfeats, nframes = x.shape          # b, 256*3, 1, 8
            emb_t = self.embed_timestep(timesteps)          # [1, 2, 512]

            force_mask = y.get('uncond', False)             # Default to false, 'uncond' not introduced in _cond
            embed_style = self.mask_cond(self.embed_style(y['y']['style']), force_mask=force_mask)       # [b, 512], (bs, 17)

            # 'cross_local_attention4' as self.cond_mode
            # y_seed [b, 768, 1, 1]
            embed_text = self.embed_text(y['y']['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  #[1, b, 128], (b, 8, 768)
            enc_text = self.WavEncoder(y['y']['audio']).permute(1, 0, 2)    # [7, b, 128], (b, 7, 18, 410)
            enc_text = torch.cat((embed_text, enc_text), axis=0)     # [8, b, 128]
            x = x.reshape(bs, njoints * nfeats, 1, nframes)                 # [b, 768, 1, 8]

            # self-attention
            x_ = self.input_process(x)  # [b, 768, 1, 8] -> [8, b, 512]
            # local-cross-attention
            packed_shape = [torch.Size([bs, self.num_head])]
            xseq = torch.cat((x_, enc_text), axis=2)  # [8, b, 640], ((8, b, 512), (8, b, 128))

            # all frames
            embed_style_2 = (embed_style + emb_t).repeat(nframes, 1, 1)  # (bs, 512) -> (8, b, 512)
            xseq = torch.cat((embed_style_2, xseq), axis=2)  # [8, b, 1152]
            xseq = self.input_process2(xseq)                        # [8, b, 512]
            xseq = xseq.permute(1, 0, 2)                            # [b, 8, 512]
            # <-- qui dovrei mettere qualocsa che faccia si che non vada come un view tutto alla fine ma un elemento di uno ed un elemento di un altro
            xseq = xseq.view(bs, nframes, self.num_head, -1)        # [b, 8, 8, 64]
            # xseq = xseq.permute(0, 2, 1, 3)                         # Need (b, 8, 8, 64)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)    # [8*b, 8, 64]
            pos_emb = self.rel_pos(xseq)                            # (8, 64)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)     # [8*b, 8, 64]

            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                              mask=y['y']['mask_local'])  # [b, 8, 8, 64]

            # ____ from here dyadic is going to be the same

            xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)

            xseq = torch.cat((embed_style + emb_t, xseq), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
            xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
            xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
            xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
            pos_emb = self.rel_pos(xseq)  # (89, 32)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
            xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
            xseq = xseq.view(bs, nframes + 1, -1)
            xseq = xseq.permute(1, 0, 2)
            output = self.seqTransEncoder(xseq)[1:]

            output = self.output_process(output)                           # [bs, njoints, nfeats, nframes]
            return output

        elif self.model_version == "dyadic":

           # dyadic procedure
            bs, njoints, nfeats, nframes = x.shape                  # bs, 256*3, 1, 8
            emb_t = self.embed_timestep(timesteps)                  # [1, 2, 512]
            packed_shape = [torch.Size([bs, self.num_head])]

            # Self attention - input process shared across different agents
            x = x.reshape(bs, njoints * nfeats, 1, nframes)         # [b, 768, 1, 8]
            x_ = self.input_process(x)                              # [b, 768, 1, 8] -> [8, b, 512]

            # Embedding agents ID
            force_mask = y.get('uncond', False)  # Default to false, 'uncond' not introduced in _cond
            style = torch.cat((y['y']['style'], y['y_inter1']['style']), axis=0)  # [bs, 2, 17]
            style = self.mask_cond(self.embed_style(style), force_mask=force_mask)       # [bs, 2, 64], (bs, 2, 17)
            embed_style_1, embed_style_2 = style[:bs, :], style[bs:, :]

            # embed_style_1 = self.mask_cond(self.embed_style(y['y']['style']), force_mask=y.get('uncond', False))  # [bs, 64], (bs, 17)
            # embed_style_2 = self.mask_cond(self.embed_style(y['y_inter1']['style']), force_mask=y.get('uncond', False))
            mixed_style = self.mixture_of_style(torch.cat((embed_style_1, embed_style_2), axis=1))     # [b, 1024]

            # Embedding conversation from agents - seed and text-audio from agents
            embed_text = torch.cat((y['y']['seed'].squeeze(2).permute(0, 2, 1), y['y_inter1']['seed'].squeeze(2).permute(0, 2, 1)), axis=0)
            embed_text = self.embed_text(embed_text).permute(1, 0, 2)
            embed_text_1, embed_text_2 = embed_text[:, :bs, :], embed_text[:, bs:, :]

            # embed_text_1 = self.embed_text(y['y']['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # [1, b, 128], (b, 1, 768)
            # embed_text_2 = self.embed_text(y['y_inter1']['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)  # [1, b, 128], (b, 1, 768)
            # embed_text = torch.cat((y['y']['seed'], y['y_inter1']['seed']), axis=2).permute(3, 0, 2, 1)  # [2, bs, 1, 768]
            audio = torch.cat((y['y']['audio'], y['y_inter1']['audio']), axis=1)
            audio_dim = audio.shape[1] // 2
            audio = self.WavEncoder(audio).permute(1, 0, 2)
            enc_text_1 = audio[:audio_dim, ...]
            enc_text_2 = audio[audio_dim:, ...]
            # enc_text_1 = self.WavEncoder(y['y']['audio']).permute(1, 0, 2)  # [7, b, 128], (b, 7, 18, 410)
            # enc_text_2 = self.WavEncoder(y['y_inter1']['audio']).permute(1, 0, 2)  # [7, b, 128], (b, 7, 18, 410)

            enc_text_1 = torch.cat((embed_text_1, enc_text_1), axis=0)  # [8, b, 128] --> encoded conversation main agent
            enc_text_2 = torch.cat((embed_text_2, enc_text_2), axis=0)  # [8, b, 128] --> encoded conversation interloctr

            xseq_1 = torch.cat((x_, enc_text_1), axis=2)  # [8, b, 640], ((8, b, 512), (8, b, 128))
            xseq_2 = torch.cat((x_, enc_text_2), axis=2)  # [8, b, 640], ((8, b, 512), (8, b, 128))

            # Prepare all frames for cross local attention
            mixed_style_12 = (embed_style_1 + emb_t).repeat(nframes, 1, 1)  # [bs, 512] --> [8, b, 512]
            mixed_style_22 = (embed_style_2 + emb_t).repeat(nframes, 1, 1)  # [bs, 512] --> [8, b, 512]
            xseq_1 = torch.cat((mixed_style_12, xseq_1), axis=2)     # [8, b, 1152]
            xseq_2 = torch.cat((mixed_style_22, xseq_2), axis=2)     # [8, b, 1152]

            xseq = torch.cat((xseq_1, xseq_2), axis=1)               # [8, 2b, 1152]
            xseq = self.input_process2(xseq).permute(1, 0, 2)               # [8, 2b, 512]
            xseq_1 = xseq[:bs]                                        # [8, b, 512]
            xseq_2 = xseq[bs:]                                        # [8, b, 512]

            # xseq_1 = self.input_process2(xseq_1).permute(1, 0, 2)            # [b, 8, 512]
            # xseq_2 = self.input_process2(xseq_2).permute(1, 0, 2)            # [b, 8, 512]

            # need to build a conversational input from dyadic setup
            xseq = torch.cat((xseq_1.unsqueeze(3), xseq_2.unsqueeze(3)), axis=3)    # [b, 8, 512, 2]
            xseq = xseq.view(bs, nframes, self.num_head, -1)                        # [b, 8, 8, 128]
            # need [b, n.head, frames, data]
            xseq = xseq.permute(0, 2, 1, 3)                                               # [b, 8, 8, 128]
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)                          # [b*8, 8, 128]
            pos_emb = self.rel_pos(xseq)                                                  # [b, 128]
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)                           # [b*8, 8, 128]

            # Apply cross local attention
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape, mask=y['y']['mask_local'])  # [b, 8, 8, 128]

            xseq = xseq.permute(0, 2, 1, 3)                                                # [b, len, 8, 128]
            xseq = xseq.reshape(bs, nframes, -1)                                           # [b, 8, 1024]
            xseq = xseq.permute(1, 0, 2)                                                   # [8, b, 1024]

            # Prepare for transformer encoder - timestept, mixed style (ID1 and ID2), time embed, interloctr gesture
            emb_t2 = self.embed_timestep2(timesteps)
            xi = y['y_inter1']['gesture'].reshape(bs, njoints * nfeats, 1, nframes)      # [b, 8, 768]
            xi_ = self.inter_input_process(xi)                                             # [8, b, 1024]
            xseq = torch.cat((mixed_style + emb_t2, xseq, xi_), axis=0)             # [2*8+1, b, 1024]

            transf_d = xseq.shape[0]                                                       # transformer dimension
            xseq = xseq.permute(1, 0, 2)                                             # [b, 8, 1024]
            xseq = xseq.view(bs, transf_d, self.num_head, -1)                              # [b, 17, 8, 128]
            xseq = xseq.permute(0, 2, 1, 3)                                                #  Need [b, 8, 17, 128]
            xseq = xseq.reshape(bs * self.num_head, transf_d, -1)                          # [b*8, 17, 128]
            pos_emb = self.rel_pos(xseq)                                                   # [17, 128]
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)                            # [b*8, 17, 128]
            xseq_rpe = xseq.reshape(bs, self.num_head, transf_d, -1)                       # [b, 8, 17, 128]
            xseq = xseq_rpe.permute(0, 2, 1, 3)                                            # [b, 17, 8, 128]
            xseq = xseq.view(bs, transf_d, -1)                                             # [b, 17, 1024]
            xseq = xseq.permute(1, 0, 2)                                                   # [17, b, 1024]
            output = self.seqTransEncoder(xseq)[transf_d-nframes:]

            return self.output_process(output)

        else:
            raise NotImplementedError(f"{self.model_version} - forward path is not been implemented yet")

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape      # [b, 768, 1, 8]
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)        # [8, b, 768]
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec',"expmap_74j"]:
            x = self.poseEmbedding(x)               # [8, b, 512], (8, b, 768)
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec', "expmap_74j"]:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError

        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, xf=None, emb=None):
        """
        x: B, T, D      , [240, 2, 256]
        xf: B, N, L     , [1, 2, 256]
        """
        x = x.permute(1, 0, 2)
        # xf = xf.permute(1, 0, 2)
        B, T, D = x.shape
        # N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(x)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        return y


class WavEncoder(nn.Module):

    def __init__(self, source_dim, motion_window_length, audio_feat_dim):
        super().__init__()
        self.pre_feature_map = nn.Linear(motion_window_length, 1)
        self.audio_feature_map = nn.Linear(source_dim, audio_feat_dim)

    def forward(self, rep):
        # features adaptations
        rep = rep.permute(0, 1, 3, 2)
        rep = self.pre_feature_map(rep)
        rep = torch.squeeze(rep, 3)
        rep = self.audio_feature_map(rep)
        return rep


def serial_printer(data):
    for key1, dict in data.items():
        for key2, value in dict.items():
            print(f"{key1}: {key2} shape {value.shape}")

if __name__ == '__main__':

    '''
    cd ./BEAT-main/model
    python mdm.py
    '''

    """
    # FORMER TEST FROM ORIGINAL mdm PAPER:
    # 1.1 same test
    # 1.2 introduced 3rd shape = 18 in audio --> still works

    n_frames = 150
    n_seed = 30
    njoints = 684 * 3
    audio_feature_dim = 1133 + 301  # audio_f + text_f
    style_dim = 2
    bs = 2

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = MDM(modeltype='', njoints=njoints, nfeats=1, cond_mode='cross_local_attention4_style1', audio_feat='wavlm',
                arch='trans_enc', latent_dim=512, n_seed=n_seed, cond_mask_prob=0.1,
                style_dim=style_dim, source_audio_dim=audio_feature_dim, audio_feat_dim_latent=128).to(device)

    x = torch.randn(bs, njoints, 1, n_frames)
    t = torch.tensor([12, 85])

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)  # [..., n_seed:]
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed - n_seed, audio_feature_dim)  # attention5
    model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed, 18,audio_feature_dim)       # attention4
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames, audio_feature_dim)  # attention3
    model_kwargs_['y']['style'] = torch.randn(bs, style_dim)
    model_kwargs_['y']['mask_local'] = torch.ones(bs, n_frames).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]  # attention3/4
    model_kwargs_['y']['seed_last'] = x[..., -n_seed:]  # attention5
    model_kwargs_['y']['gesture'] = torch.randn(bs, n_frames, njoints)
    y = model(x, t, model_kwargs_['y'])  # [bs, njoints, nfeats, nframes]
    print(y.shape)
    """

    """
    # pre test on dimensions
    conv1 = torch.ones((2, 8, 512))
    conv2 = 2 * torch.ones((2, 8, 512))
    conv = torch.cat((conv1.unsqueeze(3), conv2.unsqueeze(3)), axis=3)
    conv = conv.view(2, 8, 8, -1)
    print(f"Conv test shape: {conv.shape}")
    print(conv[1, 1, ...])
    """

    from time import perf_counter

    # FULL MODEL TEST -- to check TAG2G's pipeline adaptation

    n_frames = 8
    n_seed = 1
    njoints = 256*3
    audio_feature_dim = 108 + 302      # audio_f + text_f
    style_dim = 17
    t_frame = 18
    bs = 32

    dyadic_test  = True
    if dyadic_test:
        version = 'dyadic'
        labels = ['y', 'y_inter1']
    else:
        version = 'v0'
        labels = ['y']
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = MDM(modeltype='', njoints=njoints, nfeats=1, cond_mode='cross_local_attention4_style1', audio_feat='wavlm',
                arch='trans_enc', latent_dim=512, n_seed=n_seed, cond_mask_prob=0.1, 
                style_dim=style_dim, source_audio_dim=audio_feature_dim, audio_feat_dim_latent=128, version=version).to(device)
    x = torch.randn(bs, njoints, 1, n_frames)
    t = torch.randint(3, 100, (bs, ))

    # building conditions to test the model in both monadic and dyadic setup
    model_kwargs_ = {}
    for label in labels:

        agent_kwargs_ = {}

        agent_kwargs_['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)     # [..., n_seed:]
        # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed - n_seed, audio_feature_dim)  # attention5
        agent_kwargs_['audio'] = torch.randn(bs, n_frames - n_seed, t_frame, audio_feature_dim)         # attention4
        # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames, audio_feature_dim)  # attention3
        agent_kwargs_['style'] = torch.randn(bs, style_dim)
        agent_kwargs_['mask_local'] = torch.ones(bs, n_frames).bool()
        agent_kwargs_['seed'] = x[..., 0:n_seed]       # attention3/4
        agent_kwargs_['seed_last'] = x[..., -n_seed:]  # attention5
        agent_kwargs_['gesture'] = torch.randn(bs, n_frames, njoints)

        model_kwargs_[label] = agent_kwargs_

    tic = perf_counter()
    y = model(x, t, model_kwargs_)     # [bs, njoints, nfeats, nframes]
    toc = perf_counter()
    print(f"Output shape: {y.shape}, in {round(toc-tic, 6)} seconds")


    """
    # WavEncoder TEST 
    b, sequence, t, feat = 128, 7, 18, 410
    wav_encoder = WavEncoder(source_dim=feat, motion_window_length=t, audio_feat_dim=b)
    random_TA = torch.rand(b, sequence, t, feat, dtype=torch.float32)

    output = wav_encoder(random_TA)
    print(output.shape)
    """