import math

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint


def agg_ref_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float()  # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0)  # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0)  # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate


class FFTFilter(nn.Module):
    def __init__(self, num_channels, sigma):
        super(FFTFilter, self).__init__()

        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.sigma = sigma

        self.laplace = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Args: x: [B,C,H,W]
        """
        x = rearrange(x, 'b l c -> b c 1 l')
        b, c, h, w = x.shape
        x = x.float()

        # compute coef for gaussian 0~1
        coef = self.laplace(x.squeeze(-2)).unsqueeze(-2)
        coef = self.fc(self.pool(coef).view(b, c)).view(b, 1, 1, 1)

        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        high_filter = self._make_gaussian(h_idx, w_idx, h, w, self.sigma, device=x.device)
        y = y * (1 - coef * high_filter)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()

        y = rearrange(y, 'b c 1 l -> b l c')
        x = rearrange(x, 'b c 1 l -> b l c')
        return x + y  # , high_filter

    def _make_gaussian(self, y_idx, x_idx, height, width, sigma=7, device='cpu'):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().to(device)
        xv = xv.unsqueeze(0).float().to(device)
        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))
        return g.unsqueeze(0)


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, r_dim, embed_dim, num_heads, dropout=0.1):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = r_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self.fft_v = FFTFilter(self.embed_dim // 8, 7)
        self.fft_r = FFTFilter(self.embed_dim // 8, 7)

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)  # (bs * 8, -1, embed_dim//8)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)  # (bs * 8, seq_len_img, embed_dim//8)
        key_states = key_states.view(*proj_shape)  # (bs * 8, seq_len_text, embed_dim//8)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (bs * 8, seq_len_img, seq_len_text)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0])

        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # assert attention_mask_l.dtype == torch.int64
        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)  # (bs, seq_len)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, seq_len)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        fft_v = self.fft_v(attn_output_v)
        fft_r = self.fft_r(attn_output_l)
        fft_v = agg_ref_feat(fft_v, torch.ones(size=fft_v.shape[:2], requires_grad=False,
                                               device=fft_v.device))
        fft_r = agg_ref_feat(fft_r, torch.ones(size=fft_r.shape[:2], requires_grad=False,
                                               device=fft_r.device))

        attn_output_v = attn_output_v * fft_r.unsqueeze(-2)
        attn_output_l = attn_output_l * fft_v.unsqueeze(-2)

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, r_dim, embed_dim, num_heads, dropout=0.1,
                 drop_path=.0, init_values=1e-4, ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_r = nn.LayerNorm(r_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            r_dim=r_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_r = nn.Parameter(init_values * torch.ones((r_dim)), requires_grad=True)

    def forward(self, v, r, attention_mask_r=None):
        # v: visual features, (bs, sigma(HW), 256)
        # l: language features, (bs, seq_len, 768)
        v = self.layer_norm_v(v)
        r = self.layer_norm_r(r)
        delta_v, delta_r = self.attn(v, r, attention_mask_l=attention_mask_r)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        r = r + self.drop_path(self.gamma_r * delta_r)
        return v, r


class VR_Fuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self):
        super(VR_Fuse, self).__init__()

        self.img_dim = 256
        # mha params by default
        self.n_head = 8
        self.embed_dim = 2048
        self.ref_dim = 768

        # early fusion module
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(
            v_dim=self.img_dim,  # 256
            r_dim=self.ref_dim,  # 768
            embed_dim=self.embed_dim,  # 2048
            num_heads=self.n_head,  # 8
            dropout=0.1,
            drop_path=.0,
            init_values=1.0 / 6.0,
        )
        # TODO: save memory
        self.use_checkpoint = False

    def forward(self, x: dict):
        visual_features = x["visual"]
        referring_dict_features = x["referring"]

        if self.use_checkpoint:
            fused_visual_features, referring_features = checkpoint.checkpoint(
                self.b_attn,
                visual_features, referring_dict_features['hidden'], referring_dict_features['masks']
            )
        else:
            fused_visual_features, referring_features = self.b_attn(
                visual_features, referring_dict_features['hidden'], referring_dict_features['masks']
            )

        referring_dict_features['hidden'] = referring_features
        fused_referring_dict_features = referring_dict_features

        features_dict = {
            "visual": fused_visual_features, "referring": fused_referring_dict_features
        }

        return features_dict


class VR_Align(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # dot product soft token head
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_referring = nn.Linear(768, 256, bias=True)  # 768 -> 256
        self.log_scale = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(768), requires_grad=True)  # (768，)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)  # size (1,)

    def forward(self, x, embedding):
        """
        x: visual features (bs, num_query, 256)
        embedding: referring features (bs, L, 768)
        """
        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)  # (bs, L, 768) L is maximum sentence length
        dot_product_proj_tokens = self.dot_product_projection_referring(embedding / 2.0)  # 768 -> 256
        dot_product_proj_tokens_bias = torch.matmul(
            embedding, self.bias_lang
        ) + self.bias0  # (bs, L, 768) x (768, ) + (1, ) -> (bs, L)

        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        A = dot_product_proj_queries.shape[1]  # num_query
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)  # (bs, num_query, L)

        dot_product_logit = (torch.matmul(
            dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)
        ) / self.log_scale.exp()) + bias  # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)
        dot_product_logit = torch.clamp(dot_product_logit, max=50000)
        dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        return dot_product_logit
