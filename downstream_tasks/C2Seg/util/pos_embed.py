# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=np.float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    # 对所有位置编码进行调整，但是不用调整decoder
    for layer in checkpoint_model.keys():
        if 'pos_embed' in layer and 'decoder' not in layer:
            # TODO: check patch_embed.proj.weight  patch_embed.proj.bias head.weight head.bias 
            # 没有考虑   cls_embed    cls_token
            print(layer)
            pos_embed_checkpoint = checkpoint_model[layer]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if 'spatial' in layer:
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2]) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int( getattr(model, layer, None).shape[-2] ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Spatial position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    num_extra_tokens = 0
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    # pos_tokens = torch.nn.functional.interpolate(
                    #     pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens[:,:,:new_size,:new_size]
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model[layer] = new_pos_embed
            elif 'temporal' in layer:
                orig_size = int(pos_embed_checkpoint.shape[-2])
                new_size = int(getattr(model, layer, None).shape[-2])
                if orig_size != new_size:
                    print("Temporal position %s interpolate from %d to %d" % (layer, orig_size, new_size))
                    pos_tokens = pos_embed_checkpoint.reshape(-1, orig_size, embedding_size).permute(0, 2, 1).unsqueeze(2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(1,new_size), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.squeeze(2).permute(0, 2, 1)
                    checkpoint_model[layer] = new_pos_embed
            else:
                # TODO： 其他场景
                raise ValueError("Unknown layer name: {}".format(layer))
                return 0



