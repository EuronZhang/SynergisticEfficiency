
 # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
import torch.nn as nn
import math

# from timm.models.vision_transformer import Attention, Block, VisionTransformer

from src.models.vit_backbones.vit import Block, Attention
from src.models.vit_prompt.vit import PromptedTransformer

from e_vit.utils import complement_idx


class EvitAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(self, x):
        B, N, C = x.shape

        mixed_query_layer = self.query(x) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, attention_probs


class EvitBlock(Block):
    """
    Modifications:
     - put the evit operation in between Attention and Mlp,
       not the original coded in Attention.
       to align with ToMe.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path") else x

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path") else x

    def forward(self, x: torch.Tensor):
        # size
        B, N, C = x.shape
        # print(N)

        # attention
        x_attn, attention_probs = self.attn(self.attention_norm(x))
        x = x + self._drop_path1(x_attn)

        # Token Reduction
        keep_rate = self._evit_info["r"].pop(0) # pop a keep ratio from iterative keep rate list, not the original keep rate list
        
        # print(self._evit_info, keep_rate)

        if keep_rate < 1:
            left_tokens = math.ceil(keep_rate * (N - 1))
            cls_attn = attention_probs[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self._evit_info["fuse"]:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        # MLP
        x = x + self._drop_path2(self.ffn(self.ffn_norm(x)))

        return x, x_attn


def make_evit_class(transformer_class):
    class EvitVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            # copy a keep rate list for iteration across blocks
            rr = self._evit_info["keep_rate"] # int not involved in copy issue
            self._evit_info["r"] = [1,1,1,rr,1,1,rr,1,1,rr,1,1]

            return super().forward(*args, **kwdargs)

    return EvitVisionTransformer


def apply_patch(
    model: PromptedTransformer, keep_rate:float = 0.7, fuse = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.
    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    EvitVisionTransformer = make_evit_class(model.__class__)

    model.__class__ = EvitVisionTransformer
    model._evit_info = {
        "keep_rate": keep_rate,
        "fuse": fuse,
    }

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = EvitBlock
            module._evit_info = model._evit_info
        elif isinstance(module, Attention):
            module.__class__ = EvitAttention