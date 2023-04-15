import torch
import torch.nn.functional as F

from timm.models.vision_transformer import Block

class ProtoBlock(Block):
    
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.r, self.K, self.mode = self._proto_info["r"], self._proto_info["K"], self._proto_info["mode"]
        assert self.K is not None and self.mode is not None, f"Please set proto params before forward, got K={self.K}, mode={self.mode}"

        x_attn = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)

        if self.r > 0:
            # Apply prototyping
            cls_token, tokens = x[:, :1, :], x[:, 1:, :]
            
            num_tokens = tokens.size(1)
            knn_k = int(num_tokens * self.K)
            
            # dist_mat = ((tokens.detach().unsqueeze(2) - tokens.detach().unsqueeze(1)) ** 2).sum(-1)
            tokens_norm = F.normalize(tokens.detach(), dim=-1)
            dist_mat = (2 - 2 * torch.matmul(tokens_norm, tokens_norm.permute(0, 2, 1)))
            
            d_knn, _ = dist_mat.topk(knn_k, largest=False)

            neighbors_dist = d_knn.mean(dim=-1)
            score_first_order = 1 / neighbors_dist
            
            # del dist_mat, d_knn, ind_knn, neighbors_dist, tokens_norm
            
            p = num_tokens - min(self.r, (num_tokens - 1) // 2) # only reduce by a maximum of 50% tokens
            
            if self.mode == "min":
                _, indices = score_first_order.topk(k=p, axis=1, largest=True)
            elif self.mode == "max":
                _, indices = score_first_order.topk(k=p, axis=1, largest=False)
            elif self.mode == "minmax":
                _, indices_min = score_first_order.topk(k=p // 2, axis=1, largest=True)
                _, indices_max = score_first_order.topk(k=p - p // 2, axis=1, largest=False)
                indices = torch.cat([indices_min, indices_max], axis=1)
            else:
                raise ValueError
            
            # del score_first_order
            
            indices = indices.unsqueeze(2).expand(-1, -1, tokens.size(-1))
            selected_tokens = torch.gather(tokens, dim=1, index=indices)

            if self._proto_info["vis"]:
                self._proto_info["idx_tracker"].append(indices)

            # del tokens

            x = torch.cat([cls_token, selected_tokens], axis=1)
        
        self._proto_info["num_tokens_tracker"].append(x.size(1))
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


def make_proto_class(transformer_class):
    class ProtoVisionTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._proto_info["r"] = self.r
            self._proto_info["num_tokens_tracker"] = []
            self._proto_info["idx_tracker"] = []

            return super().forward(*args, **kwdargs)
        
        @property
        def num_tokens(self):
            return self._proto_info["num_tokens_tracker"]

        @property
        def idx_tracker(self):
            return self._proto_info["idx_tracker"]

    return ProtoVisionTransformer


def apply_patch(model, K, mode, vis=False):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ProtoVisionTransformer = make_proto_class(model.__class__)

    model.__class__ = ProtoVisionTransformer
    model.r = 0
    model._proto_info = {
        "r": model.r,
        "K": K,
        "mode": mode,
        "num_tokens_tracker": [],
        "vis": vis,
        "idx_tracker": []
    }

    for module in model.modules():
        if isinstance(module, Block):
            print("Replace with ProtoBlock_v1")
            module.__class__ = ProtoBlock
            module._proto_info = model._proto_info