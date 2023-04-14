import torch

from src.models.vit_backbones.vit import Block

class RandomDropBlock(Block):
    
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path") else x

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path") else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.r = self._rd_info["r"]
        
        x_attn, weights = self.attn(self.attention_norm(x))
        x = x + self._drop_path1(x_attn)

        if self.r > 0:
            # Apply random drop
            cls_token, tokens = x[:, :1, :], x[:, 1:, :]
            num_tokens = tokens.size(1)
            
            p = num_tokens - min(self.r, (num_tokens - 1) // 2) # only reduce by a maximum of 50% tokens
            indices = torch.cat([
                torch.randperm(num_tokens)[:p].unsqueeze(0) for _ in range(tokens.size(0))
            ], axis=0).to(tokens.device)
            indices = indices.unsqueeze(2).expand(-1, -1, tokens.size(-1))
            
            selected_tokens = torch.gather(tokens, dim=1, index=indices)
            x = torch.cat([cls_token, selected_tokens], axis=1)

            del cls_token, tokens
        
        self._rd_info["num_tokens_tracker"].append(x.size(1))
        x = x + self._drop_path2(self.ffn(self.ffn_norm(x)))
        return x, weights
    
    
def make_rd_class(transformer_class):
    class RandomDropTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._rd_info["r"] = self.r
            self._rd_info["num_tokens_tracker"] = []

            return super().forward(*args, **kwdargs)
        
        @property
        def num_tokens(self):
            return self._rd_info["num_tokens_tracker"]

    return RandomDropTransformer


def apply_patch(model):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    RandomDropTransformer = make_rd_class(model.__class__)

    model.__class__ = RandomDropTransformer
    model.r = 0
    model._rd_info = {
        "r": model.r,
        "num_tokens_tracker": []
    }

    for module in model.modules():
        if isinstance(module, Block):
            print("Replace with RandomDropBlock")
            module.__class__ = RandomDropBlock
            module._rd_info = model._rd_info