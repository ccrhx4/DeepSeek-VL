# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Tuple, Union
import habana_frameworks.torch
import torch
import torch.nn as nn

class MlpProjector(nn.Module):
    def __init__(self):
        super().__init__()

        mlp_depth = 2
        input_dim = 1024
        n_embed = 4096

        self.high_up_proj = nn.Linear(input_dim, n_embed // 2)
        self.low_up_proj = nn.Linear(input_dim, n_embed // 2)

        modules = []
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(n_embed, n_embed))
        modules = nn.Sequential(*modules)
        self.layers = modules

    def forward(
        self, x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        if isinstance(x_or_tuple, tuple):
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple

        output = self.layers(x)
        return output


if __name__ == "__main__":
    high_res = torch.load("images_features_0_bf16_hpu.pt").to("hpu")
    low_res = torch.load("images_features_1_bf16_hpu.pt").to("hpu")
    images_features = (high_res, low_res)

    m = MlpProjector().eval().to("hpu", dtype=torch.bfloat16)
    print(images_features[0].dtype)

    out = m(images_features)
    print((out == 0).sum(dim=2))
    torch.save(out.cpu(), "hpu_bug.bf16.pt")
