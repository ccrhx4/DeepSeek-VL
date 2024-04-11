import torch 
import torch.nn as nn

import torchvision.transforms as transforms
import habana_frameworks.torch as ht

class ResizeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resizer = transforms.Resize((128, 128))

    def forward(self, x):
        x = self.resizer(x)
        return x

device = "hpu"
x = torch.randn(torch.Size([1, 64, 502, 502]))

model = ResizeModel()
model = ht.hpu.wrap_in_hpu_graph(model)
x = x.to(device)
model(x)
