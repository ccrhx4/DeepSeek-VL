# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepseek_vl.models.sam import create_sam_vit

if __name__ == "__main__":
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    device_gpu = "cuda"
    dtype = torch.float32

    x_cpu = torch.zeros(2, 3, 1024, 1024).to("cpu", dtype=dtype)
    x_gpu = torch.zeros(2, 3, 1024, 1024).to(device_gpu, dtype=dtype)
    # x.permute(0, 3, 1, 2)
    
    net_cpu = create_sam_vit().to("cpu", dtype=dtype)
    net_gpu = create_sam_vit().to(device_gpu, dtype=dtype)
    

    out_cpu = net_cpu(x_cpu)
    out_gpu = net_gpu(x_gpu)

    torch.testing.assert_close(out_gpu.cpu(), out_cpu)
