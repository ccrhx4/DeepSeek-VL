# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Type, Union
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepseek_vl.models.sam import create_sam_vit
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images


model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["../images/training_pipelines.jpg"],
    },
    {"role": "Assistant", "content": ""},
]

if __name__ == "__main__":
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    device_gpu = "cuda"
    dtype = torch.bfloat16

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    )

    # copy from prepare_inputs
    bs, n = prepare_inputs.pixel_values.shape[0:2]
    images_cpu = rearrange(prepare_inputs.pixel_values, "b n c h w -> (b n) c h w").to("cpu", dtype=dtype)

    print(images_cpu.dtype)

    images_gpu = rearrange(prepare_inputs.pixel_values, "b n c h w -> (b n) c h w").to(device_gpu, dtype=dtype)
    #x_cpu = torch.zeros(2, 3, 1024, 1024).to("cpu", dtype=dtype)
    #x_gpu = torch.zeros(2, 3, 1024, 1024).to(device_gpu, dtype=dtype)
    # x.permute(0, 3, 1, 2)
    
    net_cpu = create_sam_vit().to("cpu", dtype=dtype)
    net_gpu = create_sam_vit().to(device_gpu, dtype=dtype)

    out_cpu = net_cpu(images_cpu)
    out_gpu = net_gpu(images_gpu)

    torch.save(out_gpu, 'sam_cuda.pt')

    torch.testing.assert_close(out_gpu.cpu(), out_cpu)
