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
import habana_frameworks.torch
import torch

dtype = torch.float
device = "hpu"
use_hpu_graphs = True
max_new_tokens = 512
num_beams = 1
ignore_eos = False

if device == "hpu":
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

generation_kwargs = {
        "lazy_mode": True,
        "hpu_graphs": use_hpu_graphs,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "ignore_eos": ignore_eos,
}

from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images


conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["../images/training_pipelines.jpg"],
    },
    {"role": "Assistant", "content": ""},
    ]

def generate(device, dtype, conversation):
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    if device == "hpu":
        import habana_frameworks.torch.core as htcore

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )

    if use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        vl_gpt.language_model = wrap_in_hpu_graph(vl_gpt.language_model)

    vl_gpt = vl_gpt.to(device, dtype=dtype).eval()

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device, dtype=dtype)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    print("run generation", generation_kwargs)

    # run the model to get the response
    outputs_accelerator = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )

    
    answer = tokenizer.decode(outputs_accelerator[0].cpu().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)
    return inputs_embeds, outputs_accelerator

embeds_gpu, output_gpu = generate(device, dtype, conversation)
