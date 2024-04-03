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

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

dtype = torch.bfloat16
device_cpu = "cpu"
device_accelerator = "cuda"


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)

vl_gpt = vl_gpt.to(dtype)

# to compare result on cpu and accelerator
vl_gpu_cpu = vl_gpt.eval()
vl_gpt_acclerator = vl_gpt.to(device_accelerator).eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["../images/training_pipelines.jpg"],
    },
    {"role": "Assistant", "content": ""},
]


# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs_cpu = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
)


prepare_inputs_accelerator = prepare_inputs_cpu.to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds_cpu = vl_gpt_cpu.prepare_inputs_embeds(**prepare_inputs_cpu)
inputs_embeds_accelerator = vl_gpt_accelerator.prepare_inputs_embeds(**prepare_inputs_accelerator)

print("check equallness of embedding between cpu and accelerator: ", torch.allclose(inputs_embeds_cpu, inputs_embeds_accelerator.to("cpu")))

# run the model to get the response
outputs_cpu = vl_gpt_cpu.language_model.generate(
    inputs_embeds=inputs_embeds_cpu,
    attention_mask=prepare_inputs_cpu.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

# run the model to get the response
outputs_accelerator = vl_gpt_accelerator.language_model.generate(
    inputs_embeds=inputs_embeds_accelerator,
    attention_mask=prepare_inputs_accelerator.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

print("check equallness of outputs between cpu and accelerator: ", torch.allclose(outputs_cpu, outputs_accelerator.to("cpu")))

answer = tokenizer.decode(outputs_accelerator[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
