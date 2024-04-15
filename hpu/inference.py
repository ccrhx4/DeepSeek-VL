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
import time
import torch

dtype = torch.float
device = "hpu"
use_hpu_graphs = True
max_new_tokens = 512
num_beams = 1
ignore_eos = False

generation_kwargs = {
    "max_new_tokens": max_new_tokens,
    "num_beams": num_beams,
}

if device == "hpu":
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    import habana_frameworks.torch
    adapt_transformers_to_gaudi()

    generation_kwargs.update({
        "lazy_mode": True,
        "hpu_graphs": use_hpu_graphs,
        "ignore_eos": ignore_eos,
    })
else:
    use_hpu_graphs = False

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

def prepare_model(device, dtype):
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    start_load_chat_processor = time.time()
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    print(f"load chat processor took {time.time() - start_load_chat_processor:.2f} seconds")

    if device == "hpu":
        import habana_frameworks.torch.core as htcore

    start_load_model = time.time()
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )

    if use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        vl_gpt.language_model = wrap_in_hpu_graph(vl_gpt.language_model)

    vl_gpt = vl_gpt.to(device, dtype=dtype).eval()
    print(f"load model took {time.time() - start_load_model:.2f} seconds")

    # load images and prepare for inputs
    start_load_image = time.time()
    pil_images = load_pil_images(conversation)
    inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device, dtype=dtype)
    print(f"load images took {time.time() - start_load_image:.2f} seconds")
    
    return vl_gpt, tokenizer, inputs

def generate(vl_gpt, tokenizer, inputs, max_new_tokens):
    generation_kwargs.update({"max_new_tokens": max_new_tokens})

    # run image encoder to get the image embeddings
    start_prepare_inputs = time.time()
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**inputs)
    print(f"vision encoder took {time.time() - start_prepare_inputs:.2f} seconds")
    
    start_generation = time.time()
    # run the model to get the response
    outputs_accelerator = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    duration = time.time() - start_generation
    print(f"generation took {duration:.2f} seconds")

    answer = tokenizer.decode(outputs_accelerator[0].cpu().tolist(), skip_special_tokens=True)
    print(f"{inputs['sft_format'][0]}", answer)
    return inputs_embeds, outputs_accelerator

if __name__ == "__main__":
    vl_gpt, tokenizer, inputs = prepare_model(device, dtype)
    max_tokens_list = [32, 64, 128, 256, 512]
    for max_tokens in max_tokens_list:
        print(f"Running inference with max_new_tokens={max_tokens}")
        inputs_embeds_gpu, output_gpu = generate(vl_gpt, tokenizer, inputs, max_tokens)
