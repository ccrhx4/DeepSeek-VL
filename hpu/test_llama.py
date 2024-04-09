from transformers import AutoModelForCausalLM, AutoTokenizer


device = "hpu"
use_hpu_graphs = True
max_new_tokens = 32

if device == "hpu":
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

text = "this is a reproducer of the issue that happend in my workplace."
input_ids = tokenizer.encode(text, return_tensors="pt")

generate_kwargs = {"max_new_tokens": max_new_tokens,}

if device == "hpu" and use_hpu_graphs == True:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    model = wrap_in_hpu_graph(model)
    
    generate_kwargs = {
        "lazy_mode": True,
        "hpu_graphs": use_hpu_graphs,
        "max_new_tokens": max_new_tokens,
    }

model = model.to(device)
input_ids = input_ids.to(device)

# Traditional way of generating text
outputs = model.generate(input_ids, **generate_kwargs).cpu()
print("\ngenerate + input_ids:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# From inputs_embeds -- exact same output if you also pass `input_ids`. If you don't
# pass `input_ids`, you will get the same generated content but without the prompt
inputs_embeds = model.model.embed_tokens(input_ids)
outputs = model.generate(inputs_embeds=inputs_embeds, **generate_kwargs).cpu()

print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))
