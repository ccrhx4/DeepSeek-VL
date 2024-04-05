from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

text = "this is a reproducer of the issue that happend in my work"
input_ids = tokenizer.encode(text, return_tensors="pt")

model = model.to("hpu")
input_ids = input_ids.to("hpu")
# Traditional way of generating text
outputs = model.generate(input_ids).cpu()
print("\ngenerate + input_ids:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# From inputs_embeds -- exact same output if you also pass `input_ids`. If you don't
# pass `input_ids`, you will get the same generated content but without the prompt
inputs_embeds = model.model.embed_tokens(input_ids)
outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=16).cpu()
print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))
