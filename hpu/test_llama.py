import torch

device = "hpu"
use_hpu_graphs = True
max_new_tokens = 128
num_beams = 1
dtype = torch.bfloat16
ignore_eos = True

if device == "hpu":
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()
    ignore_eos = False

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

text = "In the heart of a forgotten city, nestled between towering skyscrapers with rusting skeletons, a hidden community thrived. The Underground, as they called it, was a network of abandoned subway tunnels, repurposed into a maze of makeshift homes, workshops, and hydroponic gardens. Led by Anya, a wiry woman with eyes that held the glint of a thousand scavenged treasures, the Underground had become a haven for outcasts, refugees, and anyone deemed unfit for the harsh reality of the surface. Life wasn't easy. Resources were scarce, and the ever-present threat of radiation storms forced them to be resourceful. Yet, within the dim glow of recycled battery lamps, a spirit of resilience bloomed. Kai, a young boy with a knack for tinkering, was constantly fiddling with salvaged tech, his latest project aimed at capturing rainwater more efficiently. Amara, a former botanist, nurtured their hydroponic crops, coaxing life from nutrient-rich solutions. Even evenings were filled with activity. Stories were exchanged around flickering fires, tales of a past before the Cataclysm, whispered with a mixture of longing and determination. Laughter echoed through the tunnels as children, some born entirely underground, chased each other in games that required more agility than brute strength. One day, a tremor shook the tunnels, a faraway rumble that sent chills down spines. It was a rare earthquake, a reminder of the volatile world they inhabited. Anya called a meeting, her voice firm as she addressed the worried faces. Scouts would be dispatched to assess the damage, to see if new opportunities for resources had opened up. Fear gnawed at them, but there was also a flicker of hope. Perhaps, just perhaps, this tremor was a sign, a nudge towards a future beyond the cramped tunnels of the Underground. As the scouts ventured out, the community held its breath, a silent prayer echoing in the darkness. The future remained uncertain, but one thing was clear: the spirit of the Underground, forged in hardship and ingenuity, would endure."

input_ids = tokenizer.encode(text, return_tensors="pt")

generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        }

if device == "hpu" and use_hpu_graphs == True:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    model = wrap_in_hpu_graph(model)
    
    generate_kwargs.update({
        "lazy_mode": True,
        "hpu_graphs": use_hpu_graphs,
        "ignore_eos": ignore_eos,
        "use_cache": False,
    })

print(generate_kwargs)
model = model.to(device, dtype=dtype)
input_ids = input_ids.to(device)

# Traditional way of generating text
outputs = model.generate(input_ids, **generate_kwargs).cpu()
print("\ngenerate + input_ids:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# From inputs_embeds -- exact same output if you also pass `input_ids`. If you don't
# pass `input_ids`, you will get the same generated content but without the prompt
inputs_embeds = model.model.embed_tokens(input_ids)
outputs = model.generate(inputs_embeds=inputs_embeds, **generate_kwargs).cpu()

print("\ngenerate + inputs_embeds:", tokenizer.decode(outputs[0], skip_special_tokens=True))
