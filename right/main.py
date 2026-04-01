import torch
import tqdm
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
import transformer_lens.utils as utils


import src.utils as utils
from src.token_ds import TokenDataset
from src.extract_fv import avg_indirect_effect, average_last_prompt_activation

##### IN-CONTEXT LEARNING FOR PART-OF-SPEECH IDENTIFICATION #####
gpt2 = HookedTransformer.from_pretrained("gpt2")
samples = pd.read_csv("samples.csv", nrows=1000)
ds = TokenDataset(data=samples)
dl = ds.to_dataloader(gpt2, 16, utils.answer_pos_collate)

cache = average_last_prompt_activation(gpt2, dl)
torch.save(cache, "cache.pth")


###### EXTRACT FUNCTION VECTOR FOR PART-OF-SPEECH IDENTIFICATION ######
ds = load_dataset("datablations/c4-filter-small")

text = ds["train"]["text"][:500]

## important attention heads that we found through activation patching
## see `extract_fv.py`
attn = [
    "0.1", "0.4", "0.6", "4.11", "5.1", "5.5", "6.8", "6.9", "6.10", "7.11"
]

cache = torch.load("cache.pth")
fv = None
# create function vector
for at in attn:
    l, h = at.split(".")
    l = int(l)
    h = int(h)

    if fv is None:
        fv = cache[utils.get_act_name("z", l)][h,:]
    else:
        fv += cache[utils.get_act_name("z", l)][h,:]

fv /= len(attn)

torch.save(fv.to("cpu"), "pos_fv.pth")
gpt2 = HookedTransformer.from_pretrained("gpt2")

##### EXTRACT AVERAGE LAST TOKEN ACTIVATION FOR GPT-2 IN NEXT TOKEN PRED #####
bs = 1, MAX_EX = 500
for t in tqdm.tqdm(range(0, min(len(text), MAX_EX), bs)):
    logits, cache = gpt2.run_with_cache(text[t:min(bs*(t+1), len(text))])

    c = dict()
    for k, v in cache.items():
        c[k] = v.to("cpu")

    #### STORE THE ACTIVATIONS FOR EACH EXAMPLE SEPARATELY #####
    torch.save(c, f"lm{t}.pth")
