from easyeditor.models.memit.memit_main import *
import datasets
from easyeditor import ROMEHyperParams
from transformers import LlamaTokenizer, LlamaForCausalLM
import random
from edit_single import set_all_seed
import torch
import tqdm
import pickle
import json
import math
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


set_all_seed(42)
long_ctx_len = 5000
root = os.environ['ROOT']
hparams = ROMEHyperParams.from_hparams(f"{root}/our_hparams/ROME/llama-7b.yaml")

# read original keys
alg = 'ROME'
collected_outputs = torch.load(f"{root}/results_llama/fact_edit/rome/metrics/{alg}_collect_seed0.pt")
with open(f"{root}/results_llama/fact_edit/rome/metrics/{alg}_results_seed0.json", "r") as f:
    collected_results = json.load(f)

# load metrics data
input_file = f"{root}/results_llama/fact_edit/rome/metrics/{alg}_results_seed0.json"
with open(input_file, 'r', encoding='utf8') as f:
    jss = json.load(f)

subjects = [item['requested_rewrite']['subject'] for item in jss]
# subject_filter_keys = set([sub.split()[-1] for sub in subjects])

original_keys = [collected_outputs['pre'][i]['ks']['original'] for i in range(len(collected_outputs['pre']))]
original_keys = torch.cat(original_keys, dim=0).squeeze()

inv_cov = torch.load('EasyEditOurs/data/stats/_llama-2-7b/wikitext_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.inv_conv.pt')

subset = 'train'
wiki_data = datasets.load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1')[subset]

tok = LlamaTokenizer.from_pretrained(hparams.model_name)
tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'right'

# load model
model = LlamaForCausalLM.from_pretrained(hparams.model_name, device_map='auto' if hparams.model_parallel else None)

device = 'cuda'

# tok subjects for filtering
tok_subjects = [tok(subj, add_special_tokens=False)['input_ids'] for subj in subjects]
tok_last_subjects = [it[-1] for it in tok_subjects]
tok_last_subjects = torch.tensor(tok_last_subjects, device=device)

# Tokenize the entire dataset
def tokenize_function(examples):
    return tok(examples["text"], truncation=False, padding=False)

tokenized_dataset = wiki_data.map(tokenize_function, batched=True, remove_columns=wiki_data.column_names)

# Concatenate all tokenized texts
all_input_ids = []
for example in tokenized_dataset:
    all_input_ids.extend(example["input_ids"])

# prepare right vector
original_keys = original_keys.cuda()
right = original_keys @ inv_cov.cuda()

# Create a custom dataset
class WikiTextDataset(Dataset):
    def __init__(self, input_ids, seq_length):
        self.input_ids = input_ids
        self.seq_length = seq_length

    def __len__(self):
        return (len(self.input_ids) - 1) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 for the target
        chunk = torch.tensor(self.input_ids[start_idx:end_idx])
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:]
        }

# Create the custom dataset
seq_length = 1024
custom_dataset = WikiTextDataset(all_input_ids, seq_length)

# Create DataLoader
batch_size = 8
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

def extend_mask(mask, k):
    # Ensure mask is a boolean tensor
    if mask.dtype != torch.bool:
        mask = mask.bool()
    
    # Convert boolean mask to float
    mask_float = mask.float()
    
    # Create the convolution kernel
    kernel_size = k + 1
    
    if mask.dim() == 1:
        # 1D case
        mask_float = mask_float.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, kernel_size, device=mask.device)
        extended = F.conv1d(mask_float, kernel, padding=k)
        extended = (extended > 0).squeeze(0).squeeze(0)
        extended = extended[:-k]
    
    elif mask.dim() == 2:
        # 2D case
        mask_float = mask_float.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, 1, kernel_size, device=mask.device)
        extended = F.conv2d(mask_float, kernel, padding=(0, k))
        extended = (extended > 0).squeeze(0).squeeze(0)
        extended = extended[:, :-k]
    
    else:
        raise ValueError("Input mask must be 1D or 2D")
    
    return extended

subject_topk_dict = {subj:[] for subj in subjects}
prefix_len = 20

max_batch_cnt = 1000

for batch in tqdm.tqdm(dataloader):
    max_batch_cnt -= 1
    if max_batch_cnt <= 0: break
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    match_subj_mask = (tok_last_subjects[:, None] == inputs.reshape(-1)[None, :]) # N, bsz * L
    # extend to next k tokens, True is to be filtered.
    match_extend_mask = extend_mask(match_subj_mask, k=10).transpose(0, 1) # [bsz*L, N]

    assert len(hparams.layers) == 1
    layer = hparams.layers[0]
    with torch.no_grad():
        with nethook.TraceDict(
        module=model,
        layers=[
            hparams.rewrite_module_tmp.format(layer),
        ],
        retain_input=True,
        retain_output=True,
        ) as tr:
            _ = model(inputs)
        repr = tr[hparams.rewrite_module_tmp.format(layer)].input
        left = repr.reshape([-1, repr.shape[-1]])
        similarities = torch.matmul(left , right.T) # [bsz*l, N]
        # similarities = torch.inner(left, right) 
        # filter similarities
        similarities[match_extend_mask] = float('-inf')
        
        topk_similarities = torch.topk(similarities, 10, largest=True, sorted=True, dim=0) # [10, N]
        for i in range(len(subjects)):
            subj = subjects[i]
            topk_values, topk_indices = topk_similarities[0][:, i], topk_similarities[1][:, i]
            # gather prefix
            prefixs = []
            for j in range(10):
                end = topk_indices[j]
                bi = end // seq_length
                start = max(bi * seq_length, end-prefix_len)
                ls, le = start - bi * seq_length, end - bi * seq_length, 
                prefix = inputs[bi, ls:le+1]
                # prefix_str = " ".join(tok.convert_ids_to_tokens(prefix))
                prefix_str = tok.decode(prefix, skip_special_tokens=True)
                prefixs.append(prefix_str)
            topk_values = topk_values.cpu().tolist()
            # zip into a tuple list
            vp_tup_list = [(a, b) for a, b in zip(topk_values, prefixs)]
            subject_topk_dict[subj] += vp_tup_list
            # sort and only leave the top 10.
            subject_topk_dict[subj].sort(reverse=True)
            subject_topk_dict[subj] = subject_topk_dict[subj][:10]
            if i == 0:
                print(f"Subject: {subj}: {subject_topk_dict[subj]}")

with open('./unrelated_collect.top10.pkl', 'wb') as f:
    pickle.dump(subject_topk_dict, f)
