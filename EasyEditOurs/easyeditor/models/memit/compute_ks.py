from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .memit_hparams import MEMITHyperParams


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    track: Optional[str] = None,
):
    
    layer_ks = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in requests
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in requests
                for context_type in context_templates
                for _ in context_type
            ],
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
    # except: breakpoint()
    if track is None: track = 'in'
    if track == 'in': layer_ks = layer_ks[0]
    elif track == 'out': 
        layer_ks = layer_ks[1]
    else: raise

    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    return torch.stack(ans, dim=0)


# get_module_input_output_at_words(model,tok,layer,context_templates=[context.format(request["prompt"]) for request in requests for context_type in context_templates for context in context_type], words=[request["subject"] for request in requests for context_type in context_templates for _ in context_type], module_template=hparams.rewrite_module_tmp, fact_token_strategy=hparams.fact_token)