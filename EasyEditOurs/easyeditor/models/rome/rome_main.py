from copy import deepcopy
from typing import Dict, List, Tuple
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import agg_utils

from ...util import nethook
# from agg_utils import AverageMeter, get_context_templates

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams
from ..rome import repr_tools
from ..memit.compute_ks import *

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    # request_for_collect_key = [{"prompt": request["prompt"].replace(request["subject"], "{}"), "subject": request["subject"], "target_new": " " + request["target_new"]}]
    # original_key  = compute_ks(model, tok, request_for_collect_key, hparams, hparams.layers[0], context_templates=[['{}']]).detach().cpu()
    deltas, keys, values = execute_rome(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")
    
    if hparams.use_lora_aggregator:
        assert len(hparams.layers) == 1
        assert hparams.use_our_aggregator is False, "You cannot set both use_lora_aggregator and use_our_aggregator to True!!"
        model = train_aggregator(model, tok, request, hparams, hparams.layers[0], keys)
    elif hparams.use_our_aggregator:
        from ..aggregator import Aggregator
        model = Aggregator(hparams, model, tok, device='cuda')
        model.edit(request, keys)
        

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    # print(
    #     f"Executing ROME algorithm for the update: "
    #     f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    # )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            agg_utils.get_context_templates_rome(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            agg_utils.get_context_templates_rome(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, left_vector, right_vector


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def train_aggregator(model, tok, request, hparams, layer, target_vector):
    from peft import get_peft_model, TaskType, LoraConfig
    if hasattr(model,'peft_config'): breakpoint()
    
    # make sure target vector is not changed
    target_vector = target_vector.detach().clone()
    
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    Config = LoraConfig
    
    context_templates = agg_utils.get_context_templates(model, tok, hparams.context_template_length_params)

    # get inv_cov
    from .compute_u import get_inv_cov
    C = get_inv_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
        hparams=hparams,
    )
    dd = torch.sqrt(torch.diag(C))
    norm = torch.outer(dd, dd)
    if hparams.agg_norm_c:
        norm_C = C / norm
    else:
        norm_C = C

    # define loss weights
    loss_weights = {
        'agg': hparams.agg_loss_weight,
        'spread': hparams.spread_loss_weight,
        'kl': hparams.kl_loss_weight,
    }
    
    # if 'llama' in hparams.model_name.lower():
    #     target_modules = ['up_proj', 'gate_proj'] # key generations
    # else:
    #     raise
    # target_modules = hparams.agg_lora_module
    if 'llama' in hparams.model_name.lower():
        target_modules = ['up_proj']
    elif 'gpt2' in hparams.model_name.lower():
        target_modules = ["c_proj", ]
    else:
        raise
    
    peft_config = Config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.agg_lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=target_modules
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    
    lr = 5e-3
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=lr,
        weight_decay=0,
    )
    peft_model.print_trainable_parameters()
    
    loss_meter = agg_utils.AverageMeter()
    
    # Tokenize target into list of int token IDs
    # target_ids = tok(request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
    #     "input_ids"
    # ][0]
    
    # context_templates = get_context_templates(model, tok, )
    # fill in prompt, leave subject blank
    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        prompt_template = request['prompt'].replace(request['subject'], '{}')
    else:
        prompt_template = request['prompt']

    prompt_templates = [prompt_template, ]
    context_templates = context_templates[:]
    
    # add rephrase templates
    if hparams.agg_add_rephrases:
        rep_p_templates = [reph_prompt.replace(request['subject'], "{}") for reph_prompt in request['rephrase_prompt']]
        # filter if there are multiple {}
        # rep_p_templates = [text for text in rep_p_templates if len(re.findall(r'\{\}', text)) == 1]
        prompt_templates += rep_p_templates
    if hparams.agg_add_para_attack:
        para_attacks = request['para_attack']
        para_attack_templates = [reph_prompt.replace(request['subject'], "{}") for reph_prompt in para_attacks]
        # filter if there are multiple {}
        # para_attack_templates = [text for text in para_attack_templates if len(re.findall(r'\{\}', text)) == 1]
        prompt_templates += para_attack_templates
    if hparams.agg_add_long_context:
        assert 'long_context' in request
        random_text = request['long_context']
        context_templates += [random_text,]

    templates = [
        c_templ.format(p_templ) for c_templ in context_templates for p_templ in prompt_templates
    ]
    # filtering and dedup
    templates = list(set(templates))
    templates = [text for text in templates if len(re.findall(r'\{\}', text)) == 1]
    def is_valid_format_string(s):
        stack = []
        for char in s:
            if char == '{':
                stack.append(char)
            elif char == '}':
                if not stack:
                    return False  # Unmatched closing brace
                stack.pop()
        return len(stack) == 0  # True if all braces are matched
    templates = [s for s in templates if is_valid_format_string(s)]

    print(f'# of templates: {len(templates)}')
    subtoken=hparams.fact_token[len("subject_") :]
    
    # repha subjects
    if hparams.agg_half_reph_subj:
        reph_subjects = request['rephrase_subjects'][:5] # only use first 5
    else:
        reph_subjects = request['rephrase_subjects']
    
    subjects = reph_subjects
    if hparams.agg_add_ori:
        subjects += [request['subject'],]
    if hparams.agg_add_shuffle:
        subjects += request['shuffled_subject']
        
    words = [word for word in subjects for _ in templates]
    stack_templates = [temp for word in subjects for temp in templates]
    contexts = [stack_templates[i].format(words[i]) for i in range(len(words))] # full inputs
    idxs = repr_tools.get_words_idxs_in_templates(tok, stack_templates, words, subtoken)
        
    # reph_words = [word for word in reph_subjects for _ in range(len(templates))]
    # reph_templates = [temp for word in reph_subjects for temp in templates]
    # contexts = [reph_templates[i].format(reph_words[i]) for i in range(len(reph_words))] # full inputs
    
    # words = [request["subject"] for _ in range(len(templates))]
    # ori_idxs = repr_tools.get_words_idxs_in_templates(tok, templates, words, subtoken)
    # ori_contexts = [templates[i].format(words[i]) for i in range(len(words))] # full inputs
    # concat
    # if hparams.agg_add_ori:
        # contexts += ori_contexts
        # idxs += ori_idxs

    import random
    batch_size = hparams.agg_batch_size
    index = 0
    # create data tuple
    ctx_idx_tup = [(i, contexts[i], idxs[i]) for i in range(len(contexts))]
    random.shuffle(ctx_idx_tup)

    # init_kl_repr = None
    for it in range(hparams.num_train_agg_steps):
        loss_meter.reset()
        opt.zero_grad()
        def get_next_batch(data, index):
            if index + batch_size > len(data):
                # Reshuffle the list if we've reached the end
                random.shuffle(data)
                return data[:batch_size], 0
            else:
                return data[index:index + batch_size], index + batch_size
        
        batch, index = get_next_batch(ctx_idx_tup, index)
        batch_id = [item[1] for item in batch]
        batch_contexts = [item[1] for item in batch]
        batch_idxs = [item[2] for item in batch]
        contexts_tok = tok(
            batch_contexts,
            return_tensors="pt",
            padding=True,
        ).to(f"cuda:{hparams.device}")

        if 'llama' in hparams.model_name.lower():
            with nethook.TraceDict(
                module=model,
                layers=[
                    "model.layers.{}.mlp.down_proj".format(layer),
                    "model.layers.{}.mlp.up_proj.base_layer".format(layer),
                    "model.layers.{}.mlp.gate_proj".format(layer),
                ],
                retain_input=True,
                retain_output=True,
                # edit_output=out_fn,
            ) as tr:
                _ = model(**contexts_tok)
            repr = tr["model.layers.{}.mlp.down_proj".format(layer)].input # [bsz, l, D]
            # compute init kl repr
            base_up = tr["model.layers.{}.mlp.up_proj.base_layer".format(layer)].output
            base_gate = tr["model.layers.{}.mlp.gate_proj".format(layer)].output
            act_fn = model.get_submodule("model.layers.{}.mlp.act_fn".format(layer))
            init_kl_repr = (act_fn(base_gate) * base_up)

        elif 'gpt2' in hparams.model_name.lower():
            with nethook.TraceDict(
                module=model,
                layers=[
                    "transformer.h.{}.mlp.c_fc".format(layer),
                    hparams.rewrite_module_tmp.format(layer),
                ],
                retain_input=True,
                retain_output=True,
                # edit_output=out_fn,
            ) as tr:
                _ = model(**contexts_tok)
            repr = tr[hparams.rewrite_module_tmp.format(layer)].input # [bsz, l, D]
            init_kl_repr = tr["transformer.h.{}.mlp.c_fc".format(layer)].output
        else:
            raise
            
        # gather loss
        assert all(len(idx)==1 for idx in batch_idxs)
        idxs_t = torch.tensor(batch_idxs).long().to(repr.device)
        # agg_repr = repr[[torch.arange(21).cuda()[:, None], torch.tensor(idxs).cuda()]][0]
        
        mask = torch.ones(repr.size(0), repr.size(1), dtype=torch.bool).to(repr.device)
        batch_indices = torch.arange(repr.size(0))[:, None].to(mask.device)
        mask[batch_indices, idxs_t] = False
        
        agg_repr = repr[~mask]
        # before and after lora.
        kl_repr = repr[mask]
        init_kl_repr = init_kl_repr[mask]
        # if init_kl_repr is None: init_kl_repr = kl_repr.detach().clone()
        # v1: mse
        # agg_loss = mse_loss(agg_repr, target_vector[None])
        # v2: k C^-1 k
        # agg_losses = -1 * (agg_repr @ C @ target_vector)
        # agg_loss = agg_losses.mean()
        
        # v3: k1^T (C / norm) k2
        # agg_losses = -1 * torch.abs(agg_repr @ norm_C @ target_vector)
        # agg_loss = agg_losses.mean()
        # v4: norm agg_repr and kl_repr
        if hparams.agg_loss_norm:
            norm_agg_repr = agg_repr / agg_repr.norm(dim=-1, keepdim=True)
        else:
            norm_agg_repr = agg_repr
        # agg_losses = -1 * torch.abs(norm_agg_repr @ norm_C @ target_vector)
        # v5: remove abs in agg loss
        if not hparams.agg_remove_c:
            agg_losses = -1 * (norm_agg_repr @ norm_C @ target_vector)
        else:
            agg_losses = -1 * (norm_agg_repr @ target_vector)
            
        agg_loss = agg_losses.mean()
        # v6: add kl loss
        # norm_kl_repr = kl_repr / kl_repr.norm(dim=-1, keepdim=True)
        if hparams.kl_loss_type == 'whiten_diff':
            if hparams.agg_remove_c:
                kl_loss = torch.abs(init_kl_repr @ (kl_repr - init_kl_repr).T)
            else:
                kl_loss = torch.abs(init_kl_repr @ norm_C @ (kl_repr - init_kl_repr).T)
        elif hparams.kl_loss_type == 'whiten':
            if not hparams.agg_remove_c:
                kl_loss = -1 * init_kl_repr @ norm_C @ kl_repr.T
            else:
                kl_loss = -1 * init_kl_repr @ kl_repr.T
        elif hparams.kl_loss_type == 'mse':
            import torch.nn.functional as F
            kl_loss = F.mse_loss(init_kl_repr, kl_repr, reduction='mean')
        else:
            raise
            # kl_loss = torch.abs(init_kl_repr @ norm_C @ (kl_repr - init_kl_repr).T)
        kl_loss = kl_loss.mean()
        # kl_loss = 0.0
        
        # v5: min(k1^T (C / norm) k2)
        # agg_losses = -1 * torch.abs(agg_repr @ norm_C @ target_vector)
        # agg_loss = agg_losses.min()
        
        # kl_loss = mse_loss(kl_repr, init_kl_repr)
        # kl_loss = torch.abs(kl_repr)
        if not hparams.agg_remove_c:
            spread_loss = torch.abs(kl_repr @ norm_C @ target_vector).mean()
        else:
            spread_loss = torch.abs(kl_repr @ target_vector).mean()
        loss = agg_loss * loss_weights['agg'] + spread_loss * loss_weights['spread'] + kl_loss * loss_weights['kl']
        print(f"Step {it}: Agg loss {agg_loss.item()}; Spread loss: {spread_loss}; KL loss: {kl_loss}")
        # print(f"Step {it}: Agg loss {agg_loss.item()}; KL loss: {kl_loss}")
        # print(f"Step {it}: Agg loss worst {agg_losses.max()}")
        loss_meter.update(loss.item(), n=1)
        loss.backward()
        opt.step()
    
    return peft_model