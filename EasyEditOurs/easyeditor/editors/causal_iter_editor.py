import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from transformers import LlamaTokenizer, LlamaForCausalLM
from ..util.globals import *
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_bias_edit_quality, compute_icl_bias_edit_quality
from ..util import nethook
from ..util.alg_dict import *
from .editor import BaseEditor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class CausalEditor(BaseEditor):
    """Base editor for all methods"""

    def apply_causal_trace(
        self,
        model, 
        prompt,
        subject,
        target,
        noise,
        choosed={},
        samples=4
    ):
        # tokenization
        inp = make_inputs(self.tok, [prompt] * samples)
        if not isinstance(self.tok, LlamaTokenizer): # llama don't need to add additional space.
            target = " "+target
        print("prompt:", prompt, "\nsubject:", subject, "\ntarget: ", target)

        assert subject in prompt
        # Our changes
        original_lprobs = predict_tokens_new(prompt, model, self.tok, target, samples=samples)
        print(original_lprobs)
        # Combine multiple tokens    
        e_range = find_token_range(self.tok, inp["input_ids"][0], subject)
        # corrupt all subject tokens
        low_lprobs = trace_with_patch_new(
            model, self.tok, inp, [], target , e_range, noise=noise, samples=samples
        )
        diff = original_lprobs - low_lprobs
        
        breakpoint()
        top1_index = diff.squeeze(-1).argmax().item()
        
        # TODO: gather prompt and target
        expected_tensor = self.tok(target, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to("cuda")
        
        new_prompt = torch.concat((inp["input_ids"][0], expected_tensor[0,:top1_index]), dim=-1)
        p = self.tok.decode(new_prompt, skip_special_token=True)
        t = self.tok.decode(expected_tensor[0, top1_index])
        prompt = p
        target_new = t.strip()
        
        return prompt, target_new, choosed

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             extra_eval_points=[],
             eval_func=None,
             only_some_step=False,
             causal_target_lprobs_diff=6.55,
             causal_max_edit=5,
             causal_noise_level='s3',
             known_path="../data_construction/outputs/steroset_llama/edit.json",
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        print('Prepare request.')

        requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                          locality_inputs, portability_inputs, 
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        # compute noise 
        # parse noise
        filename = os.path.join(known_path)
        with open(filename) as f:
            knowns = json.load(f)

        noise_level = causal_noise_level
        if isinstance(causal_noise_level, str):
            if causal_noise_level.startswith("s"):
                # Automatic spherical gaussian
                factor = float(causal_noise_level[1:]) if len(causal_noise_level) > 1 else 1.0
                noise_level = factor * collect_embedding_std(
                    self.model, self.tok, [k["subject"] for k in knowns]
                )
                print(f"Using causal_noise_level {causal_noise_level} to match model times {factor}")
            elif causal_noise_level == "m":
                # Automatic multivariate gaussian
                noise_level = collect_embedding_gaussian(mt)
                print(f"Using multivariate gaussian to match model noise")
            elif causal_noise_level.startswith("t"):
                # Automatic d-distribution with d degrees of freedom
                degrees = float(causal_noise_level[1:])
                noise_level = collect_embedding_tdist(mt, degrees)
            elif causal_noise_level.startswith("u"):
                uniform_noise = True
                noise_level = float(causal_noise_level[1:])
        
        # # for test
        # self.apply_causal_trace(
        #     prompt=requests[0]['prompt'],
        #     subject=requests[0]['subject'],
        #     target=requests[0]['target_new'],
        #     noise=noise_level,
        # )
        
        print('Compute pre.')
        all_metrics = []
        locality_metrics = []
        for i, request in enumerate(requests):
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                metrics = {
                    "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
            else:
                # metrics = {
                #     "pre": compute_bias_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                #                             self.hparams.device, test_generation=test_generation)
                # }
                # locality_metrics.append(metrics["pre"]["locality"])
                metrics = {}
                # NOTE: REMEMBER TO UNCOMMENT THIS!!!
            all_metrics.append(metrics)

        print('Start Edit.')
        extra_metrics = {}
        icl_examples = kwargs['icl_examples'] if 'icl_examples' in kwargs else []
        cache_inputs = []
        cache_labels = []
        for i, request in enumerate(requests):
            start = time()
            # eval
            if i in extra_eval_points and eval_func != None:
                print(f'Meet extra eval point: step {i}')
                # edited_data = [d["original_pairs"] for d in requests[:i]]
                # non_edited_data = [d["original_pairs"] for d in requests[i:]]
                edited_data = requests[:i]
                non_edited_data = requests[i:]
                if self.alg_name == 'IKE':
                    # add icl prefix.
                    icl_prefix = "".join(icl_examples)
                    edited_data = [[icl_prefix + item[0], icl_prefix + item[1]] for item in edited_data]
                    non_edited_data = [[icl_prefix + item[0], icl_prefix + item[1]] for item in non_edited_data]
                    
                m_edit = eval_func(model=edited_model, eval_data=edited_data, icl_examples=icl_examples, desc=f"Eval edited samples at Step {i}, total {len(edited_data)} samples")
                m_non_edit = eval_func(model=edited_model, eval_data=non_edited_data, icl_examples=icl_examples, desc=f"Eval unedited samples at Step {i}, total {len(non_edited_data)} samples")
                
                extra_metrics[i] = {}
                extra_metrics[i]['edited'] = m_edit
                extra_metrics[i]['non_edited'] = m_non_edit

                # save models
                # try:
                # alter save path based on steps
                save_dir = kwargs['model_save_dir']
                if not os.path.exists(save_dir):
                   os.makedirs(save_dir, exist_ok=True)
                assert os.path.isdir(save_dir)
                if save_dir.endswith('/'): save_dir = save_dir[:-1]
                if 'seed' not in kwargs:
                    cur_save_dir = f"{save_dir}-step-{i}"
                else:
                    cur_save_dir = f"{save_dir}-step-{i}-seed-{kwargs['seed']}"
                os.makedirs(cur_save_dir, exist_ok=True)
                print(f"Save step {i} model in {cur_save_dir}")
                if self.alg_name == "SERAC":
                    obj = edited_model.state_dict()
                    torch.save(obj, f"{cur_save_dir}/serac.pt")
                    json.dump(edited_model.cache_inputs, open(os.path.join(cur_save_dir, 'cache_inputs.json'), 'w'), indent=4)
                    json.dump(edited_model.cache_labels, open(os.path.join(cur_save_dir, 'cache_labels.json'), 'w'), indent=4)
                elif self.alg_name == "IKE":
                    import pickle
                    with open(f"{cur_save_dir}/icl_exmaples.pkl", 'wb') as fp:
                        pickle.dump(icl_examples, fp)
                else:
                    edited_model.save_pretrained(cur_save_dir)
                if i == extra_eval_points[-1] and only_some_step:
                    return all_metrics, edited_model, weights_copy, extra_metrics, locality_metrics
                # except Exception as e:
                #     print(f'Intermediate model at step {i} cannot be saved due to: {e}')

            edited_model = self.model
            
            prompt_bk = request['prompt']
            target_bk = request['target_new']
            original_lprobs = request['original_lprobs']
            if isinstance(original_lprobs, list):
                assert len(original_lprobs) == 2
                original_lprobs = original_lprobs[0]
            whole_sent = prompt_bk + " " + target_bk
            choosed = {}
            for k in range(causal_max_edit):
                # TODO: compute current lprobs, if it matches target ratio, exit.
                
                from easyeditor.evaluate import run_lprobs
                if k > 0:
                    cur_lprobs = run_lprobs(edited_model, self.tok, whole_sent, device='cuda').item()
                    print(f"Step: {k}\nOriginal lprobs: {original_lprobs}\nCurrent lprobs: {cur_lprobs}.")
                    if original_lprobs - cur_lprobs > causal_target_lprobs_diff:
                        break
                
                # compute current prompt
                cur_prompt, cur_target = self.apply_causal_trace(
                    model=edited_model,
                    prompt=prompt_bk,
                    subject=request['subject'],
                    target=target_bk,
                    noise=noise_level,
                    choosed=choosed,
                )
                request['prompt'] = cur_prompt
                request['target_new'] = cur_target
                
                if isinstance(request["prompt"], list):
                    new_requests = [{"prompt": p, "target_new": t, "subject":request["subject"] }for p, t in zip(request["prompt"], request["target_new"])]
                else: 
                    new_requests = [request]
                print(new_requests)
                
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    new_requests,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    cache_inputs=cache_inputs,
                    cache_labels=cache_labels
                )
            
            request['prompt'] = prompt_bk
            request['target_new'] = target_bk
                
            print(f'Causal Edit Num: {k}')
                
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            all_metrics[i].update({
                'case_id': i,
                "requested_rewrite": request,
                "time": exec_time,
                # "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                "post":compute_bias_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)
            })
            if self.alg_name == 'KN':
                with torch.no_grad():
                    weights_copy() # unpatch_fn
            elif self.alg_name == 'LoRA' and keep_original_weight:
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == "SERAC" and not keep_original_weight:
                cache_inputs = deepcopy(edited_model.cache_inputs)
                cache_labels = deepcopy(edited_model.cache_labels)
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    assert len(all_metrics[i]['post']['locality'][f'{locality_key}_output']) == \
                            len(all_metrics[i]['pre']['locality'][f'{locality_key}_output'])
                    locality_result = []
                    for ans,label in zip(all_metrics[i]['post']['locality'][f'{locality_key}_output'],all_metrics[i]['pre']['locality'][f'{locality_key}_output']):
                        locality_result.append(np.mean(np.equal(ans, label)))
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                all_metrics[i]['pre'].pop('locality')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)
        # if len(extra_eval_points) == 0:
        #     return all_metrics, edited_model, weights_copy
        # else:
        return all_metrics, edited_model, weights_copy, extra_metrics, locality_metrics

@torch.no_grad()
def predict_tokens_new(prompt, model, tokenizer, target, samples=2, inp=None):
    # 1. pad to a single tensor
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if inp is None:
        # NOTE: without any special tokens
        inp = make_inputs(tokenizer, [prompt]*samples) #[1, Lp] 
    
    # since we are dealing with multiple tokens, we do not use padding.
    # default padding on the left would cause error.
    assert isinstance(target, str) # cannot hold batch input now.
    expected_tensor = tokenizer(target, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to("cuda")
    target_input = expected_tensor.repeat(samples,1) 
    input_ids = torch.concat((inp['input_ids'], target_input), dim=-1)[:, :-1] #[N*K, Lp]

    # model forward
    output = model(input_ids=input_ids).logits
    lprobs = F.log_softmax(output, dim=-1)
    lprobs = lprobs[1:,-expected_tensor.shape[1]:].mean(dim=0) # [Lt, |V|]
    
    # 3. gathering
    # expected_tensor = expected_tensor[0,:]
    expected_tensor = expected_tensor[0][:, None] # [L_t, 1]
    gathered_lprobs = lprobs.gather(-1, torch.where(expected_tensor != -100, expected_tensor, 0)).squeeze(0)
    
    return gathered_lprobs

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    # if "[PAD]" in tokenizer.all_special_tokens:
    #     pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    # else:
    #     pad_id = 0
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = tokenizer.eos_token_id
        
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    # print("tok", toks, "substring", substring)
    substring = substring.replace(" ", "")
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array if t != tokenizer.pad_token_id]

def trace_with_patch_new(
    model,  # The model and tokenizer
    tokenizer,
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    target,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
    samples=2,
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    # Backup
    # model = mt.model
    ori_model = model
            
    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    from collections import defaultdict
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(ori_model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # NOTE: MODIFICATION STARTS NOW.
    
    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    
    with torch.no_grad(), nethook.TraceDict(ori_model,[embed_layername] + list(patch_spec.keys()) + additional_layers, edit_output=patch_rep,) as td:
        lprobs = predict_tokens_new(None, model, tokenizer, target, inp=inp, samples=samples)

    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return lprobs, all_traced
    
    return lprobs

def layername(model, num, kind=None):
    #import pdb;pdb.set_trace()
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def collect_embedding_std(model, tokenizer, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(tokenizer, [s])
        with nethook.Trace(model, layername(model, 0, "embed")) as t:
            model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level
