"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    # test_prediction_acc,
    test_generation_quality, 
    PPL,
    kl_loc_loss,
    es_sent,
    run_lprobs
)
import json
import pickle
import os
from .evaluate_utils import test_prediction_acc_ours as test_prediction_acc
from itertools import permutations
import random
from ..models.memit.memit_main import *
from ..models.memit.compute_ks import *
from ..models.memit.compute_z import *
import datasets
import copy

current_file_dir = os.path.dirname(os.path.abspath(__file__))
rephrases = pickle.load(open(f"{current_file_dir}/../../../InfoDeletionAttacks/third_party/data/parap_all_new.pkl","rb"))
try:
    if not os.path.exists("../data_construction/wikitext-103-raw-v1"):
        wiki_data = datasets.load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1')#, download_mode="force_redownload", revision="master")
        wiki_data.save_to_disk("../data_construction/wikitext-103-raw-v1")
    else:
        # load_from_disk
        print('Directly load from disk!')
        wiki_data = datasets.load_from_disk("../data_construction/wikitext-103-raw-v1")
except:
    print('Failed to load Wiki Data. Skip evaluating long context!!!!!')
    wiki_data = None

def compute_bias_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    post = True,
    collect = 'rephrase_long_shuffle',
    long_ctx_len = 300,
    dump=True,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    
    # First, unpack rewrite evaluation record.
    # original_lprobs = record["original_lprobs"]
    ret = {}
    
    prompt = record["prompt"]
    target_new = record['target_new']

    acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
    ret['acc'] = acc
    
    # compute reversion
    target_old = record['target_true']
    print(prompt)
    assert target_old != target_new
    acc = test_prediction_acc(model, tok, hparams, prompt, target_old, device)
    ret['rev'] = acc
    
    # compute generalize
    if 'rephrase_prompt' in record.keys() and record["rephrase_prompt"] != None and post:
        ret['generalization'] = []
        for p in record["rephrase_prompt"]:
            if p != "":
                gen_acc = test_prediction_acc(model, tok, hparams, p, record["rephrase_target"], device)
                ret['generalization'].append(gen_acc)
 

    # if post:
    #     if 'rephrase_prompt' in record.keys() and 'rephrase_target' in record.keys() and len(record["rephrase_target"]) > 0:
    #         ret["para_attack_succ"], ret["para_attack_rev"] = para_attack(record["prompt"].strip(), record["target_true"].strip(), record["target_new"].strip(), tok, model)
   

    # compute locality
    ret['locality'] = {}
    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], str) and isinstance(record['locality'][locality_key]['prompt'], list):
                prompts = record['locality'][locality_key]['prompt']
                targets = [record['locality'][locality_key]['ground_truth']]*len(prompts)
            # init locality
            
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         prompts,
                                         targets, device=device))
    
    # original_hparams = hparams.rewrite_module_tmp
    # if post: hparams.rewrite_module_tmp = "model." + hparams.rewrite_module_tmp
    # # para_attack
    # try:
    #     paraps = rephrases[(prompt+" "+target_old).strip()]
    #     paraps = [prompt] + paraps
    #     if len(paraps)>5:
    #         num_attack_parap = 5
    #     else: 
    #         num_attack_parap = len(paraps)
    #     attack_paraps = paraps[-num_attack_parap:]
    #     attack_paraps = [x.replace(target_old,"") for x in attack_paraps]
    #     ret['para_attack'] = attack_paraps
    # except:
    #     ret['para_attack'] = []
    # collecting function
    collect_keys = set(collect.split('_'))
    # rephrase_subjects
    context_templates = [['{}']]
    # context_templates = get_context_templates(model, tok)
    rephrase_ks = []
    rephrase_vs = []
    rephrase_vvs = []
    rephrase_accs = []
    rephrase_reverision = []
    if "rephrase_subjects" in record.keys() and any(record['rephrase_subjects']) and 'rephrase' in collect_keys:
        for re_sub in record['rephrase_subjects']:
            re_sub_prompt = record['prompt'].replace(record["subject"], re_sub)
            rephrase_subject_acc = test_prediction_acc(model, tok, hparams, re_sub_prompt, target_new if post else target_old, device)
            rephrase_accs.append(rephrase_subject_acc)
            rephrase_subject_reversion = test_prediction_acc(model, tok, hparams, re_sub_prompt, target_old, device)
            rephrase_reverision.append(rephrase_subject_reversion)
        if post:
            ret["edited_rephrase_sub_target_new_acc"] = rephrase_accs
            ret["edited_rephrase_sub_target_new_reversion"] = rephrase_subject_reversion
        else:
            ret["original_rephrase_sub_target_old_acc"] = rephrase_accs

        ret['rephrase_subject'] = record['rephrase_subjects']

        # collect
        if dump:
            for re_sub in record['rephrase_subjects']:
                # rephrase_subject_target_lprobs = run_lprobs(model, tok, record["target_new"], "cuda:0", prefix=re_sub_prompt)
                # all_rephrase_sub_lprobs.append(rephrase_subject_target_lprobs.item())
                
                # collect key for rephrase subject
                rephrase_k = []
                rephrase_v = []
                rephrase_vv = []
                request = [{"prompt": record["prompt"].replace(record["subject"], "{}"), "subject": re_sub, "target_new": " " + record["target_new"]}]
                for i, layer in enumerate(hparams.layers):
                    layer_ks = compute_ks(model, tok, request, hparams, layer, context_templates) #(1, 11008)
                    rephrase_k.append(layer_ks.detach().cpu())
                    """
                    layer_vs = compute_ks(model, tok, request, hparams, layer, context_templates, track='out') # (1, 4096)
                    rephrase_vv.append(layer_vs.detach().cpu())
                    rephrase_original_v = compute_z(model, tok, request[0], hparams, layer, context_templates, eval=True)
                    rephrase_v.append(rephrase_original_v.detach().cpu())
                    """
                rephrase_k = torch.stack(rephrase_k) #[layer, 1, 11008]
                rephrase_ks.append(rephrase_k)
                """
                rephrase_v = torch.stack(rephrase_v) #[layer, 1, 11008]
                rephrase_vs.append(rephrase_v)
                rephrase_vv = torch.stack(rephrase_vv) #[layer, 1, 11008]
                rephrase_vvs.append(rephrase_vv)
                """

            rephrase_ks = torch.stack(rephrase_ks) #[sub_num, layer, 1, 11008]
            # rephrase_vs = torch.stack(rephrase_vs)
            # rephrase_vvs = torch.stack(rephrase_vvs)
        else:
            print('Skip rephrase dump')
            rephrase_ks, rephrase_vs, rephrase_vvs = None, None, None

    # shuffle
    tokens = tok.tokenize(record["subject"])
    shuffled_ks = []
    shuffled_vs = []
    shuffled_vvs = []

    if len(tokens) > 1 and 'shuffle' in collect_keys:
        # shuffled_sub_lprobs = []
        shuffled_sub_accs = []
        shuffled_sub_reversion = []
        if "shuffled_subject" in record.keys():
            test_subject = record["shuffled_subject"]
        else:
            # all_permutations = permutations(tokens)
            # shuffled_subject = ["".join(perm) for perm in all_permutations]
            import math
            sample_num = min(11, math.factorial(len(tokens)))
            perms = random_permutation(tokens, sample_num)
            test_subject = ["".join(perm) for perm in perms]
            try:
                test_subject.remove("".join(tokens))
            except:
                test_subject = test_subject[:10]
            assert len(test_subject) == sample_num-1

            # test_subject = random.sample(shuffled_subject, sample_num)
            ret["shuffled_subject"] = test_subject
        
        # breakpoint()
        for shuffle_sub in test_subject:
            shuffle_sub_prompt = record['prompt'].replace(record["subject"], shuffle_sub)
            shuffled_sub_accs.append(test_prediction_acc(model, tok, hparams, shuffle_sub_prompt, target_new if post else target_old, device))
            shuffled_sub_reversion.append(test_prediction_acc(model, tok, hparams, shuffle_sub_prompt,target_old, device))
            if post:
                ret["edited_shuffled_sub_target_new_acc"] = shuffled_sub_accs
                ret["edited_shuffled_sub_target_new_reversion"] = shuffled_sub_reversion
            else:
                ret["original_shuffled_sub_target_old_acc"] = shuffled_sub_accs
        
        # collect
        if dump:
            for shuffle_sub in test_subject:
                shuffled_k = []
                shuffled_v = []
                shuffled_vv = []
                request = [{"prompt": record["prompt"].replace(record["subject"], "{}"), "subject": shuffle_sub, "target_new": " " + record["target_new"]}]
                for i, layer in enumerate(hparams.layers):
                    layer_ks = compute_ks(model, tok, request, hparams, layer, context_templates) #(1, 11008)
                    shuffled_k.append(layer_ks.detach().cpu())
                    """
                    layer_vs = compute_ks(model, tok, request, hparams, layer, context_templates, track='out') #(1, 4096)
                    shuffled_vv.append(layer_vs.detach().cpu())
                    shuffled_original_v = compute_z(model, tok, request[0], hparams, layer, context_templates, eval=True)
                    shuffled_v.append(shuffled_original_v.detach().cpu())
                    """
                shuffled_k = torch.stack(shuffled_k)
                shuffled_ks.append(shuffled_k)
                """
                shuffled_v = torch.stack(shuffled_v)
                shuffled_vs.append(shuffled_v)
                shuffled_vv = torch.stack(shuffled_vv)
                shuffled_vvs.append(shuffled_vv)
                """
            shuffled_ks = torch.stack(shuffled_ks)
            # shuffled_vs = torch.stack(shuffled_vs)
            # shuffled_vvs = torch.stack(shuffled_vvs)
        else:
            print('Skip shuffle dump')
            shuffled_ks, shuffled_vs, shuffled_vvs = None, None, None

    ret["subject_length"] = len(tokens)
    # possessive
    if 'possessive' in collect_keys:
        possessive_prompt = record['prompt'].replace(record["subject"], record["subject"] + "'s")
        # possessive_sub_target_lprobs = run_lprobs(model, tok, record["target_new"], "cuda:0", prefix=possessive_prompt).item()
        possessive_sub_target_acc = test_prediction_acc(model, tok, hparams, possessive_prompt, target_new if post else target_old, device)
        if post:
            ret["edited_possessive_sub_target_new_acc"] = possessive_sub_target_acc
        else:
            ret["original_possessive_sub_target_old_acc"] = possessive_sub_target_acc

   
    ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=record["prompt"] if isinstance(record["prompt"],list) else [record["prompt"],], max_out_len=100)
    
    original_ks = []
    original_vs = []
    original_vvs = [] # value retrieved by key
    original_delta_vs = []
    
    if 'original' in collect_keys and dump:
        request = [{"prompt": record["prompt"].replace(record["subject"], "{}"), "subject": record["subject"], "target_new": " " + record["target_new"]}]
        for i, layer in enumerate(hparams.layers):
            layer_ks = compute_ks(model, tok, request, hparams, layer, context_templates) #(1, 11008)
            original_ks.append(layer_ks.detach().cpu())
            """
            # retrieve vs
            layer_vs = compute_ks(model, tok, request, hparams, layer, context_templates, track='out') #(1, 4096)
            original_vvs.append(layer_vs.detach().cpu())
            original_v, original_delta_v = compute_z(model, tok, request[0], hparams, layer, context_templates, eval=True, seperate=True)
            original_vs.append(original_v.detach().cpu())
            original_delta_vs.append(original_delta_v.detach().cpu())
            """
        original_ks = torch.stack(original_ks)
        # original_vs = torch.stack(original_vs)
        # original_delta_vs = torch.stack(original_delta_vs)
        # original_vvs = torch.stack(original_vvs)
    else:
        print('Skip origin dump')
        original_ks, original_vs, original_delta_vs, original_vvs = None, None, None, None
  
    # wiki_data = None
    if wiki_data is not None and 'long' in collect_keys:
        # long document
        if "long_context" in record.keys():
            assert post is True
            random_text = record["long_context"]
        else:
            def select_random():
                random_sample_index = random.randint(0, len(wiki_data["train"]) - 1)
                extracted_tokens = []
                while len(extracted_tokens) < long_ctx_len:
                    random_sample = wiki_data["train"][random_sample_index]["text"]
                    tokens = random_sample.split()
                    if len(tokens) + len(extracted_tokens) > long_ctx_len:
                        extracted_tokens+= tokens[: long_ctx_len-len(extracted_tokens)]
                    else:
                        extracted_tokens += tokens
                    random_sample_index += 1
                random_text = " ".join(extracted_tokens)
                random_text = random_text.replace('{', '').replace('}', '') # filter {} that in random text.
                return random_text
            random_text = select_random()
            ret["long_context"] = random_text
            ret['long_context_held'] = [select_random() for _ in range(5)]
        
        ret["edited_long_context_held_target_new_acc"] = []
        # eval
        if post:
            ret["edited_long_context_target_new_acc"] = test_prediction_acc(model, tok, hparams, random_text + " " + record["prompt"], target_new, device)
            ret["edited_long_context_target_new_reversion"] = test_prediction_acc(model, tok, hparams, random_text + " " + record["prompt"], target_old, device)
            if 'long_context_held' in record:
                ret["edited_long_context_held_target_new_acc"] = [test_prediction_acc(model, tok, hparams, r + " " + record["prompt"], target_new, device) for r in record['long_context_held']]
                ret["edited_long_context_held_target_new_reversion"] = [test_prediction_acc(model, tok, hparams, r + " " + record["prompt"], target_old, device) for r in record['long_context_held']]
            #  run_lprobs(model, tok, record["target_new"], "cuda:0", random_text + " " + record["prompt"]).item()
        else:
            ret["original_long_context_target_old_acc"] = test_prediction_acc(model, tok, hparams, random_text + " " + record["prompt"], target_old, device)
            if 'long_context_held' in ret:
                ret["edited_long_context_held_target_new_acc"] = [test_prediction_acc(model, tok, hparams, r + " " + record["prompt"], target_old, device) for r in ret['long_context_held']]

            # run_lprobs(model, tok, record["target_new"], "cuda:0", random_text + " " + record["prompt"]).item()

        if dump:
            long_ks = []
            long_vs = []
            long_vvs = []
        
            for i, layer in enumerate(hparams.layers):
                layer_ks = compute_ks(model, tok, request, hparams, layer, [[random_text.strip() + " {}"]]) #(1, 11008)
                long_ks.append(layer_ks.detach().cpu())
                """
                long_v = compute_z(model, tok, request[0], hparams, layer, [[random_text.strip() + " {}"]], eval=True)
                long_vs.append(long_v.detach().cpu())
                # vss
                layer_vs = compute_ks(model, tok, request, hparams, layer, [[random_text.strip() + " {}"]], track='out') #(1, 4096)
                long_vss.append(layer_vs.detach().cpu())
                """

            long_ks = torch.stack(long_ks)
            # long_vs = torch.stack(long_vs)
            # long_vvs = torch.stack(long_vss)
        else:
            print('Skip long dump') 
            long_ks, long_vs, long_vvs = None, None, None
    else:
        long_ks, long_vs, long_vvs = None, None, None

    # if post: hparams.rewrite_module_tmp = original_hparams
    collected_ks = {
        "original": original_ks,
        "rephrase": rephrase_ks,
        "shuffled": shuffled_ks,
        "long": long_ks
    }
    collected_vs = {
        "original": original_vs,
        "original_delta": original_delta_vs,
        "rephrase": rephrase_vs,
        "shuffled": shuffled_vs,
        "long": long_vs
    }
    collected_vvs = {
        "original": original_vvs,
        "rephrase": rephrase_vvs,
        "shuffled": shuffled_vvs,
        "long": long_vvs
    }
    collected_outputs = {
        'ks': collected_ks,
        'vs': collected_vs,
        'vss': collected_vvs,
    }

    return ret, collected_outputs

def simple_make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def para_attack(prompt, target_old, target_new, tok, model):
    paraps = rephrases[(prompt+" "+target_old).strip()]
    paraps = [prompt] + paraps
    if len(paraps)>5:
        num_attack_parap = 5
    else: 
        num_attack_parap = len(paraps)
    # print(num_attack_parap)
    # print("attack parap")
            
    attack_paraps = paraps[-num_attack_parap:]
    attack_paraps = [x.replace(target_old,"") for x in attack_paraps]
           
    batch = simple_make_inputs(tok, attack_paraps)
    pad_token_id = tok.pad_token_id
    pad_token_id = pad_token_id if pad_token_id else 0
             
    outputs = model.generate(**batch, do_sample=True, max_new_tokens=1,
                pad_token_id=pad_token_id, num_return_sequences=4)
            
    outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
    preds = [tok.decode(output) for output in outputs]
    ori_preds = preds[:]
              
    preds = [pred.replace(query_input, "").replace("!", "").strip() for pred, query_input in zip(preds, [element for element in attack_paraps for i in range(4)])]
              
    preds_attack_succ = [1.0 if target_new in x else 0.0 for x in preds]
    preds_attack_rev = [1.0 if target_old in x else 0.0 for x in preds ]
    # if len(preds_attack_succ) > 0: 
    #     ori_preds_succ = [x.replace('<unk>', '') for x in ori_preds if target_new in x]
    #     print("=================")
    #     print(f"Ori Fact: {prompt} {target_new}")
    #     print(f"Attack success paras:")
    #     for item in ori_preds_succ:
    #         print(item)
    #     print()
    # return len(preds_attack_succ)/len(preds), len(preds_attack_rev)/len(preds)
    return preds_attack_succ, preds_attack_rev

def eval_para(model, tok, para_pairs, device, prefix=None):
    device = torch.device(f'cuda:{device}')
    assert len(para_pairs) == 2
    more, less = para_pairs
    more_lprobs, less_lprobs = [], []
    for sent_more in more:
        if sent_more.strip():
            more_lprobs.append(run_lprobs(model, tok, sent_more.strip(), device, prefix=prefix).item())
    for sent_less in less:
        if sent_less.strip():
            less_lprobs.append(run_lprobs(model, tok, sent_less.strip(), device, prefix=prefix).item())
    return [more_lprobs, less_lprobs]

def eval_para_fact(model, tok, target, device, prefix=None):
    device = torch.device(f'cuda:{device}')
    para_lprobs, para_target_lprobs = [], []
    if prefix:
        for p in prefix:
            para_lprobs.append(run_lprobs(model, tok, (p + " " + target).strip(), device).item())
            para_target_lprobs.append(run_lprobs(model, tok, target, device, prefix=p.strip()).item())
    else:
        for t in target:
            para_lprobs.append(run_lprobs(model, tok, t.strip(), device).item())
    
    return para_lprobs, para_target_lprobs

def eval_decrease(model, tok, prompt, target, original_target_lprobs, device):
    device = torch.device(f'cuda:{device}')
    lprobs = run_lprobs(model, tok, target, device, prefix=prompt).item()
    difference = int((original_target_lprobs - lprobs) < - 0.5 * original_target_lprobs)

    if lprobs < -6.907755278982137: # prob=0.1%
        return lprobs, difference, 1
    else:
        return lprobs, difference, 1

def eval_reverse(model, tok, sentence_pairs, original_lprobs, device):
# def eval_reverse(model, tok, sentence_pairs, original_lprobs):

#     with open("../data_construction/outputs/crows_pairs/eval_reverse.json", "r")as f:
#         data = json.loads(f)
    device = torch.device(f'cuda:{device}')
    new_lprobs1 = run_lprobs(model, tok, sentence_pairs[0], device).item()
    new_lprobs2 = run_lprobs(model, tok, sentence_pairs[1], device).item()
    if (new_lprobs1-new_lprobs2)*(original_lprobs[0]-original_lprobs[1]) <0:
        return 1, [new_lprobs1, new_lprobs2]
    else:
        return 0, [new_lprobs1, new_lprobs2]


    
def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
               compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if  test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompt, target_new, device)
        ret = {
            f"{key}_ppl": ppl
        }
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
        ret = {
            f"{key}_acc": acc
        }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:
        loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

@torch.no_grad()
def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    prompt = record["prompt"]
    # rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target_new, new_fact)
    ret = {
        f"acc": edit_acc
    }

    # compute reversion
    target_old = record['target_true']
    assert target_old != target_new
    acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, prompt)
    ret['rev'] = acc

    # compute generalize
    if 'rephrase_prompt' in record.keys() and record["rephrase_prompt"] != None and not pre_edit:
        ret['generalization'] = []
        for p in record["rephrase_prompt"]:
            if p != "":
                gen_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, f'New Fact: {prompt} {target_new}\nPrompt: {p}')
                ret['generalization'].append(gen_acc)


    ret['locality'] = {}
  
    # if rephrase is not None:
    #     rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
    #     ret['rephrase_acc'] = rephrase_acc

    breakpoint()
    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if pre_edit:
               
                neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality'][locality_key]['ground_truth'], record['locality'][locality_key]['prompt'], neighborhood=True)
            else:
                
                neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality'][locality_key]['ground_truth'],
                f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
            if type(neighbor) is not list:
                neighbor = [neighbor,]
            ret['locality'].update({f"{locality_key}_output": neighbor})
    # compute rephrase subjects
    rephrase_accs = []
    rephrase_reverision = []
    if "rephrase_subjects" in record.keys() and any(record['rephrase_subjects']):
        for re_sub in record['rephrase_subjects']:

            re_sub_prompt = record['prompt'].replace(record["subject"], re_sub)

            if not pre_edit:
                rephrase_subject_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, f'New Fact: {prompt} {target_new}\nPrompt: {re_sub_prompt}')

                rephrase_subject_reversion = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, f'New Fact: {prompt} {target_new}\nPrompt: {re_sub_prompt}')
            else:
                rephrase_subject_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, re_sub_prompt)
            
                rephrase_subject_reversion = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, re_sub_prompt)

            rephrase_accs.append(rephrase_subject_acc)
            rephrase_reverision.append(rephrase_subject_reversion)
        if not pre_edit:
            ret["edited_rephrase_sub_target_new_acc"] = rephrase_accs
            ret["edited_rephrase_sub_target_new_reversion"] = rephrase_subject_reversion
        else:
            ret["original_rephrase_sub_target_old_acc"] = rephrase_accs

        ret['rephrase_subject'] = record['rephrase_subjects']

    tokens = tok.tokenize(record["subject"])
    if len(tokens) > 1:
        # shuffled_sub_lprobs = []
        shuffled_sub_accs = []
        shuffled_sub_reversion = []
        if "shuffled_subject" in record.keys():
            test_subject = record["shuffled_subject"]
        else:
            # all_permutations = permutations(tokens)
            # shuffled_subject = ["".join(perm) for perm in all_permutations]
            import math
            sample_num = min(11, math.factorial(len(tokens)))
            perms = random_permutation(tokens, sample_num)
            test_subject = ["".join(perm) for perm in perms]
            try:
                test_subject.remove("".join(tokens))
            except:
                test_subject = test_subject[:10]
            assert len(test_subject) == sample_num-1

            # test_subject = random.sample(shuffled_subject, sample_num)
            ret["shuffled_subject"] = test_subject
    # print(tokens)
    # print(test_subject)
        for shuffle_sub in test_subject:
            shuffle_sub_prompt = record['prompt'].replace(record["subject"], shuffle_sub)

            if not pre_edit:
                shuffled_sub_accs.append(icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, f'New Fact: {prompt} {target_new}\nPrompt: {shuffle_sub_prompt}'))
                shuffled_sub_reversion.append(icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, f'New Fact: {prompt} {target_new}\nPrompt: {shuffle_sub_prompt}'))
            else:
                shuffled_sub_accs.append(icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, shuffle_sub_prompt))
                shuffled_sub_reversion.append(icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, shuffle_sub_prompt))

            if not pre_edit:
                ret["edited_shuffled_sub_target_new_acc"] = shuffled_sub_accs
                ret["edited_shuffled_sub_target_new_reversion"] = shuffled_sub_reversion
            else:
                ret["original_shuffled_sub_target_old_acc"] = shuffled_sub_accs

    if not pre_edit:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_fact if isinstance(new_fact,list) else [new_fact,], max_out_len=100)

    if wiki_data is not None:
        # long document
        if "long_context" in record.keys():
            assert pre_edit is False
            random_text = record["long_context"]
        else:
            def select_random():
                random_sample_index = random.randint(0, len(wiki_data["train"]) - 1)
                extracted_tokens = []
                while len(extracted_tokens) < 300:
                    random_sample = wiki_data["train"][random_sample_index]["text"]
                    tokens = random_sample.split()
                    if len(tokens) + len(extracted_tokens) > 300:
                        extracted_tokens+= tokens[: 300-len(extracted_tokens)]
                    else:
                        extracted_tokens += tokens
                    random_sample_index += 1
                random_text = " ".join(extracted_tokens)
                random_text = random_text.replace('{', '').replace('}', '') # filter {} that in random text.
                return random_text
            random_text = select_random()
            ret["long_context"] = random_text
            ret['long_context_held'] = [select_random() for _ in range(5)]
        
        ret["edited_long_context_held_target_new_acc"] = []
        # eval
        if not pre_edit:
            ret["edited_long_context_target_new_acc"] = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, f'New Fact: {prompt} {target_new}\nPrompt: {random_text + " " + record["prompt"]}')
            ret["edited_long_context_target_new_reversion"] = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, f'New Fact: {prompt} {target_new}\nPrompt: {random_text + " " + record["prompt"]}')
            
            if 'long_context_held' in record:
                ret["edited_long_context_held_target_new_acc"] = [icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_new, f'New Fact: {prompt} {target_new}\nPrompt: {r + " " + record["prompt"]}') for r in record['long_context_held']]
                ret["edited_long_context_held_target_new_reversion"] = [icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, f'New Fact: {prompt} {target_new}\nPrompt: {r + " " + record["prompt"]}') for r in record['long_context_held']]
            #  run_lprobs(model, tok, record["target_new"], "cuda:0", random_text + " " + record["prompt"]).item()
        else:
            ret["original_long_context_target_old_acc"] = icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, random_text + " " + record["prompt"])

            if 'long_context_held' in ret:
                ret["edited_long_context_held_target_new_acc"] = [icl_lm_eval(model, model_name, hparams, tok, icl_examples, target_old, r + " " + record["prompt"]) for r in ret['long_context_held']]

    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return [torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()]
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]   
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return [torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()]        
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return [torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()]

def compute_icl_multimodal_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    # vis_tok,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)
    
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(hparams.device)
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]
    
    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target, prompt, image)
    else:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target, new_fact, image)
    ret = {
        f"rewrite_acc": edit_acc
    }
    if rephrase is not None:
        rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc
        
    if "image_rephrase" in record.keys():
        rephrase_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc
    
    if "locality_prompt" in record.keys():
        locality_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                loc_a, f'New Fact: {loc_q} {loc_a}\nPrompt: {loc_q}', None)
        ret['locality_acc'] = locality_acc
    
    if "multimodal_locality_image" in record.keys():
        locality_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               m_loc_a, f'New Fact: {m_loc_q} {m_loc_a}\nPrompt: {m_loc_q}', m_loc_image)
        ret['locality_image_acc'] = locality_image_acc
            
    return ret

def icl_multimodal_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    
    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    
    return compute_multimodal_edit_quality(model, samples)

def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt,)) for prompt in prompts]  
        target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len        
    } 
    return ret

def compute_multimodal_edit_quality(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy()
  
def compute_multimodal_edit_quality_demo(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy(), logits

def compute_multimodal_edit_results(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret
  
def compute_multimodal_edit_results_demo(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits


    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]

def compute_sent_metric(
    model,
    edited_model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    metric_kwargs: typing.Dict,
    device,
    test_generation=True
    ):
    
    if "llama" not in model_name:
        raise NotImplementedError("currently only support for llama")
        
    def get_edit_labels(ids, prompts=None):
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100
        return labels
        
    same_mask = torch.tensor([i == o for i, o in zip(metric_kwargs["inner_target"], metric_kwargs["all_target"])], device=device)
    edit_toks = {
        f"{k1}_{k2}": v2.to(device)
        for k1, v1 in {
            "inner": metric_kwargs["inner_all_qa"],
            "outer": metric_kwargs["outer_all_qa"],
        }.items()
        for k2, v2 in tok(
            v1,
            return_tensors="pt",
            padding=True,
            max_length=128,
            truncation=True,
        ).items()
    }
    for key in ["inner", "outer"]:
        value = edit_toks[f"{key}_input_ids"]
        mask = [([True] * value.shape[-1])] * value.shape[0]
        for i in range(value.shape[0]):
            sep_idx = list(value[i]).index(tok.convert_tokens_to_ids("</s>"))
            for j in range(sep_idx): #连带</s>一块mask掉
                mask[i][j] = False
        edit_toks[key + "_q_mask"] = torch.tensor(mask).to(device)

    with torch.no_grad():
        inner_base_logits = model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        inner_edit_logits = edited_model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        
        outer_base_logits = model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
        outer_edit_logits = edited_model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
    
    result = {
        "es": es_sent(inner_base_logits, inner_edit_logits, edit_toks["inner_q_mask"], get_edit_labels(edit_toks["inner_input_ids"]), same_mask).item(),
        "dd": kl_loc_loss(outer_base_logits, outer_edit_logits, edit_toks["outer_q_mask"]).item(),
    }
    if  test_generation:
        result['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=metric_kwargs["inner_q"] if isinstance(metric_kwargs["inner_q"],list) else [metric_kwargs["inner_q"],], max_out_len=100)
    return result

def compute_icl_bias_edit_quality(*args, **kwargs):
    raise # dummy function


def random_permutation(lst, num_samples):
    seen = set()
    results = []

    while len(results) < num_samples:
        perm = tuple(random.sample(lst, len(lst)))
        if perm not in seen:
            seen.add(perm)
            results.append(perm)

    return results