import json
from easyeditor import BaseErasor
# from easyeditor import CausalEditor
from easyeditor import FTHyperParams,\
ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
SERACTrainingHparams, SERACHparams, IKEHyperParams,LoRAHyperParams,R_ROMEHyperParams, PMETHyperParams, DINMHyperParams, EMMETHyperParams
from easyeditor import CrowsPairsDataset, CounterFactDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
# from easyeditor.evaluate import run_lprobs, compute_icl_bias_edit_quality, compute_bias_edit_quality
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import argparse
from easyeditor.trainer import *
import torch
import random

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']= str(seed)

def read_data(dir, k=100):
    with open(dir,"r") as f:
        data = json.load(f)
    print(len(data))
    # shuffle data
    if not k:
        k = len(data)
    data = random.sample(data, k=k)
    if "prompts" in data[0].keys():
        prompts = [d["prompts"] for d in data]
    elif "prompt" in data[0].keys():
        prompts = [d["prompt"] for d in data]

    if "target_new" in data[0].keys():
        target_new =  [d["target_new"] for d in data]
    else: raise
    
    if "target_true" in data[0].keys():
        target_true =  [d["target_true"] for d in data]
    else: raise

    if "original_pairs" in data[0].keys():
        original_pairs = [d["original_pairs"] for d in data]
    else:
        original_pairs = []
    if "subject_rephrase" in data[0].keys():
        subject_rephrase = [d["subject_rephrase"] for d in data]
    else:
        subject_rephrase = []

    if "long_context" in data[0].keys():
        long_context = [d["long_context"] for d in data]
    else:
        long_context = []
    
    original_lprobs = []
    if "original_lprobs" in data[0].keys():
        original_lprobs = [d["original_lprobs"] for d in data]
    
    subjects = [d["subject"] for d in data]
    # read more info
    if "para_pairs" in data[0].keys():
        para_pairs = [d["para_pairs"] for d in data]
        
    else:
        para_pairs = []

    if "rephrase_prompt" in data[0].keys():
        rephrase_prompts =  [d["rephrase_prompt"] for d in data]
    else:
        rephrase_prompts = []

    if "rephrase_target" in data[0].keys():
        rephrase_targets = [d["rephrase_target"] for d in data]
    else:
        rephrase_targets = []

    if "token_list" in data[0].keys():
        token_lists = [d["token_list"] for d in data]
        
    else:
        token_lists = []
    
    if "original_token_lprobs_list" in data[0].keys():
        original_token_lprobs_lists = [d["original_token_lprobs_list"] for d in data]
    else:
        original_token_lprobs_lists = []

    
    locality_prompts = [d['locality_prompt'] for d in data]
    locality_targets = [d['locality_ground_truth'] for d in data]
    
    if 'original_target_lprobs' in data[0].keys():
        original_target_lprobs = [d['original_target_lprobs'] for d in data]
    else:
        original_target_lprobs = []
    
    # form into locality input to be compatible with EasyEdit computations
    locality_inputs = {}
    # for i in range(len(locality_prompts)):
    locality_inputs['zsre'] = {}
    locality_inputs['zsre']['prompt'] = locality_prompts
    locality_inputs['zsre']['ground_truth'] = locality_targets

    return prompts, target_new, target_true, original_lprobs, original_pairs, subjects, para_pairs, locality_inputs, original_target_lprobs, token_lists, original_token_lprobs_lists, rephrase_prompts, rephrase_targets, subject_rephrase, long_context

def count_rate(all_metrics):
    reverse_success_count = 0
    for item in all_metrics:
        if "reverse_success_rate" in item["post"].keys():

            if item["post"]["reverse_success_rate"] == 1:
                reverse_success_count += 1
        else:
            return all_metrics
    all_metrics.append({"total_reverse_success_rate" : reverse_success_count/len(all_metrics)})

    return all_metrics

def main():
    print("NOTE: We don't support sequential edit in edit_single.py !!!!")
    BIAS_TYPE = ["race", "gender","religion", "profession"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_method', type=str, default=None)
    parser.add_argument('--train_model', type=str, default=None)
    parser.add_argument('--editing_method', type=str)
    parser.add_argument('--hparams_dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default="../data_construction/outputs/steroset/MEND/edit.json")
    parser.add_argument('--train_data_dir', type=str, default='../data_construction/outputs/steroset_more')
    parser.add_argument('--train_hparams_file', type=str, default='./our_hparams/TRAINING/SERAC/gpt2-xl.yaml')
    parser.add_argument('--data_points', type=int, default=None)
    parser.add_argument('--metrics_save_dir', default='./results/sequential/metrics/steroset', type=str)
    parser.add_argument('--model_save_dir', default='./results/sequential/models', type=str)
    parser.add_argument('--sequential_editing', action='store_true', help='Description of your argument')
    parser.add_argument('--only_some_step', action='store_true', help='Description of your argument')
    # update hparams
    parser.add_argument('--update_hparams', type=str, default='{}', help='Hyperparameters dictionary in string format')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--dump', action='store_true', help='Dump collect results.')
    parser.add_argument('--collect', type=str, default='', help='collect string to indicate what to collect.')
    # analyze hparams

    args = parser.parse_args()

    assert args.train_method is None, "edit_single.py do not support trainer!"

    if args.editing_method in ['FT', 'FT_L']:
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    elif args.editing_method == 'PMET':
        editing_hparams = PMETHyperParams
    elif args.editing_method == 'EMMET':
        editing_hparams = EMMETHyperParams
    elif args.editing_method == 'DINM':
        editing_hparams = DINMHyperParams
    # elif args.editing_method == 'SUE_FREE':
        # editing_hparams = SUEFreeHyperParams
    else:
        raise NotImplementedError
    set_all_seed(args.seed)
    prompts, target_new, target_true, original_lprobs, original_pairs, subjects, para_pairs, locality_inputs,original_target_lprobs, token_lists, original_token_lprobs_lists, rephrase_prompts, rephrase_targets, rephrase_subjects, long_context  = read_data(args.data_dir, k=args.data_points)
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    
    # update hparams
    update_dict = json.loads(args.update_hparams)
    print("Update hparams with dict: ")
    print(update_dict)
    for key, value in update_dict.items():
        assert hasattr(hparams, key), f"The updated key {key} is not in the hparam!"
        if not isinstance(value, type(getattr(hparams, key))):
            value = type(getattr(hparams, key))(value)
        setattr(hparams, key, value)
    
    print(hparams)
    
    assert args.sequential_editing is False
    
    extra_metrics = None
    locality_metrics = None
    overall_metrics = None

    if args.editing_method == 'IKE':
        train_dir = '../data_construction/dataset_aug/counterfact_para_prompt_para_subject.json'
        train_ds =  CounterFactDataset(train_dir,size=100)
        # if in single edit mode
        # use the original IKE algs with retrieval
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
        icl_examples = []
    else:
        icl_examples = []
        train_ds = None

    # editor = BaseEditor.from_hparams(hparams)
    editor = BaseErasor.from_hparams(hparams)
    if args.editing_method == 'IKE':
         metrics, edited_model, _, _,_,_, collect = editor.ike_edit(
            prompts=prompts,
            target_new=target_new,
            target_true=target_true, 
            subject=subjects,
            train_ds=train_ds,
            original_lprobs = original_lprobs, 
            original_pairs=original_pairs,
            icl_examples = icl_examples,
            keep_original_weight=False if args.sequential_editing else True,
            locality_inputs=locality_inputs,
            para_pairs=para_pairs,
            original_target_lprobs=original_target_lprobs,
            token_lists=token_lists,
            original_token_lprobs_lists=original_token_lprobs_lists,
            test_generation=False, # set to false for fast develop
            rephrase_prompts=rephrase_prompts,
            rephrase_targets=rephrase_targets,
            rephrase_subjects=rephrase_subjects,
            long_context = long_context,
            debug=args.debug,
            collect=args.collect,
            dump=args.dump,
        )
    else:
        metrics, edited_model, _, _,_,_, collect = editor.edit(
            prompts=prompts,
            target_new=target_new,
            target_true=target_true, 
            subject=subjects,
            train_ds=train_ds,
            original_lprobs = original_lprobs, 
            original_pairs=original_pairs,
            icl_examples = icl_examples,
            keep_original_weight=False if args.sequential_editing else True,
            locality_inputs=locality_inputs,
            para_pairs=para_pairs,
            original_target_lprobs=original_target_lprobs,
            token_lists=token_lists,
            original_token_lprobs_lists=original_token_lprobs_lists,
            test_generation=False, # set to false for fast develop
            rephrase_prompts=rephrase_prompts,
            rephrase_targets=rephrase_targets,
            rephrase_subjects=rephrase_subjects,
            long_context = long_context,
            debug=args.debug,
            collect=args.collect,
            dump=args.dump,
        )
        
    # final eval and save
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir, exist_ok=True)

    overall_metrics = count_rate(metrics)

    if locality_metrics is not None:
        json.dump(locality_metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_pre_locality_seed{args.seed}.json'), 'w'), indent=4)
    if overall_metrics is not None:
        json.dump(overall_metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results_seed{args.seed}.json'), 'w'), indent=4)
    if extra_metrics is not None:
        with open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_extra_results_seed{args.seed}.json'), 'w') as fm:
            json.dump(extra_metrics, fm, indent=4)

    # torch.save(pre_kss, os.path.join(args.metrics_save_dir, f'{args.editing_method}_pre_kss_seed{args.seed}.pt'))
    # torch.save(post_kss, os.path.join(args.metrics_save_dir, f'{args.editing_method}_post_kss_seed{args.seed}.pt'))
    # for k1 in ['pre', 'post']:
        # for k2 in collect[k1]:
    if args.dump:
        torch.save(collect, os.path.join(args.metrics_save_dir, f'{args.editing_method}_collect_seed{args.seed}.pt'))
    return 

if __name__ == '__main__':
    
    main()
