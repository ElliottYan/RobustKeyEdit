import json
from easyeditor import BaseEditor
# from easyeditor import CausalEditor
from easyeditor import FTHyperParams,\
ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
SERACTrainingHparams, SERACHparams, IKEHyperParams,LoRAHyperParams, SUEHyperParams, SUEFreeHyperParams
from easyeditor import CrowsPairsDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from easyeditor.evaluate import run_lprobs, compute_icl_bias_edit_quality, compute_bias_edit_quality
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

def step_eval_func_v2(
    model, 
    model_name, 
    tok, 
    eval_data, 
    hparams,
    icl_examples,
    device,
    desc
):  
    assert (icl_examples != []) == (model_name == 'IKE')
    all_metrics = {}
    for i, record in tqdm(enumerate(eval_data), desc=desc):
        if model_name == 'IKE':
            each_ret = compute_icl_bias_edit_quality(model, model_name, hparams, tok, icl_examples, record, device)
        else:
            each_ret = compute_bias_edit_quality(model, model_name, hparams, tok, record, device, test_generation=True)
        all_metrics[i] = {
            'case_id': i,
            'post': each_ret,
        }
    return all_metrics

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

    if "targets_new" in data[0].keys():
        target_new = [d["targets_new"] for d in data]
    elif "target_new" in data[0].keys():
        target_new =  [d["target_new"] for d in data]

    if "original_pairs" in data[0].keys():
        original_pairs = [d["original_pairs"] for d in data]
    else:
        original_pairs = []
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
    original_target_lprobs = [d['original_target_lprobs'] for d in data]
    
    # form into locality input to be compatible with EasyEdit computations
    locality_inputs = {}
    # for i in range(len(locality_prompts)):
    locality_inputs['zsre'] = {}
    locality_inputs['zsre']['prompt'] = locality_prompts
    locality_inputs['zsre']['ground_truth'] = locality_targets

    return prompts, target_new, original_lprobs, original_pairs, subjects, para_pairs, locality_inputs, original_target_lprobs, token_lists, original_token_lprobs_lists, rephrase_prompts, rephrase_targets

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

def MEND_Meta_Train_Llama(train_data_dir, train_hparams_file):
    training_hparams = MENDTrainingHparams.from_hparams(train_hparams_file)
    train_ds = CrowsPairsDataset(f'{train_data_dir}/train.json', config=training_hparams)
    eval_ds = CrowsPairsDataset(f'{train_data_dir}/val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

def MEND_Meta_Train_gpt2(train_data_dir, train_hparams_file):
    training_hparams = MENDTrainingHparams.from_hparams(train_hparams_file)
    train_ds = CrowsPairsDataset(f'{train_data_dir}/train.json', config=training_hparams)
    eval_ds = CrowsPairsDataset(f'{train_data_dir}/val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

def SERAC_train_llama(train_data_dir, train_hparams_file):
    training_hparams = SERACTrainingHparams.from_hparams(train_hparams_file)
    train_ds = CrowsPairsDataset(f'{train_data_dir}/train.json', config=training_hparams)
    eval_ds = CrowsPairsDataset(f'{train_data_dir}/val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

def SERAC_train_gpt2(train_data_dir, train_hparams_file):
    training_hparams = SERACTrainingHparams.from_hparams(train_hparams_file)
    train_ds = CrowsPairsDataset(f'{train_data_dir}/train.json', config=training_hparams)
    eval_ds = CrowsPairsDataset(f'{train_data_dir}/val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()
    

def main():
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
    parser.add_argument('--enable_causal_edit', action='store_true', help='Enable of causal editor.')
    parser.add_argument('--causal_edit_max_n', type=int, default=5, help='Max turn of edit for each sample.')
    parser.add_argument('--causal_edit_target_diff', type=float, default=3.35, help='Expected diff after of causal edit.') 
    # update hparams
    parser.add_argument('--update_hparams', type=str, default='{}', help='Hyperparameters dictionary in string format')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    args = parser.parse_args()

    if args.train_method == 'MEND':
        assert args.train_model is not None, "Have to set --train_model"
        if 'llama' in args.train_model:
            MEND_Meta_Train_Llama(args.train_data_dir, args.train_hparams_file)
        elif args.train_model == 'gpt2-xl':
            MEND_Meta_Train_gpt2(args.train_data_dir, args.train_hparams_file)
        else:
            raise
    elif args.train_method == 'SERAC':
        assert args.train_model is not None, "Have to set --train_model"
        if 'llama' in args.train_model:
            SERAC_train_llama(args.train_data_dir, args.train_hparams_file)
        elif args.train_model == 'gpt2-xl':
            SERAC_train_gpt2(args.train_data_dir, args.train_hparams_file)
        else:
            raise

    else:
        if args.editing_method in ['FT', 'FT_L']:
            editing_hparams = FTHyperParams
        elif args.editing_method == 'IKE':
            editing_hparams = IKEHyperParams
        elif args.editing_method == 'SERAC':
            editing_hparams = SERACHparams
        elif args.editing_method == 'MEMIT':
            editing_hparams = MEMITHyperParams
        elif args.editing_method == 'SUE':
            editing_hparams = SUEHyperParams
        elif args.editing_method == 'ROME':
            editing_hparams = ROMEHyperParams
        elif args.editing_method == 'MEND':
            editing_hparams = MENDHyperParams
        elif args.editing_method == 'LoRA':
            editing_hparams = LoRAHyperParams
        elif args.editing_method == 'SUE_FREE':
            editing_hparams = SUEFreeHyperParams
        else:
            raise NotImplementedError
        set_all_seed(args.seed)
        prompts, target_new, original_lprobs, original_pairs, subjects, para_pairs, locality_inputs,original_target_lprobs, token_lists, original_token_lprobs_lists, rephrase_prompts, rephrase_targets  = read_data(args.data_dir, k=args.data_points)
        
        hparams = editing_hparams.from_hparams(args.hparams_dir)
        # update hparams
        update_dict = json.loads(args.update_hparams)
        print("Update hparams with dict: ")
        print(update_dict)
        for key, value in update_dict.items():
            assert hasattr(hparams, key), f"The updated key {key} is not in the hparam!"
            # hparams[key] = value
            setattr(hparams, key, value)

        edited_bias = None
        extra_metrics = None
        locality_metrics = None
        overall_metrics = None
        for bias_type in BIAS_TYPE:
            if bias_type in args.data_dir:
                edited_bias = bias_type
                break

        if args.editing_method == 'IKE':
            train_dir = '../data_construction/outputs/v2/MEND_bias_upper_repha/train.json'
            train_ds =  CrowsPairsDataset(train_dir)
            if not args.sequential_editing:
                # if in single edit mode
                # use the original IKE algs with retrieval
                sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
                encode_ike_facts(sentence_model, train_ds, hparams)
                icl_examples = []
            else:
                with open(train_dir, "r") as f:
                    data = json.load(f)
                examples = random.sample(data, hparams.k)
                icl_examples = [f"New Fact: {request['prompt'] + ' '+ request['target_new']}\nPrompt: {request['prompt'] + ' '+ request['target_new']}\n\n" for request in examples]
                # NOTE: in sequential editing, we use random new fact to construct icl examples.
                # Also, we cannot add each new fact in icl example in advance.
        else:
            icl_examples = []
            train_ds = None
        
        # if not args.enable_causal_edit:
        if args.sequential_editing is True:
            from functools import partial
            editor = BaseEditor.from_hparams(hparams)
            eval_func = partial(step_eval_func_v2, tok=editor.tok, hparams=hparams, device=hparams.device, model_name=hparams.alg_name)
        
            if hparams.alg_name == "LoRA":
                _, edited_model, _, locality_metrics = editor.batch_edit(
                    prompts=prompts,
                    target_new=target_new,
                    keep_original_weight=False if args.sequential_editing else True,
                    original_lprobs = original_lprobs, 
                    original_pairs=original_pairs,
                    icl_examples = icl_examples,
                    subject=subjects,
                    locality_inputs=locality_inputs,
                    para_pairs=para_pairs,
                    original_target_lprobs=original_target_lprobs,
                    token_lists=token_lists,
                    original_token_lprobs_lists=original_token_lprobs_lists,
                    rephrase_prompts=rephrase_prompts,
                    rephrase_targets=rephrase_targets
                )
            else:
                if edited_bias is None and not args.only_some_step:
                    extra_eval_points=[] # [1, 4, 16, 64, 256,]
                elif args.only_some_step:
                    # extra_eval_points=[1, 4, 16]
                    extra_eval_points=[]
                else:
                    extra_eval_points = []
                metrics, edited_model, _, extra_metrics, locality_metrics, requests = editor.edit(
                    prompts=prompts,
                    target_new=target_new,
                    subject=subjects,
                    train_ds=train_ds,
                    original_lprobs = original_lprobs, 
                    original_pairs=original_pairs,
                    icl_examples = icl_examples,
                    keep_original_weight=False if args.sequential_editing else True,
                    extra_eval_points=extra_eval_points,
                    eval_func=eval_func,
                    save_when_eval=True,
                    model_save_dir=args.model_save_dir, 
                    locality_inputs=locality_inputs,
                    para_pairs=para_pairs,
                    seed=args.seed,
                    only_some_step=args.only_some_step,
                    original_target_lprobs=original_target_lprobs,
                    token_lists=token_lists,
                    original_token_lprobs_lists=original_token_lprobs_lists,
                    test_generation=True,
                    rephrase_prompts=rephrase_prompts,
                    rephrase_targets=rephrase_targets,
                )
        else: # single
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _, _,_,_ = editor.edit(
                prompts=prompts,
                target_new=target_new,
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
                test_generation=True,
                rephrase_prompts=rephrase_prompts,
                rephrase_targets=rephrase_targets,
                debug=args.debug
            )

        # else: # enable_causal_edit == True
        #     assert args.sequential_editing is False # not implemented yet.
        #     editor = CausalEditor.from_hparams(hparams)
        #     metrics, edited_model, _, _,_ = editor.edit(
        #         prompts=prompts,
        #         target_new=target_new,
        #         subject=subjects,
        #         train_ds=train_ds,
        #         original_lprobs = original_lprobs, 
        #         original_pairs=original_pairs,
        #         icl_examples = icl_examples,
        #         keep_original_weight=False if args.sequential_editing else True,
        #         locality_inputs=locality_inputs,
        #         para_pairs=para_pairs,
        #         original_target_lprobs=original_target_lprobs,
        #         token_lists=token_lists,
        #         original_token_lprobs_lists=original_token_lprobs_lists,
        #         test_generation=True
        #     )
            
        # final eval and save
        if not os.path.exists(args.metrics_save_dir):
            os.makedirs(args.metrics_save_dir, exist_ok=True)
        if args.sequential_editing:
            if edited_bias is not None:
                for b in BIAS_TYPE:
                    data_file = os.path.join(args.data_dir.rsplit('/', 1)[0], f"{b}_edit.json")
                    with open(data_file, "r") as f:
                        edited_bias_data =  json.load(f)
                    for d in edited_bias_data:
                        d["locality"]={
                            "zsre":{
                                "prompt": d["locality_prompt"],
                                "ground_truth": d["locality_ground_truth"]
                            }
                        }
                    bias_metrics =  eval_func(model=edited_model, eval_data=edited_bias_data,icl_examples=icl_examples, desc=f"Eval edited model after {edited_bias} {hparams.alg_name} using {len(edited_bias_data)} {b} samples")

                    json.dump(bias_metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_edit_{edited_bias}_eval_{b}.json'), 'w'), indent=4)

            elif not args.only_some_step:
                # with open(args.data_dir, "r")as f:
                #     all_eval_data = json.load(f)
                # if not args.data_points:
                #     k = len(all_eval_data)
                # else:
                #     k = args.data_points
                # for d in all_eval_data:
                #         d["locality"]={
                #             "zsre":{
                #                 "prompt": d["locality_prompt"],
                #                 "ground_truth": d["locality_ground_truth"]
                #             }
                #         }
                #         if "token_list" in d.keys():
                #             d["token_lists"] = d["token_list"]
                #         if "original_token_lprobs_list" in d.keys():
                #             d["original_token_lprobs_lists"] = d["original_token_lprobs_list"]
        
                overall_metrics = eval_func(model=edited_model, eval_data=requests,icl_examples=icl_examples, desc=f"Eval edited model after {hparams.alg_name} using {len(requests)} samples")
                
                save_dir = args.model_save_dir
                if save_dir.endswith('/'):
                    save_dir = save_dir[:-1]
                save_dir = f"{save_dir}-seed-{args.seed}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                if args.editing_method == 'IKE':
                    import pickle
                    with open(f"{save_dir}/icl_exmaples.pt", 'wb') as fp:
                        pickle.dump(icl_examples, fp)
                elif isinstance(edited_model, SERAC):
                    obj = edited_model.state_dict()
                    torch.save(obj, f"{save_dir}/serac.pt")
                    json.dump(edited_model.cache_inputs, open(os.path.join(save_dir, 'cache_inputs.json'), 'w'), indent=4)
                    json.dump(edited_model.cache_labels, open(os.path.join(save_dir, 'cache_labels.json'), 'w'), indent=4)
                else:
                    edited_model.save_pretrained(save_dir)
            else:
                # pass eval
                pass
        if not args.sequential_editing:
            overall_metrics = count_rate(metrics)

        if locality_metrics is not None:
            json.dump(locality_metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_pre_locality_seed{args.seed}.json'), 'w'), indent=4)
        if overall_metrics is not None:
            json.dump(overall_metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results_seed{args.seed}.json'), 'w'), indent=4)
        if extra_metrics is not None:
            with open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_extra_results_seed{args.seed}.json'), 'w') as fm:
                json.dump(extra_metrics, fm, indent=4)

    return 

if __name__ == '__main__':
    
    main()
