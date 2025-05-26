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

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
# from accelerate import Accelerator
from ..util.globals import *
from .singleton_editor import SingletonEditor
# from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_bias_edit_quality, compute_icl_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class BaseErasor:
    """Base erasor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map='auto' if hparams.model_parallel else None)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto' if hparams.model_parallel else None)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                # self.model = LlamaForCausalLM.from_pretrained(self.model_name, device_map='auto' if hparams.model_parallel else None)
                # self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto' if hparams.model_parallel else None)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, device_map='auto' if hparams.model_parallel else None)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch.float32, device_map='auto' if hparams.model_parallel else None)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, device_map='auto')
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, device_map='auto' if hparams.model_parallel else None)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch.float32 if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map='auto' if hparams.model_parallel else None)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=True if hparams.alg_name == 'ROME' else False ,trust_remote_code=True, device_map='auto' if hparams.model_parallel else None)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            else:
                raise NotImplementedError

            # RIGHT padding????? why is this
            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name
        # device_map = {
        #     0: [_ for _ in range(0, 16)],
        #     1: [_ for _ in range(16, 32)],
        #     2: [_ for _ in range(32, 48)]
        # }
        # self.model.parallelize(device_map=device_map)
        self.tok.truncation_side="left"
        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             target_true: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             collect='',
             dump=True,
             cached_pre_eval_path='',
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
            target_true = [target_true,]

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

        requests = self._prepare_requests(prompts, target_new, target_true, ground_truth, rephrase_prompts,
                                          locality_inputs, portability_inputs, 
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        assert self.alg_name != 'FT-Api', "We do not support FT-Api for erasor."

        all_metrics = []
        locality_metrics = []
        collects = {
            'pre': [],
            'post': []
        }
        if cached_pre_eval_path != "":
            assert os.path.exists(cached_pre_eval_path) and cached_pre_eval_path.endswith('json')
            with open(cached_pre_eval_path, 'r') as f:
                cached_js = json.load(f)
            # assert len(cached_js) == len(requests), 'The size of cached pre mismatches the size of requests!'
            if len(cached_js) != len(requests):
                print(f'NOTE: Truncate cache to {len(requests)}!')
                cached_js = cached_js[:len(requests)]
            cached_pre = [{'pre': item['pre']} for item in cached_js]
        else:
            cached_pre = None

        # pre_kss = []
        # pre_vss = []
        if cached_pre is None:
            for i, request in tqdm(enumerate(requests)):
                assert self.alg_name != 'IKE'
                if kwargs.get('debug', False) is False:
                    metric, pre_collect = compute_bias_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,self.hparams.device, test_generation=test_generation, post=False, collect=collect, dump=dump)
                    metrics = {
                        "pre": metric
                    }
                    if 'shuffled_subject' in metric: request["shuffled_subject"] = metric["shuffled_subject"]
                    if 'long_context' in metric: request["long_context"] = metric["long_context"]
                    if 'long_context_held' in metric: 
                        request["long_context_held"] = metric["long_context_held"]
                else:
                    metrics = {'pre': None}
                    collected_outputs = {}
                locality_metrics.append(metrics)
                if dump:
                    collects['pre'].append(pre_collect)
                all_metrics.append(metrics)
        else:
            print(f'Skip pre-eval! Using cached pre from path : {cached_pre_eval_path}')
            all_metrics = cached_pre
        
        # add evals into requests
        # if 'long_context' in all_metrics
        for i in range(len(all_metrics)):
            # append keys
            append_keys = ['long_context', 'shuffled_subject', 'para_attack']
            for key in append_keys:
                if key in all_metrics[i]['pre']: requests[i][key] = all_metrics[i]['pre'][key]
        
        extra_metrics = {}
        cache_inputs = []
        cache_labels = []
        for i, request in enumerate(requests):
            start = time()
            assert self.alg_name != 'IKE'
            if isinstance(request["prompt"], list):
                new_requests = [{"prompt": p, "target_new": t, "subject":request["subject"] }for p, t in zip(request["prompt"], request["target_new"])]
            else: 
                new_requests = [request]
            # print(new_requests)
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
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")
            # if in debug mode, directly jump into the next case.
            if kwargs.get('debug', False) is True: continue

            start = time()
            post_metric, post_collect = compute_bias_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation, collect=collect, dump=dump)
            if dump:
                collects['post'].append(post_collect)
            all_metrics[i].update({
                'case_id': i,
                "requested_rewrite": request,
                "time": exec_time,
                "post":post_metric
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
            elif hasattr(edited_model,'peft_config') and keep_original_weight: # peft in edited model
                edited_model.unload()
                del self.model.peft_config
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
                # all_metrics[i]['pre'].pop('locality')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
        return all_metrics, edited_model, weights_copy, extra_metrics, locality_metrics, requests, collects

    def ike_edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             target_true: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             collect='',
             dump=True,
             cached_pre_eval_path='',
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
            target_true = [target_true,]

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

        requests = self._prepare_requests(prompts, target_new, target_true, ground_truth, rephrase_prompts,
                                          locality_inputs, portability_inputs, 
                                          **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        assert self.alg_name != 'FT-Api', "We do not support FT-Api for erasor."

        all_metrics = []
        locality_metrics = []
        collects = {
            'pre': [],
            'post': []
        }
        if cached_pre_eval_path != "":
            assert os.path.exists(cached_pre_eval_path) and cached_pre_eval_path.endswith('json')
            with open(cached_pre_eval_path, 'r') as f:
                cached_js = json.load(f)
            # assert len(cached_js) == len(requests), 'The size of cached pre mismatches the size of requests!'
            if len(cached_js) != len(requests):
                print(f'NOTE: Truncate cache to {len(requests)}!')
                cached_js = cached_js[:len(requests)]
            cached_pre = [{'pre': item['pre']} for item in cached_js]
        else:
            cached_pre = None

        # pre_kss = []
        # pre_vss = []
        if cached_pre is None:
            for i, request in tqdm(enumerate(requests)):
                assert self.alg_name == 'IKE'
                metric = {"pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True)}
                if 'shuffled_subject' in metric["pre"]: request["shuffled_subject"] = metric["pre"]["shuffled_subject"]
                if 'long_context' in metric["pre"]: request["long_context"] = metric["pre"]["long_context"]
                if 'long_context_held' in metric["pre"]: 
                    request["long_context_held"] = metric["pre"]["long_context_held"]
                all_metrics.append(metric)
        else:
            print(f'Skip pre-eval! Using cached pre from path : {cached_pre_eval_path}')
            all_metrics = cached_pre
        
        # add evals into requests
        # if 'long_context' in all_metrics
        for i in range(len(all_metrics)):
            # append keys
            append_keys = ['long_context', 'shuffled_subject', 'para_attack']
            for key in append_keys:
                if key in all_metrics[i]['pre']: requests[i][key] = all_metrics[i]['pre'][key]
        
        extra_metrics = {}
        cache_inputs = []
        cache_labels = []
        for i, request in enumerate(requests):
            start = time()
            assert self.alg_name == 'IKE'
    
            # print(new_requests)
            edited_model, weights_copy, icl_examples = self.model, {},  self.apply_algo(
                self.model,
                self.tok,
                request,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=False,
                train_ds=kwargs['train_ds'],
                cache_inputs=cache_inputs,
                cache_labels=cache_labels
            )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            post_metric = compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device)

            all_metrics[i].update({
                'case_id': i,
                "requested_rewrite": request,
                "time": exec_time,
                "post":post_metric
            })
            
            
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
            #     # all_metrics[i]['pre'].pop('locality')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
        return all_metrics, edited_model, weights_copy, extra_metrics, locality_metrics, requests, collects

    # def edit_dataset(self,
    #                  ds: Dataset,
    #                  keep_original_weight=False,
    #                  verbose=True
    #                  ):
    #     # Make Sure dataset supported
    #     assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in DS_DICT.values()]) > 0 \
    #     or print(f'DataSet {ds} not supported yet.')

    #     is_singleton = SingletonEditor.is_singleton_method(self.alg_name)

    #     if is_singleton:
    #         num_edits = 1 # Single editor method found
    #     else:
    #         assert hasattr(self.hparams, 'batch_size') or \
    #                print(f'Method {self.alg_name} found, pls set the batch_size correctly')

    #         num_edits = self.hparams.batch_size

    #     all_metrics = []

    #     for record_chunks in tqdm(self._chunks(ds, num_edits), desc='Editing dataset', total=len(ds)/num_edits):

    #         start = time()
    #         edited_model, weights_copy = self.apply_algo(
    #             self.model,
    #             self.tok,
    #             record_chunks,
    #             self.hparams,
    #             copy=False,
    #             return_orig_weights=True,
    #             keep_original_weight=keep_original_weight
    #         )
    #         exec_time = time() - start
    #         LOG.info(f"Execution took {exec_time}")

    #         start = time()
    #         all_metrics = []
    #         for i, request in enumerate(record_chunks):

    #             metrics = {
    #                 'case_id': request['case_id'],
    #                 "requested_rewrite": request,
    #                 "time": exec_time,
    #                 "post":compute_bias_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device)
    #                 # "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
    #             }
    #             all_metrics.append(metrics)

    #         with torch.no_grad():
    #             for k, v in weights_copy.items():
    #                 nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

    #         for i, request in enumerate(record_chunks):
    #             all_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
    #                                                   self.hparams.device)

    #             if verbose:
    #                 LOG.info(
    #                     f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
    #                 )

    #         LOG.info(f"Evaluation took {time() - start}")

    #     return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          target_true: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'target_true': target_true_,
            'ground_truth': ground_truth_,
            'portability': {},
            'locality': {}
        }
        for prompt, ground_truth_, target_new_, target_true_ in zip(prompts, ground_truth, target_new, target_true)
        ]
        count = 0
        failed = 0 
        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                if isinstance(prompt_, list):
                    for p in prompt_:
                        if subject_ in p:
                            pass
                        else:
                            print(f'Subject:{subject_} do not exist in prompt: {p}')
                            failed += 1
                        count+=1
                        # assert subject_ in p or print(f'Subject:{subject_} do not exist in prompt: {p}')
                else:
                    if not subject_ in prompt_:
                        print(f'Subject:{subject_} do not exist in prompt: {prompt_}')
                    # assert subject_ in prompt_ or print(f'Subject:{subject_} do not exist in prompt: {prompt_}')
            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )
        if 'original_lprobs' in kwargs and len(kwargs['original_lprobs']) > 0:
            assert len(kwargs['original_lprobs']) == len(prompts)
            for i, request in enumerate(requests):
                request.update(
                    {
                        'original_lprobs': kwargs['original_lprobs'][i]
                    })
        # if 'original_pairs' in kwargs and len(kwargs['original_pairs']) > 0:
        #     assert len(kwargs['original_pairs']) == len(prompts)
        #     for i, request in enumerate(requests):
        #         request.update(
        #             {
        #                 'original_pairs': kwargs['original_pairs'][i]
        #             })
        
        if 'original_target_lprobs' in kwargs and kwargs['original_target_lprobs'] != []:
            assert len(kwargs['original_target_lprobs']) == len(prompts)
            for i, request in enumerate(requests):
                request.update(
                    {
                        'original_target_lprobs': kwargs['original_target_lprobs'][i]
                    })
        
        if 'token_lists' in kwargs and len(kwargs["token_lists"])>0:
            assert len(kwargs['token_lists']) == len(prompts)
            for i, request in enumerate(requests):
                request.update(
                    {
                        'token_lists': kwargs['token_lists'][i]
                    })
        
        if 'original_token_lprobs_lists' in kwargs and len(kwargs["original_token_lprobs_lists"])>0:
            assert len(kwargs['original_token_lprobs_lists']) == len(prompts)
            for i, request in enumerate(requests):
                request.update(
                    {
                        'original_token_lprobs_lists': kwargs['original_token_lprobs_lists'][i]
                    })
        
        if 'rephrase_targets' in kwargs and len(kwargs["rephrase_targets"])>0:
            assert len(kwargs['rephrase_targets']) == len(prompts)
            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_target': kwargs['rephrase_targets'][i]
                    })
        if 'rephrase_subjects' in kwargs:
            assert len(kwargs['rephrase_subjects']) == len(prompts)
            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_subjects': kwargs['rephrase_subjects'][i]
                    })
                
        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        
        # add para pairs for bias eval
        if 'para_pairs' in kwargs and len(kwargs['para_pairs']) > 0:
            for i, request in enumerate(requests):
                request.update(
                    {
                        'para_pairs': kwargs['para_pairs'][i],
                    }
                )
        
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                    locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                == len(requests) or print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )
        
        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
                == len(requests) or print('One Edit instance needs one portability input.....')

                for i, request in enumerate(requests):
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
        return requests


# if __name__ == "__main__":
#
#     editor = BaseEditor(alg_name='KN', model_name='/nature/peng/serac/hugging_cache/t5-3b-finetuned-counterfact-10000', hparams_fname='t5-3b.json')
#
#     editor.edit(
#         prompts='What university did Watts Humphrey attend?',
#         ground_truth='Illinois Institute of Technology',
#         target_new='University of Michigan'
#     )
#
#     metrics, edited_model, _ = editor.edit(prompts='What university did Watts Humphrey attend?', ground_truth='Illinois Institute of Technology', target_new='University of Michigan')


    def edit_requests(self,
             requests,
             keep_original_weight=False,
             verbose=True,
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
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1 or \
                      print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

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

        all_metrics = []
        for i, request in tqdm(enumerate(requests)):
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
            #     metrics = {
            #         "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
            #                                          request, self.hparams.device, pre_edit=True)
            #     }
            # else:
            assert self.alg_name != 'IKE'
            metrics = {
                "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                        self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
            }
            all_metrics.append(metrics)

        for i, request in tqdm(enumerate(requests)):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys() or print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
                all_metrics[i]['pre'].pop('locality')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )

            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    # "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                     "post": compute_bias_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if self.alg_name == 'KN':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                elif self.alg_name == 'LoRA' and keep_original_weight:
                    edited_model.unload()
                    del self.model.peft_config
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

        return all_metrics, edited_model, weights_copy
