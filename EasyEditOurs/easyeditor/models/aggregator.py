import torch
import re
import math
import transformers
from functools import partial
# from .agg_utils import AverageMeter, get_context_templates
from . import agg_utils
from .rome import repr_tools
from ..util import nethook
import torch.nn.functional as F

class Aggregator(torch.nn.Module):
    def __init__(self, hparams, model, tok, device):
        super(Aggregator, self).__init__()
        # self.config = config
        self.log_dict = {}
        self.model = model
        self.tok = tok
        self.hparams = hparams
        
        self.device = device


        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        layers = self.hparams.layers
        
        self.layer_names = [hparams.rewrite_module_tmp.format(layer) for layer in layers]
        layer = agg_utils.get_module(self.model, self.layer_names[0])
        
        if isinstance(layer, torch.nn.Linear):
            feature_dim = layer.in_features
        elif 'gpt2' in hparams.model_name:
            feature_dim = layer.weight.shape[0]
        else:
            raise
        
        self.adaptors = [Adaptor(hparams, feature_dim).to('cuda') for _ in layers]

    def __call__(self, **kwargs):
        adapt_fn_dict = {}
        for i, layer_name in enumerate(self.layer_names):
            cur_adapt_fn = partial(adapt_fn, layer_name=layer_name, adaptor=self.adaptors[i])
            adapt_fn_dict[layer_name] = cur_adapt_fn
            
        with nethook.TraceInputDict(
            self.model, 
            layers=self.layer_names, 
            # edit_input=cur_adapt_fn
            edit_input_dict=adapt_fn_dict
        ) as tr:
            out = self.model(**kwargs)

        return out
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    def edit(self, request, target_vectors):
        self.training = True
        for i in range(len(self.adaptors)):
            self.adaptors[i].training = True

        layers = self.hparams.layers
        
        # make sure target vector is not changed
        if isinstance(target_vectors, list):
            target_vectors = torch.stack(target_vectors).clone()
        else:
            target_vectors = target_vectors.detach().clone()
            if target_vectors.ndim == 1: target_vectors = target_vectors[None]
        assert target_vectors.shape[0] == len(layers)
        target_vectors = target_vectors.float()
        
        # not sure the difference between these two impls. 
        try:
            context_templates = agg_utils.get_context_templates_rome(self.model, self.tok, self.hparams.context_template_length_params)
        except:
            context_templates = agg_utils.get_context_templates_rome(self.model, self.tok, [[5, 10], [10, 10]]) # use default value from ROME.
            # context_templates = [x for xs in context_templates for x in xs]

        # get inv_cov
        from .rome.compute_u import get_inv_cov
        Cs = []
        for layer in layers:
            C = get_inv_cov(
                self.model,
                self.tok,
                self.hparams.rewrite_module_tmp.format(layer),
                self.hparams.mom2_dataset,
                self.hparams.mom2_n_samples,
                self.hparams.mom2_dtype,
                hparams=self.hparams,
            )
            dd = torch.sqrt(torch.diag(C))
            norm = torch.outer(dd, dd)
            if self.hparams.agg_norm_c:
                norm_C = C / norm
            else:
                norm_C = C
            Cs.append(norm_C)

        # define loss weights
        loss_weights = {
            'agg': self.hparams.agg_loss_weight,
            'spread': self.hparams.spread_loss_weight,
            'kl': self.hparams.kl_loss_weight,
            'target_consistency': self.hparams.target_consistency_weight
        }
        
        # lr = 5e-4 # llama 5e-3
        lr = self.hparams.agg_lr

        opt = torch.optim.Adam(
            (param for adaptor in self.adaptors for param in adaptor.parameters()),
            lr=lr,
            weight_decay=0,
        )
        # self.adaptor.print_trainable_parameters()
        
        loss_meter = agg_utils.AverageMeter()

        # fill in prompt, leave subject blank
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

            prompt_template = request['prompt'].replace(request['subject'], '{}')
        else:
            prompt_template = request['prompt']

        prompt_templates = [prompt_template, ]
        context_templates = context_templates[:]

        assert self.hparams.agg_add_ori is True
        base_subjects = [request['subject'],]
        subtoken = self.hparams.fact_token[len("subject_") :]

        base_templates = get_templates(context_templates, prompt_templates)
        ctxs, idxs = get_ctx_idx(self.tok, base_subjects, base_templates, subtoken)
        
        # add rephrase templates
        if self.hparams.agg_add_rephrases:
            raise
            rep_p_templates = [reph_prompt.replace(request['subject'], "{}") for reph_prompt in request['rephrase_prompt']]
            if self.hparams.agg_train_on_held:
                half = math.ceil(len(rep_p_templates)/2)
                rep_p_templates = rep_p_templates[:half]
            # filter if there are multiple {}
            # rep_p_templates = [text for text in rep_p_templates if len(re.findall(r'\{\}', text)) == 1]
            prompt_templates += rep_p_templates
            
        if self.hparams.agg_add_para_attack and 'para_attack' in request:
            raise
            para_attacks = request['para_attack']
            para_attack_templates = [reph_prompt.replace(request['subject'], "{}") for reph_prompt in para_attacks]
            if self.hparams.agg_train_on_held:
                half = math.ceil(len(para_attack_templates)/2)
                para_attack_templates = para_attack_templates[:half]
            # filter if there are multiple {}
            # para_attack_templates = [text for text in para_attack_templates if len(re.findall(r'\{\}', text)) == 1]
            prompt_templates += para_attack_templates
        
        if self.hparams.agg_add_long_context:
            context_templates_long = []
            if self.hparams.agg_train_on_held: # for 
                # half = math.ceil(len(para_attack_templates)/2)
                # para_attack_templates = para_attack_templates[:half]
                random_texts = request['long_context_held']
                context_templates_long += random_texts
            else:
                assert 'long_context' in request
                random_text = request['long_context']
                context_templates_long += [random_text,]
            
            templates_long = get_templates(context_templates_long, prompt_templates)
            # base_templates = get_templates(context_templates, prompt_templates)
            ctxs_l, idxs_l = get_ctx_idx(self.tok, base_subjects, templates_long, subtoken)
            ctxs += ctxs_l
            idxs += idxs_l

    

        # repha subjects
        if self.hparams.agg_train_on_held:
            half = math.ceil(len(request['rephrase_subjects'])/2)
            reph_subjects = request['rephrase_subjects'][:half] # only use first half
        else:
            reph_subjects = request['rephrase_subjects']
        
        ctxs_r, idxs_r = get_ctx_idx(self.tok, reph_subjects, base_templates, subtoken)
        ctxs += ctxs_r
        idxs += idxs_r
        
        # subjects = reph_subjects
        if self.hparams.agg_add_shuffle and 'shuffled_subject' in request:
            if self.hparams.agg_train_on_held:
                half = math.ceil(len(request['shuffled_subject'])/2)
                shuf_subjects = request['shuffled_subject'][:half]
            else:
                shuf_subjects = request['shuffled_subject']
                
            # base_templates = get_templates(context_templates, prompt_templates)
            ctxs_s, idxs_s = get_ctx_idx(self.tok, shuf_subjects, base_templates, subtoken)
            ctxs += ctxs_s
            idxs += idxs_s
            
            
        import random
        batch_size = self.hparams.agg_batch_size
        index = 0
        # create data tuple
        ctx_idx_tup = [(i, ctxs[i], idxs[i]) for i in range(len(ctxs))]
        print(f'Data Size: {len(ctx_idx_tup)}')
        random.shuffle(ctx_idx_tup)

        for i, c in enumerate(ctx_idx_tup):
            if c[1] == request["prompt"]:
                target_idx = i
                
        original = ctx_idx_tup.pop(target_idx)

        print(f'Data Size: {len(ctx_idx_tup)}')        
        
        for it in range(self.hparams.num_train_agg_steps):
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
            # add original case to each batch
            batch += [original]
            batch_id = [item[1] for item in batch]
            batch_contexts = [item[1] for item in batch]
            batch_idxs = [item[2] for item in batch]
            contexts_tok = self.tok(
                batch_contexts,
                return_tensors="pt",
                padding=True,
            ).to(f"cuda:{self.hparams.device}")
            bsz, L = contexts_tok.input_ids.shape[:2]
            

        
            adapt_fn_dict = {}
            for i, layer_name in enumerate(self.layer_names):
                cur_adapt_fn = partial(adapt_fn, layer_name=layer_name, adaptor=self.adaptors[i])
                adapt_fn_dict[layer_name] = cur_adapt_fn
                
            with nethook.TraceInputDict(
                module=self.model, 
                layers=self.layer_names, 
                retain_input=True,
                edit_input_dict=adapt_fn_dict
            ) as tr:
                _ = self.model(**contexts_tok)

            # make agg loss mask
            assert all(len(idx)==1 for idx in batch_idxs)
            idxs_t = torch.tensor(batch_idxs).long().to(f"cuda:{self.hparams.device}")
            
            mask = torch.ones(bsz, L, dtype=torch.bool).to(f"cuda:{self.hparams.device}")
            batch_indices = torch.arange(bsz)[:, None].to(mask.device)
            
            mask[batch_indices, idxs_t] = False

            
            loss_dict = {}
            init_kl_reprs = [tr[layer_name].input.detach() for layer_name in self.layer_names] # [bsz, l, D]
            # original_reprs = [kl[-1] for kl in init_kl_reprs]
            # init_kl_reprs = [kl[:-1]for kl in init_kl_reprs]
            for i in range(len(layers)):
                target_vector = target_vectors[i]                
                norm_C = Cs[i]
                init_kl_repr = init_kl_reprs[i][:-1]
                original_repr =  init_kl_reprs[i][-1]
                

                # normalize init_kl_repr
                repr = self.adaptors[i](init_kl_repr)
                # repr, cls = self.adaptors[i](init_kl_repr)
                targeted_repr = self.adaptors[i](original_repr)
                target_consistency_loss = F.mse_loss(targeted_repr[~mask[-1]], original_repr[~mask[-1]] ,reduction='mean')
                # targeted_repr = self.adaptors[i](target_vector)
                # target_consistency_loss = F.mse_loss(targeted_repr, target_vector ,reduction='mean')
            
                # if self.hparams.cls_loss_weight > 0:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                assert self.hparams.cls_loss_weight == 0.0, 'We cannot use cls loss when use edit_input_fn.'
                # cls_loss = loss_fn(cls.squeeze(-1), (~mask).float())
                cls_loss = 0.0
                
                agg_repr = repr[~mask[:-1]]
                # before and after lora.
                kl_repr = repr[mask[:-1]]
                init_kl_repr = init_kl_repr[mask[:-1]]
               
                if self.hparams.agg_loss_norm:
                    norm_agg_repr = agg_repr / agg_repr.norm(dim=-1, keepdim=True)
                else:
                    norm_agg_repr = agg_repr
                
                # v5: remove abs in agg loss
                if not self.hparams.agg_remove_c:
                    agg_losses = -1 * (norm_agg_repr @ norm_C @ target_vector.to(norm_agg_repr.dtype))
                else:
                    agg_losses = -1 * (norm_agg_repr @ target_vector.to(norm_agg_repr.dtype))
                    
                agg_loss = agg_losses.mean()
                # v6: add kl loss
                if self.hparams.kl_loss_type == 'whiten_diff':
                    if self.hparams.agg_remove_c:
                        kl_loss = torch.abs(init_kl_repr @ (kl_repr - init_kl_repr).T)
                    else:
                        kl_loss = torch.abs(init_kl_repr @ norm_C @ (kl_repr - init_kl_repr).T)
                elif self.hparams.kl_loss_type == 'whiten':
                    if not self.hparams.agg_remove_c:
                        kl_loss = -1 * init_kl_repr @ norm_C @ kl_repr.T
                    else:
                        kl_loss = -1 * init_kl_repr @ kl_repr.T
                elif self.hparams.kl_loss_type == 'mse':
                    
                    kl_loss = F.mse_loss(init_kl_repr, kl_repr, reduction='mean')
                else:
                    raise
                kl_loss = kl_loss.mean()
            
                if not self.hparams.agg_remove_c:
                    spread_loss = torch.abs(kl_repr @ norm_C @ target_vector).mean()
                else:
                    spread_loss = torch.abs(kl_repr @ target_vector).mean()
                
                loss_dict[layer] = {
                    'agg_loss': agg_loss,
                    'spread_loss': spread_loss,
                    'kl_loss': kl_loss,
                    'cls_loss': cls_loss,
                    'target_consistency_loss': target_consistency_loss
                }
            
            agg_loss_sum = sum([item['agg_loss'] for key, item in loss_dict.items()])
            spread_loss_sum = sum([item['spread_loss'] for key, item in loss_dict.items()])
            kl_loss_sum = sum([item['kl_loss'] for key, item in loss_dict.items()])
            cls_loss_sum = sum([item['cls_loss'] for key, item in loss_dict.items()])
            target_consistency_loss_sum = sum([item['target_consistency_loss'] for key, item in loss_dict.items()])
            
            loss = agg_loss_sum * loss_weights['agg'] + spread_loss_sum * loss_weights['spread'] + kl_loss_sum * loss_weights['kl'] + cls_loss_sum * self.hparams.cls_loss_weight + target_consistency_loss_sum * loss_weights['target_consistency']
            print(f"Step {it}: Agg loss {agg_loss_sum.item()}; Spread loss: {spread_loss_sum}; KL loss: {kl_loss_sum}; CLS loss: {cls_loss_sum}; Target Consistency loss: {target_consistency_loss_sum}; {loss_weights['target_consistency']}")
            
            loss_meter.update(loss.item(), n=1)
            loss.backward()
            opt.step()
        
        self.training = False
        # self.adaptor.training = False
        for i in range(len(self.adaptors)):
            self.adaptors[i].training = False

        

class Adaptor(torch.nn.Module):
    def __init__(self, hparams, feature_dim):
        super(Adaptor, self).__init__()
        
        self.hparams = hparams        
        # default the same as in lora aggregator
        r = 32
        lora_alpha = 32
        lora_dropout = 0.1
        
        self.lora_A = torch.nn.Linear(feature_dim, r, bias=False)
        self.lora_B = torch.nn.Linear(r, feature_dim, bias=False)
        
        import math
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

        self.scaling = lora_alpha / r

        # self.dropout = self.lora_dropout[active_adapter]
        if lora_dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=lora_dropout)
        else:
            self.dropout = torch.nn.Identity()
        
        # cls
        if hparams.agg_use_cls:
            inner_dim = int(feature_dim * hparams.agg_gate_inner_dim_ratio)
            self.cls_mlp1 = torch.nn.Linear(feature_dim, inner_dim, bias=True)
            self.cls_act = torch.nn.GELU()
            self.cls_mlp2 = torch.nn.Linear(inner_dim, 1, bias=True)
            self.cls_sigmoid = torch.nn.Sigmoid()
        
        self.cls_threshold = self.hparams.cls_threshold
    
    def forward(self, x):
        result = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling # [bsz, L, D]
        if self.hparams.agg_use_cls:
            # compute cls
            cls_out = self.cls_mlp2(self.cls_act(self.cls_mlp1(x)))
            cls_pred = self.cls_sigmoid(cls_out)
        else:
            cls_pred = torch.ones((result.shape[0], result.shape[1], 1)).to(result.device) # dummy pred
        
        if self.training is True:
            result = x + result * cls_pred
            return result
        else:
            result = x + result * (cls_pred > self.cls_threshold).float()
            return result

def adapt_fn(layer_name, adaptor, inputs, layer):
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1 # we only deal with case with only one inputs
    inputs = inputs[0]
    if layer == layer_name:
        # return tuple(adaptor(inputs[0]), )
        return adaptor(inputs)
    else:
        return inputs

def get_ctx_idx(tok, subjects, templates, subtoken):
    # subtoken=self.hparams.fact_token[len("subject_") :]
    words = [word for word in subjects for _ in templates]
    stack_templates = [temp for word in subjects for temp in templates]
    contexts = [stack_templates[i].format(words[i]) for i in range(len(words))] # full inputs
    idxs = repr_tools.get_words_idxs_in_templates(tok, stack_templates, words, subtoken)
    return contexts, idxs

def get_templates(context_templates, prompt_templates):
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
    return templates