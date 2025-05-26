import sys, json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict
from scipy import stats
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.manifold import TSNE
import seaborn as sns
from adjustText import adjust_text
import fire

def main():
    # input_file = f"results_llama/fact_edit/rome/metrics/{alg}_results_seed0.json"
    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf8') as f:
        jss = json.load(f)
    
    if len(sys.argv) > 2:
        head = int(sys.argv[2])
    else: head = -1
    if len(sys.argv) > 3:
        tail = int(sys.argv[3])
    else:
        tail = -1

    if head != -1 and tail != -1:
        jss = jss[head:tail]
    elif head != -1:
        jss = jss[:head]
        # jss = jss[:head]

    all_metrics = {}
    plot_metrics = defaultdict(list)
    for js in jss:
        try:
            pre = js['pre']
            post = js['post']
            case_id = js['case_id']
        except:
            continue
        
        # compute pre
        # for k in ['pre', 'post']:
        for k in ['post']:
            js_ret = gather_all_metrics_each_item(js[k])
            for mkey in js_ret:
                kkey = f"{k}-{mkey}"
                # init metrics
                if kkey not in all_metrics: all_metrics[kkey] = [0, 0]
                kk_suc, kk_tot = js_ret[mkey]
                if kk_tot == 0:
                    continue
                try:
                    all_metrics[kkey][0] += kk_suc
                    all_metrics[kkey][1] += kk_tot 
                except:
                    breakpoint()
            
            plot_metrics["acc"].append(js_ret["acc"][0]/js_ret["acc"][1])
            if 'rephrase_acc' in js_ret:
                plot_metrics["rephrase_subject_acc"].append(js_ret["rephrase_acc"][0]/js_ret["rephrase_acc"][1])
            if 'shuffled_acc' in js_ret:
                plot_metrics["shuffled_subject_acc"].append(js_ret["shuffled_acc"][0]/ js_ret["shuffled_acc"][1])
            if 'long_acc' in js_ret:
                plot_metrics["long_context_acc"].append(js_ret["long_acc"][0]/js_ret["long_acc"][1])

    # print all results
    for key in all_metrics:
        print(f"{key}: {all_metrics[key][0]/all_metrics[key][1]*100}%")
    # compare_keys(plot_metrics, metric)

def gather_all_metrics_each_item(js):
    ret = {}
    if 'acc' in js:
        ret['acc'] = (sum(js['acc']), len(js['acc']))
    # if 'rev' in js:
    #     ret['rev'] = (sum(js['rev']), len(js['rev']))
    # if 'para_attack_rev' in js:
    #     ret['para_attack_succ'] = (sum(js['para_attack_succ']), len(js['para_attack_succ']))
    # if 'para_attack_rev' in js:
    #     ret['para_attack_rev'] = (sum(js['para_attack_rev']), len(js['para_attack_rev']))
    if 'locality' in js:
        if isinstance(js['locality']['zsre_acc'], list):
            # assert len(js['locality']['zsre_acc']) == 1
            ret['loca_acc'] = sum(js['locality']['zsre_acc']), len(js['locality']['zsre_acc'])
        else:
            ret['loca_acc'] = js['locality']['zsre_acc'], 1
            
    if 'edited_rephrase_sub_target_new_acc' in js:
        half = math.ceil(len(js['edited_rephrase_sub_target_new_acc'])/2)
        rephrase_od_acc, rephrase_id_acc = 0, 0
        for item in js['edited_rephrase_sub_target_new_acc'][half:]:
            rephrase_od_acc += item[0]
        for item in js['edited_rephrase_sub_target_new_acc'][:half]:
            rephrase_id_acc += item[0]
        ret['rephrase_od_acc'] = (rephrase_od_acc , len(js['edited_rephrase_sub_target_new_acc'][half:]))
        ret['rephrase_id_acc'] = (rephrase_id_acc , len(js['edited_rephrase_sub_target_new_acc'][:half]))
    
    if 'edited_shuffled_sub_target_new_acc' in js:
        half = math.ceil(len(js['edited_shuffled_sub_target_new_acc'])/2)

        shuffle_od_acc, shuffle_id_acc = 0, 0
        for item in js['edited_shuffled_sub_target_new_acc'][half:]:
            shuffle_od_acc += item[0]
        for item in js['edited_shuffled_sub_target_new_acc'][:half]:
            shuffle_id_acc += item[0]

        ret['shuffle_od_acc'] = (shuffle_od_acc , len(js['edited_shuffled_sub_target_new_acc'][half:]))
        ret['shuffle_id_acc'] = (shuffle_id_acc , len(js['edited_shuffled_sub_target_new_acc'][:half]))
    
    if 'edited_long_context_target_new_acc' in js:
        ret["long_acc"] = (js['edited_long_context_target_new_acc'][0], 1)

    if 'edited_long_context_held_target_new_acc' in js:
        long_held_acc = 0.0
        for item in js['edited_long_context_held_target_new_acc']:
            long_held_acc += item[0]
        ret['long_held_acc'] = (long_held_acc , len(js['edited_long_context_held_target_new_acc']))
        # ret["long_held_acc"] /= len(js['edited_long_context_held_target_new_acc'])

    return ret


if __name__ == "__main__":
    main()
    # compare_keys()