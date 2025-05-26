
import sys
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import numpy as np

results_dir = sys.argv[1]
alg = sys.argv[2]
inv_cov = torch.load('EasyEditOurs/data/stats/_llama2-7b/wikitext_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.inv_conv.pt')

def calculate_whiten_cosine(a, b):
    left = torch.matmul(a, inv_cov)
    return torch.inner(left.flatten(), b.flatten())



def compare_keys_qualitive(pre_original_keys, pre_rephrase_keys, pre_shuffled_keys, pre_long_keys, post_original_keys, post_rephrase_keys, post_shuffled_keys, post_long_keys, collected_results, idx=40):
    # prepare data
    other_original_keys = pre_original_keys[:idx] + pre_original_keys[idx+1:]
    random_rep = random.choice(other_original_keys)[0]

    pre_keys = [pre_original_keys[idx], pre_rephrase_keys[idx], pre_shuffled_keys[idx], pre_long_keys[idx]]
    post_keys = [post_original_keys[idx], post_rephrase_keys[idx], post_shuffled_keys[idx], post_long_keys[idx]]
    original_key = [pre_original_keys[idx], post_original_keys[idx]]
   
    original_pair_dis = calculate_pairwise_variance_rep(pre_keys)
    post_pair_dis = calculate_pairwise_variance_rep(post_keys)
    key_shift = calculate_pairwise_variance_rep(original_key)
    
    print("Original Sim: ", original_pair_dis)
    print("Post Sim: ", post_pair_dis)
    print("Key Sim: ",key_shift)
    
    pre_rephrase_keys_original = [pre_original_keys[idx], pre_rephrase_keys[idx]]
    pre_shuffle_keys_original = [pre_original_keys[idx], pre_shuffled_keys[idx]]
    pre_long_keys_original =  [pre_original_keys[idx], pre_long_keys[idx]]
    pre_original_original = [pre_original_keys[idx], pre_original_keys[idx]]

    original_rephrase_keys_sim = calculate_pairwise_variance_rep(pre_rephrase_keys_original)
    original_shuffle_keys_sim = calculate_pairwise_variance_rep(pre_shuffle_keys_original)
    original_long_keys_sim = calculate_pairwise_variance_rep(pre_long_keys_original)
    original_original_sim = calculate_pairwise_variance_rep(pre_original_original)

    post_rephrase_keys_original = [post_original_keys[idx], post_rephrase_keys[idx]]
    post_shuffle_keys_original = [post_original_keys[idx], post_shuffled_keys[idx]]
    post_long_keys_original =  [post_original_keys[idx], post_long_keys[idx]]
    # post_original_original = [post_original_keys[idx], pre_original_keys[idx]]

    post_original_rephrase_keys_sim = calculate_pairwise_variance_rep(post_rephrase_keys_original)
    post_original_shuffle_keys_sim = calculate_pairwise_variance_rep(post_shuffle_keys_original)
    post_original_long_keys_sim = calculate_pairwise_variance_rep(post_long_keys_original)
    # original_original_sim = calculate_pairwise_variance_rep(post_original_original)

    post_random_key = [post_original_keys[idx], random_rep]
    random_key = [pre_original_keys[idx], random_rep]
    pre_original_random_key = calculate_pairwise_variance_rep(random_key)
    post_original_random_key = calculate_pairwise_variance_rep(post_random_key)
    return original_rephrase_keys_sim/original_original_sim, original_shuffle_keys_sim/original_original_sim, original_long_keys_sim/original_original_sim, pre_original_random_key / original_original_sim, post_original_rephrase_keys_sim/original_original_sim, post_original_shuffle_keys_sim/original_original_sim, post_original_long_keys_sim/original_original_sim, post_original_random_key / original_original_sim, original_original_sim

def calculate_pairwise_variance_rep(data):
    data = torch.vstack(data)
    all_data = np.array(data)
    left = all_data @ inv_cov.numpy() # [1, 11]
    right = all_data
    distance_matrix = np.inner(left, right)
    np.fill_diagonal(distance_matrix, 0)
    distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)].tolist()
    distances = sum(distances) / len(distances)
 
    return distances


def collect_eval_results(file):
    with open(file, 'r', encoding='utf8') as f:
        jss = json.load(f)
    
    all_metrics = {}
    plot_metrics = defaultdict(list)
    fail_cases = []
    for i, js in enumerate(jss):
        for stage in ['post']:
            js_ret = gather_all_metrics_each_item(js[stage])
            for mkey in js_ret:
                kkey = f"{stage}-{mkey}"
                if kkey not in all_metrics:
                    all_metrics[kkey] = [0, 0]
                kk_acc, kk_num = js_ret[mkey]
                if kk_num == 0:
                    continue
                try:
                    all_metrics[kkey][0] += kk_acc
                    all_metrics[kkey][1] += kk_num
                except:
                    breakpoint()
            
            # collect list for each sample
            
            plot_metrics["acc"].append(js_ret["acc"][0]/js_ret["acc"][1])
            if 'rephrase_acc' in js_ret:
                plot_metrics["rephrase_subject_acc"].append(js_ret["rephrase_acc"][0]/js_ret["rephrase_acc"][1])
            if 'shuffled_acc' in js_ret:
                plot_metrics["shuffled_subject_acc"].append(js_ret["shuffled_acc"][0]/ js_ret["shuffled_acc"][1])
            if 'long_acc' in js_ret:
                plot_metrics["long_context_acc"].append(js_ret["long_acc"][0]/js_ret["long_acc"][1])
        if js_ret["acc"][0]/js_ret["acc"][1] == 0:
            fail_cases.append(i)
        
    # print all results
    for key in all_metrics:
        print(f"{key}: {all_metrics[key][0]/all_metrics[key][1]*100}%")
    return fail_cases
    



def gather_all_metrics_each_item(js):
    ret = {}
    ret['acc'] = (sum(js['acc']), len(js['acc']))
    ret['rev'] = (sum(js['rev']), len(js['rev']))
    if  "locality" in js:
        if isinstance(js['locality']['zsre_acc'], list):
            ret['loca_acc'] = sum(js['locality']['zsre_acc']), len(js['locality']['zsre_acc'])
        else:
            ret['loca_acc'] = js['locality']['zsre_acc'], 1
    if 'edited_rephrase_sub_target_new_acc' in js:
        rephrase_acc = 0
        for item in js['edited_rephrase_sub_target_new_acc']:
            rephrase_acc += item[0]
        ret['rephrase_acc'] = (rephrase_acc , len(js['edited_rephrase_sub_target_new_acc']))
    
    if 'edited_shuffled_sub_target_new_acc' in js:
        shuffle_acc = 0
        for item in js['edited_shuffled_sub_target_new_acc']:
            shuffle_acc += item[0]
        ret['shuffled_acc'] = (shuffle_acc , len(js['edited_shuffled_sub_target_new_acc']))
    
    if 'edited_long_context_target_new_acc' in js:
        ret["long_acc"] = (js['edited_long_context_target_new_acc'][0], 1)
    
    return ret

def compare_keys_for_case(original_keys, rephrase_keys, shuffled_keys, long_keys, collected_results, idx=20):
    # prepare data
    keys = [original_keys[idx], rephrase_keys[idx], shuffled_keys[idx], long_keys[idx]]
    torch_keys = torch.vstack(keys).cpu().numpy()
    labels = [collected_results[idx]['requested_rewrite']['subject']] + \
             collected_results[idx]['pre']['rephrase_subject'] + \
             collected_results[idx]['pre']['shuffled_subject'] + \
             ["long_context"]
    pca = PCA(n_components=2)
    keys_pca = pca.fit_transform(torch_keys)

    plt.figure(figsize=(20, 16))
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 40
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 55
    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40

    colors = ['red'] + ['blue'] * len(collected_results[idx]['pre']['rephrase_subject']) + \
            ['green'] * len(collected_results[idx]['pre']['shuffled_subject']) + ['purple']
    markers = ['o', 'o', 's', 'D']

    texts = []
    for i, (label, color, marker) in enumerate(zip(labels, colors, [markers[i//11] for i in range(len(labels))])):
        plt.scatter(keys_pca[i, 0], keys_pca[i, 1], c=color, marker=marker, s=450, edgecolors='black', linewidth=2.0, alpha=0.8)
        texts.append(plt.text(keys_pca[i, 0], keys_pca[i, 1], 
                            f'{label}', 
                            fontsize=30, 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)))
        
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), expand_points=(1.2, 1.2), force_points=(0.1, 0.1))

    plt.title('PCA visualization of keys for same subject', fontsize=44, pad=20)
    plt.xlabel('PCA feature 1', fontsize=40, labelpad=20)
    plt.ylabel('PCA feature 2', fontsize=40, labelpad=20)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=20, label=l, markeredgecolor='black', linewidth=2)
            for c, l in zip(['red', 'blue', 'green', 'purple'], 
                            ['Original', 'Rephrase', 'Shuffle', 'Long Context'])]
    plt.legend(handles=handles, fontsize=35, loc="lower left", framealpha=0.8)

    plt.tight_layout()
    
    plt.savefig(f"plots/pca_{idx}_subject.png", dpi=300, bbox_inches='tight')
    plt.close()


def compare_keys_for_pair(pre_original_keys, pre_rephrase_keys, pre_shuffled_keys, pre_long_keys, post_original_keys, post_rephrase_keys, post_shuffled_keys, post_long_keys, collected_results, idx=40):
    # prepare data

    keys = [pre_original_keys[idx], pre_rephrase_keys[idx], pre_shuffled_keys[idx], pre_long_keys[idx], post_original_keys[idx], post_rephrase_keys[idx], post_shuffled_keys[idx], post_long_keys[idx]]
    torch_keys = torch.vstack(keys).cpu().numpy()
    labels = [collected_results[idx]['requested_rewrite']['subject']] + \
             collected_results[idx]['pre']['rephrase_subject'] + \
             collected_results[idx]['pre']['shuffled_subject'] + \
             ["long_context"]
    labels = labels * 2
    pca = PCA(n_components=2)
    keys_pca = pca.fit_transform(torch_keys)

    plt.figure(figsize=(20, 16))
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 40
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 55
    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40

    colors = ['red'] + ['blue'] * len(collected_results[idx]['pre']['rephrase_subject']) + \
             ['green'] * len(collected_results[idx]['pre']['shuffled_subject']) + ['purple']
    colors = colors * 2
    nums = int(len(labels)/2)
    markers = ['o'] * nums + ['*'] * nums
    sizes = [450] * nums + [600] * nums

    texts = []
    for i, (label, color, marker, size) in enumerate(zip(labels, colors, markers, sizes)):
        plt.scatter(keys_pca[i, 0], keys_pca[i, 1], 
                   c=color, marker=marker, s=size, 
                   edgecolors='black', linewidth=2.0, alpha=0.8)

    plt.title('PCA visualization of keys for same subject', fontsize=44, pad=20)
    plt.xlabel('PCA feature 1', fontsize=40, labelpad=20)
    plt.ylabel('PCA feature 2', fontsize=40, labelpad=20)

    handles = []
    for c, l in zip(['red', 'blue', 'green', 'purple'],
                    ['Original', 'Rephrase', 'Shuffle', 'Long Context']):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                                markersize=20, label=l, markeredgecolor='black', linewidth=2))
    handles.extend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=20, label='Pre-editing', markeredgecolor='black', linewidth=2),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                  markersize=25, label='Post-editing', markeredgecolor='black', linewidth=2)
    ])

    plt.legend(handles=handles, fontsize=35, loc="lower left", framealpha=0.8)
    plt.tight_layout()
    
    plt.savefig(f"plots/pca_{idx}_subject_agg.png", dpi=300, bbox_inches='tight')
    plt.close()

def load_keys(results_dir, post=False):
    collected_outputs = torch.load(f"{results_dir}/tr0.9_50000/metrics/{alg}_collect_seed0.pt")
    with open(f"{results_dir}/tr0.9_50000/metrics/{alg}_results_seed0.json", "r") as f:
        collected_results = json.load(f)
    
    key_ori = "post" if post else "pre"
    original_keys = []
    rephrase_keys = []
    shuffled_keys = []
    long_keys = []
    for i in range(len(collected_outputs[key_ori])):
        if not isinstance(collected_outputs[key_ori][i]['ks']['shuffled'], list):
            original_keys.append(collected_outputs[key_ori][i]['ks']['original'].squeeze(1))
            rephrase_keys.append(collected_outputs[key_ori][i]['ks']['rephrase'].squeeze())
            shuffled_keys.append(collected_outputs[key_ori][i]['ks']['shuffled'].squeeze())
            long_keys.append(collected_outputs[key_ori][i]['ks']['long'].squeeze())
        else:
            del collected_results[i]

    return original_keys, rephrase_keys, shuffled_keys, long_keys, collected_results

def main():
   
    original_fail_cases= collect_eval_results(f"{results_dir}/baselines/metrics/{alg}_results_seed0.json")
    rep_fail_cases = collect_eval_results(f"{results_dir}/tr0.9_50000/metrics/{alg}_results_seed0.json")
    
    # case key
    original_keys, rephrase_keys, shuffled_keys, long_keys, collected_results = load_keys(results_dir)

    post_original_keys, post_rephrase_keys, post_shuffled_keys, post_long_keys, _ = load_keys(results_dir, post=True)

    original_rephrase_keys_sim = []
    original_shuffle_keys_sim = []
    original_long_keys_sim = []
    post_original_rephrase_keys_sim = []
    post_original_shuffle_keys_sim = []
    post_original_long_keys_sim = []
    original_original_keys_sim = []
    original_random_keys_sim = []
    post_original_random_keys_sim = []
    for i in range(len(original_keys)):
        print(i)
        original_rephrase_sim, original_shuffle_sim, original_long_sim, original_random_sim, post_original_rephrase_sim, post_original_shuffle_sim, post_original_long_sim, post_original_random_sim, original_original_sim  = compare_keys_qualitive(original_keys, rephrase_keys, shuffled_keys, long_keys, post_original_keys, post_rephrase_keys, post_shuffled_keys, post_long_keys, collected_results, idx=i)
        original_rephrase_keys_sim.append(original_rephrase_sim)
        original_shuffle_keys_sim.append(original_shuffle_sim)
        original_long_keys_sim.append(original_long_sim)
        post_original_rephrase_keys_sim.append(post_original_rephrase_sim)
        post_original_shuffle_keys_sim.append(post_original_shuffle_sim)
        post_original_long_keys_sim.append(post_original_long_sim)
        original_random_keys_sim.append(original_random_sim)
        post_original_random_keys_sim.append(post_original_random_sim)
        original_original_keys_sim.append(original_original_sim)
    print("Original Rephrase Sim: ", sum(original_rephrase_keys_sim)/ len(original_rephrase_keys_sim))
    print("Original Shuffle Sim: ", sum(original_shuffle_keys_sim)/ len(original_shuffle_keys_sim))
    print("Original Long Sim: ", sum(original_long_keys_sim)/ len(original_long_keys_sim))
    print("Original Original Sim: ", sum(original_original_keys_sim) / len(original_original_keys_sim))
    print("Original Random Sim: ", sum(original_random_keys_sim) / len(original_random_keys_sim))
    random_mean = np.mean(original_random_keys_sim)
    plt.figure(figsize=(12, 8))

    with open("plots/keys_sim.json", "w") as f:
        json.dump({"original_rephrase_keys_sim": original_rephrase_keys_sim, "original_shuffle_keys_sim": original_shuffle_keys_sim, "original_long_keys_sim": original_long_keys_sim, "post_original_rephrase_keys_sim": post_original_rephrase_keys_sim, "post_original_shuffle_keys_sim": post_original_shuffle_keys_sim, "post_original_long_keys_sim": post_original_long_keys_sim, "original_original_keys": original_original_keys_sim, "original_random_keys_sim": original_random_keys_sim, "post_original_random_keys": post_original_random_keys_sim}, f)
    # 定义 bin 边缘
    bin_edges = np.arange(-0.05, 1.05, 0.1)


    plt.hist(original_rephrase_keys_sim, bins=bin_edges, alpha=0.7, label='Rephrased Keys', color='blue')
    plt.hist(original_shuffle_keys_sim, bins=bin_edges, alpha=0.7, label='Shuffled Keys', color='green')
    plt.hist(original_long_keys_sim, bins=bin_edges, alpha=0.7, label='Long Keys', color='red')

    plt.axvline(random_mean, color='grey', linestyle='dashed', linewidth=2, label=f'Random Keys Mean: {random_mean:.2f}')


    plt.xlabel('Whiten Dot Product', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.title('Distribution of Whiten Dot Product', fontsize=24)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=18)

    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    plt.show()
    plt.savefig("plots/post_distribution_keys.pdf")
    # for i in rep_fail_cases + [0,1,2,3]:
    #     print(f"Case{i}: ")
    #     compare_keys_for_case(original_keys, rephrase_keys, shuffled_keys, long_keys, collected_results, idx=i)
    #     compare_keys_for_pair(original_keys, rephrase_keys, shuffled_keys, long_keys, post_original_keys, post_rephrase_keys, post_shuffled_keys, post_long_keys, collected_results, idx=i)
    #     compare_keys_qualitive(original_keys, rephrase_keys, shuffled_keys, long_keys, post_original_keys, post_rephrase_keys, post_shuffled_keys, post_long_keys, collected_results, idx=i)


if __name__ == "__main__":
    main()
    