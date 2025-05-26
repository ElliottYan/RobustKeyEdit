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
alg = sys.argv[1]
metric = sys.argv[2]  # 新增参数，用于选择相似度计算方法

def calculate_similarity(a, b, metric="cosine", inv_cov=None):
    """计算两个向量的相似度."""
    if metric == "cosine":
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    elif metric == 'whiten_cosine':
        assert inv_cov is not None
        left = torch.matmul(a, inv_cov)
        return torch.inner(left.flatten(), b.flatten())
    elif metric == "inner":
        return torch.inner(a.flatten(), b.flatten())
    elif metric == "euclidean":
        return -torch.cdist(a.unsqueeze(0), b.unsqueeze(0)) # 使用负距离表示相似度
    elif metric == "manhattan":
        return -torch.cdist(a.unsqueeze(0), b.unsqueeze(0), p=1)# Manhattan distance，使用负距离表示相似度
    elif metric == "kl_divergence":
        a = F.softmax(a, dim=-1)
        b = F.softmax(b, dim=-1)
        return -torch.sum(a * torch.log(b / a), dim=-1)  # 使用负KL散度表示相似度
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def compare_values(plot_metrics, keys_similarities, metric="cosine"):
    pre_values = torch.load(f"results_llama/fact_structured/single/metrics/{alg}_pre_vss_seed0.pt")
    post_values = torch.load(f"results_llama/fact_structured/single/metrics/{alg}_post_vss_seed0.pt")

    similarities = defaultdict(list)
    golden_similarities = defaultdict(list)

    for pre_item, post_item in zip(pre_values, post_values):
        pre_original_value = pre_item["original"]
        post_original_value = post_item["original"]
        original_gap = post_original_value - pre_original_value
        pre_rephrase_value = pre_item["rephrase"]
        post_rephrase_value = post_item["rephrase"]
        pre_shuffled_value = pre_item["shuffled"]
        post_shuffled_value = post_item["shuffled"]
        pre_long_value = pre_item["long"]
        post_long_value = post_item["long"]
        
        for i, (original_gap_single_layer, golden_single_layer) in enumerate(zip(original_gap, pre_item["original_delta"])):
            golden_similarities[i].append(calculate_similarity(original_gap_single_layer, golden_single_layer, metric).item())

        tmp1, tmp2, tmp3 = defaultdict(list), defaultdict(list), defaultdict(list)
        for pre_sub, post_sub in zip(pre_rephrase_value, post_rephrase_value):
            for i, (pre_sub_single, post_sub_single) in enumerate(zip(pre_sub, post_sub)):
                tmp1[i].append(calculate_similarity(pre_item["original_delta"][i].cpu(), (post_sub_single-pre_sub_single).cpu(), metric).item())
        for pre_sub, post_sub in zip(pre_shuffled_value, post_shuffled_value):
            for i, (pre_sub_single, post_sub_single) in enumerate(zip(pre_sub, post_sub)):
                tmp2[i].append(calculate_similarity(pre_item["original_delta"][i].cpu(), post_sub_single-pre_sub_single.cpu(), metric).item())

        for i, (pre_long_single, post_long_single) in enumerate(zip(pre_long_value, post_long_value)):
            tmp3[i] = calculate_similarity(pre_item["original_delta"][i].cpu(), post_long_single-pre_long_single, metric).item()
        for i in tmp1.keys():
            similarities[i].append({'type': 'rephrase_subject', 'similarity': sum(tmp1[i])/len(tmp1[i])})
            similarities[i].append({'type': 'shuffled_subject', 'similarity': sum(tmp2[i])/len(tmp2[i])})
            similarities[i].append({'type': 'long_context', 'similarity': tmp3[i]})
    
    metrics_df = pd.DataFrame(plot_metrics)
    
    # 计算每一层的相关性
    for layer in similarities.keys():
        df = pd.DataFrame(similarities[layer])
        
        rephrase_df = df[df['type'] == 'rephrase_subject'].reset_index(drop=True)
        shuffle_df = df[df['type'] == 'shuffled_subject'].reset_index(drop=True)
        long_df = df[df['type'] == 'long_context'].reset_index(drop=True)

        keys_df = pd.DataFrame(keys_similarities[layer])
        rephrase_keys_sim = keys_df[keys_df['type'] == 'rephrase_subject'].reset_index(drop=True)
        shuffle_keys_sim = keys_df[keys_df['type'] == 'shuffled_subject'].reset_index(drop=True)
        long_keys_sim = keys_df[keys_df['type'] == 'long_context'].reset_index(drop=True)
     
        # rephrase_keys_sim = pd.DataFrame([item for item in keys_similarities[layer] if item["type"] == "rephrase_subject"])
        # shuffle_keys_sim = pd.DataFrame([item for item in keys_similarities[layer] if item["type"] == 'shuffled_subject'])
        # long_keys_sim =  pd.DataFrame([item for item in keys_similarities[layer] if item["type"] == 'long_context'])
        
        print(f"\nLayer {layer} correlations:")
        
        breakpoint()
        # Value & Performance correlation
        corr_rephrase_value_perf = rephrase_df['similarity'].corr(metrics_df['rephrase_subject_prob_decrease'])
        corr_shuffle_value_perf = shuffle_df['similarity'].corr(metrics_df['shuffled_subject_prob_decrease'])
        corr_long_value_perf = long_df['similarity'].corr(metrics_df['long_context_prob_decrease'])
        print(f"Rephrase Value & Performance: {corr_rephrase_value_perf:.4f}")
        print(f"Shuffle Value & Performance: {corr_shuffle_value_perf:.4f}")
        print(f"Long Context Value & Performance: {corr_long_value_perf:.4f}")
        
        # Value & Key correlation
        corr_rephrase_value_key = rephrase_df['similarity'].corr(rephrase_keys_sim['similarity'])
        corr_shuffle_value_key = shuffle_df['similarity'].corr(shuffle_keys_sim['similarity'])
        corr_long_value_key = long_df['similarity'].corr(long_keys_sim['similarity'])
        print(f"Rephrase Value & Key: {corr_rephrase_value_key:.4f}")
        print(f"Shuffle Value & Key: {corr_shuffle_value_key:.4f}")
        print(f"Long Context Value & Key: {corr_long_value_key:.4f}")
        
        # Key & Performance correlation
        corr_rephrase_key_perf = rephrase_keys_sim['similarity'].corr(metrics_df['rephrase_subject_prob_decrease'])
        corr_shuffle_key_perf = shuffle_keys_sim['similarity'].corr(metrics_df['shuffled_subject_prob_decrease'])
        corr_long_key_perf = long_keys_sim['similarity'].corr(metrics_df['long_context_prob_decrease'])
        print(f"Rephrase Key & Performance: {corr_rephrase_key_perf:.4f}")
        print(f"Shuffle Key & Performance: {corr_shuffle_key_perf:.4f}")
        print(f"Long Context Key & Performance: {corr_long_key_perf:.4f}")
        
        # Golden similarity for this layer
        print(f"Golden Similarity: {np.mean(golden_similarities[layer]):.4f}")
    


def calculate_variance_rep(rephrase, original, all_rephrases, inv_cov=None):
    assert rephrase.shape[1] == 1
    other_rephrases = [rep for rep in all_rephrases if not torch.equal(rep, rephrase)]
    random_rep = random.choice(other_rephrases)[0]

    # vector_dim = rephrase.shape[-1]
    # random_rep = torch.normal(mean=0, std=1, size=(1, 1, vector_dim))

    cluster_data = torch.cat((rephrase.squeeze(), original.squeeze(1)), dim=0)
    cluster_data = np.array(cluster_data)
    
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=1, random_state=0)
    kmeans.fit(cluster_data_scaled)

    all_data = torch.cat((rephrase.squeeze(), original.squeeze(1), random_rep.squeeze(1)), dim=0)
    all_data = np.array(all_data)
    all_data_scaled = scaler.transform(all_data)  # 使用之前拟合的scaler

    # 计算所有点到聚类中心的距离
# <<<<<<< HEAD
    # if metric == "euclidean":
# =======
    if metric == "distance":
# >>>>>>> ff487e4fd269ef2c246ae90dcb072c71837f1bcf
        distances = np.min(kmeans.transform(all_data_scaled), axis=1)
    elif metric == "cosine":
        cluster_center = kmeans.cluster_centers_[0]
        distances = cosine_similarity(all_data_scaled, cluster_center.reshape(1, -1)).flatten()
    elif metric == "inner":
        distances = [np.inner(point, cluster_center) for point in all_data_scaled]
    
    mean = np.mean(distances[:-1])
    variance = np.var(distances[-1])
    baseline = distances[-1]
    return mean, variance, baseline

def calculate_pairwise_variance_rep(rephrase, original, all_rephrases, inv_cov=None):
    assert rephrase.shape[1] == 1
    # vector_dim = rephrase.shape[-1]
    # random_rep = torch.normal(mean=0, std=1, size=(1, 1, vector_dim))
    other_rephrases = [rep for rep in all_rephrases if not torch.equal(rep, rephrase)]
    flat_rephrases = [x for xs in other_rephrases for x in xs]

    random_reps = random.choices(flat_rephrases, k=1+len(rephrase)) # the same size as all_data

    all_data = torch.cat((original.squeeze(1), rephrase.squeeze()), dim=0)
    all_data = np.array(all_data)
    
    # 标准化数据
    # scaler = StandardScaler()
    # all_data_scaled = scaler.fit_transform(all_data)
    
    # random_rep_scaled = scaler.transform(random_rep.squeeze(1).numpy())
    all_data_scaled = all_data
    random_rep_scaled = np.stack([rand.squeeze(1).numpy() for rand in random_reps], axis=0)
    
    if metric == "cosine":
        distance_matrix = cosine_similarity(all_data_scaled) #(11,11)
        random_distances = cosine_similarity(random_rep_scaled, all_data_scaled)[0][1] #[1, 11]
    elif metric == "whiten_cosine":
        left = all_data_scaled @ inv_cov.numpy() # [1, 11]
        right = all_data_scaled
        distance_matrix = np.inner(left, right)
        np.fill_diagonal(distance_matrix, 0)
        # random_distances = np.inner(left, random_rep_scaled)[1][0]
        random_distances = (left[:,None,:] @ random_rep_scaled.transpose(0,2,1))[1,0] # squeeze
    
    elif metric == "euclidean":
        distance_matrix = np.linalg.norm(all_data_scaled[np.newaxis,:] - all_data_scaled[:, np.newaxis, :], axis = -1)
        random_distances = np.linalg.norm(random_rep_scaled - all_data_scaled, axis=1)[0]
    elif metric == "inner":
        distance_matrix = np.inner(all_data_scaled, all_data_scaled)
        np.fill_diagonal(distance_matrix, 0)  
        random_distances = np.inner(random_rep_scaled, all_data_scaled)[0][1]
    else:
        raise ValueError("Unsupported metric. Use 'cosine', 'euclidean', or 'inner'.")

    distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)].tolist()
 
    baseline = random_distances.tolist()
    return distances, None, baseline

def tsne_keys(original_keys, rephrase_keys, method="tsne"):
    keys = []
    original_keys, rephrase_keys = original_keys[:10], rephrase_keys[:10]
    for o, r in zip(original_keys, rephrase_keys):
        keys.append(torch.concat((o.squeeze(0), r.squeeze()), dim=0))
    
    keys = torch.stack(keys, dim=0)
    keys_reshaped = keys.reshape(-1, 11008)
    keys_np = keys_reshaped.cpu().numpy()
    if method == "tsne":
        tsne = TSNE(n_components=2, random_state=42, perplexity=100)
        keys_redu = tsne.fit_transform(keys_np)
    elif method == "pca":
        pca = PCA(n_components=2)
        keys_redu = pca.fit_transform(keys_np)

    num_subjects = len(original_keys)
    colors = sns.color_palette("husl", n_colors=num_subjects)
    color_map = np.repeat(colors, 11, axis=0)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(keys_redu[:, 0], keys_redu[:, 1], c=color_map, s=50, alpha=0.6)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                label=f'Subject {i+1}', 
                                markerfacecolor=colors[i], markersize=10)
                    for i in range(num_subjects)]

    plt.legend(handles=legend_elements, title="Subjects", 
            loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)

    plt.title(f'{method} visualization of keys')
    plt.xlabel(f'{method} feature 1')
    plt.ylabel(f'{method} feature 2')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/{method}_keys_rephrase.pdf")
    # tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    # keys_tsne = tsne.fit_transform(keys)    


def dimension_reduction(original_keys, rephrase_keys, shuffled_keys, long_keys, collected_results, idx=20): #18 51
    # 准备数据
    keys = [original_keys[idx].squeeze(0), rephrase_keys[idx].squeeze(), shuffled_keys[idx].squeeze(), long_keys[idx].squeeze()]
    keys = torch.vstack(keys).cpu().numpy()
    
    labels = [collected_results[idx]['requested_rewrite']['subject']] + \
             collected_results[idx]['pre']['rephrase_subject'] + \
             collected_results[idx]['pre']['shuffled_subject'] + \
             ["long_context"]
    
    pca = PCA(n_components=2)
    keys_pca = pca.fit_transform(keys)
    
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

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                expand_points=(1.2, 1.2), force_points=(0.1, 0.1))

    plt.title('PCA visualization of keys for same subject', fontsize=44, pad=20)
    plt.xlabel('PCA feature 1', fontsize=40, labelpad=20)
    plt.ylabel('PCA feature 2', fontsize=40, labelpad=20)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=20, label=l, markeredgecolor='black', linewidth=2)
            for c, l in zip(['red', 'blue', 'green', 'purple'], 
                            ['Original', 'Rephrase', 'Shuffle', 'Long Context'])]
    plt.legend(handles=handles, fontsize=35, loc="lower left", framealpha=0.8)

    plt.tight_layout()
    
    plt.savefig("plots/pca_1_subject.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def find_unrelated_similar_text(original_keys, inv_cov):
    wiki_keys_data = torch.load("results_llama/fact_edit/rome/metrics/wiki_ks_ROME.pt")

    wiki_keys = []
    for i, sample in enumerate(wiki_keys_data):
        wiki_keys.append(sample['key'][:, :, -1:, :]) #[1,1,11008]
    wiki_keys = torch.stack(wiki_keys, dim=0).squeeze() #(1000, 11008)
   
    original_key = original_keys[0].squeeze() #(1000, 11008) # repeat(wiki_keys.shape[0], 1, 1)
    left = torch.matmul(wiki_keys , inv_cov) # [1000, 11008]
    right = original_key
    similarities = torch.inner(left, right)
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    top_similarities = sorted_similarities[0]
    print("Top Similarity: ", top_similarities)
    index = sorted_indices[0]
    sample = wiki_keys_data[index]
    sample['key'] = sample['key'][:, :, -1, :]
    return sample, top_similarities

    


def compare_keys(plot_metrics, metric="cosine"):
    collected_outputs = torch.load(f"results_llama/fact_edit/llama_final_v2_tr0.9/metrics/{alg}_collect_seed0.pt")
    with open(f"results_llama/fact_edit/llama_final_v2_tr0.9/metrics/{alg}_results_seed0.json", "r") as f:
        collected_results = json.load(f)
    
    
    original_keys = [collected_outputs['pre'][i]['ks']['original'] for i in range(len(collected_outputs['pre']))]
    rephrase_keys = [collected_outputs['pre'][i]['ks']['rephrase'] for i in range(len(collected_outputs['pre']))]
    shuffled_keys = [collected_outputs['pre'][i]['ks']['shuffled'] for i in range(len(collected_outputs['pre']))]
    long_keys = [collected_outputs['pre'][i]['ks']['long'] for i in range(len(collected_outputs['pre']))]

    original_vs = [collected_outputs['pre'][i]['ks']['original'] for i in range(len(collected_outputs['pre']))]
    rephrase_vs = [collected_outputs['pre'][i]['ks']['rephrase'] for i in range(len(collected_outputs['pre']))]
    shuffled_vs = [collected_outputs['pre'][i]['ks']['shuffled'] for i in range(len(collected_outputs['pre']))]
    long_vs = [collected_outputs['pre'][i]['ks']['long'] for i in range(len(collected_outputs['pre']))]

    
    dimension_reduction(original_keys, rephrase_keys, shuffled_keys, long_keys, collected_results)
    
    mean_distances = []
    variance_distances = []
    baseline_distances = []
    unrelated_similar_distances = []
    similarities = defaultdict(list)
    if metric == 'whiten_cosine':
        inv_cov = torch.load('EasyEditOurs/data/stats/_llama2-7b/wikitext_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.inv_conv.pt')
        from functools import partial
    else: inv_cov = None
    
    tsne_keys(original_keys, rephrase_keys)
    calculate_similarity_func = partial(calculate_similarity, inv_cov=inv_cov)

    cnt = 0
    for i, (original_key, rephrase_key, shuffled_key, long_key) in enumerate(zip(original_keys, rephrase_keys, shuffled_keys, long_keys)):
        cnt += 1
        if cnt % 10 == 0: print(f"Processed Cnt: {cnt}")
        tmp1, tmp2, tmp3 = defaultdict(list), defaultdict(list) , defaultdict(list)

        # if metric == 'whiten_cosine':
        #     unrelated_similar_sample, unrelated_similar_similarity = find_unrelated_similar_text(original_key,  inv_cov)
        #     unrelated_similar_distances.append(unrelated_similar_similarity)
        # else:
        #     unrelated_similar_sample, unrelated_similar_similarity = None, None
        # rephrase
        for sub in rephrase_key:
            for i, (ori_single, sub_single) in enumerate(zip(original_key, sub)):
                rephrase_sim = calculate_similarity_func(ori_single, sub_single, metric).item()
                tmp1[i].append(rephrase_sim)
      
        mean, variance, baseline = calculate_pairwise_variance_rep(rephrase_key, original_key, rephrase_keys, inv_cov=inv_cov)
        #mean, variance, baseline = calculate_variance_rep(rephrase_key, original_key, rephrase_keys)
        mean_distances += mean
        baseline_distances += baseline
        print(f'len: {len(mean_distances)}')
        # mean_distances.append(mean)
        # variance_distances.append(variance)
        # baseline_distances.append(baseline)
        # shuffled
        for sub in shuffled_key:
            for i, (ori_single, sub_single) in enumerate(zip(original_key, sub)):

                shuffle_sim = calculate_similarity_func(ori_single, sub_single, metric).item()
                tmp2[i].append(shuffle_sim)
        
        #long
        for i, (ori_single, long_single) in enumerate(zip(original_key, long_key)):
            long_sim = calculate_similarity_func(ori_single, long_single, metric).item()
            tmp3[i] = long_sim       
        
        for i in tmp1.keys():
            similarities[i].append({'type': 'rephrase_subject', 'similarity': sum(tmp1[i])/len(tmp1[i])})
            similarities[i].append({'type': 'shuffled_subject', 'similarity': sum(tmp2[i])/len(tmp2[i])})
            similarities[i].append({'type': 'long_context', 'similarity': tmp3[i]})  
           
        
        metrics_df = pd.DataFrame(plot_metrics)
        
    for layer in similarities.keys():
        df = pd.DataFrame(similarities[layer])
        
       
        for type_ in ['rephrase_subject', 'shuffled_subject', "long_context"]:
            combined_df = pd.concat([
                df[df['type'] == type_].reset_index(drop=True), 
                metrics_df[f'{type_}_acc']
            ], axis=1)
           
            correlation = combined_df['similarity'].corr(combined_df[f'{type_}_acc'])
            print(f"Layer {layer}, {type_} & Performance - Correlation coefficient: {correlation:.4f}")
    
    import pickle
    with open('./tmp.distances.pkl', 'wb') as f:
        pickle.dump(mean_distances, f)
    with open('./tmp.random.pkl', 'wb') as f:
        pickle.dump(baseline_distances, f)
            
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))

    
    ax1.hist(mean_distances, bins=30, edgecolor='black')
    ax1.set_title("Distribution of $k_1C^{-1}k_2$ on\nRephrased Subjects' Keys", fontsize=14)
    ax1.set_xlabel(f'{metric} Mean', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.axvline(x=sum(baseline_distances)/len(baseline_distances), color='r', linestyle='--', linewidth=2, label='Value between Two Random Subjects')
    ax1.axvline(x=sum(unrelated_similar_distances)/len(unrelated_similar_distances), color='g', linestyle='--', linewidth=2, label='Unrelated Text')

    # # 绘制方差距离的直方图
    # ax2.hist(variance_distances, bins=30, edgecolor='black')
    # ax2.set_title("Distribution of Variance Distances on\nRephrased Subjects' Representations", fontsize=14)
    # ax2.set_xlabel('Distance Variance', fontsize=12)
    # ax2.set_ylabel('Frequency', fontsize=12)
    # ax2.tick_params(axis='both', which='major', labelsize=10)
    ax1.legend(loc='best', frameon=True, framealpha=0.7, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.7)
    # ax1.legend()
    # ax2.grid(True, linestyle='--', alpha=0.7)

    # 调整布局并保存
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/rephrase_key_distance_distribution_{metric}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return similarities
    
   

def main():
    input_file = f"results_llama/fact_edit/rome/metrics/{alg}_results_seed0.json"
    with open(input_file, 'r', encoding='utf8') as f:
        jss = json.load(f)

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
            plot_metrics["rephrase_subject_acc"].append(js_ret["rephrase_acc"][0]/js_ret["rephrase_acc"][1])
            plot_metrics["shuffled_subject_acc"].append(js_ret["shuffled_acc"][0]/ js_ret["shuffled_acc"][1])
            plot_metrics["long_context_acc"].append(js_ret["long_acc"][0]/js_ret["long_acc"][1])

    # print all results
    for key in all_metrics:
        print(f"{key}: {all_metrics[key][0]/all_metrics[key][1]*100}%")
    compare_keys(plot_metrics, metric)

def gather_all_metrics_each_item(js):
    ret = {}
    if 'acc' in js:
        ret['acc'] = (sum(js['acc']), len(js['acc']))
    if 'rev' in js:
        ret['rev'] = (sum(js['rev']), len(js['rev']))
    if 'para_attack_rev' in js:
        ret['para_attack_succ'] = (sum(js['para_attack_succ']), len(js['para_attack_succ']))
    if 'para_attack_rev' in js:
        ret['para_attack_rev'] = (sum(js['para_attack_rev']), len(js['para_attack_rev']))
    # if "reverse_success_rate" in js:
    #     ret['edit_acc'] = js['reverse_success_rate'], 1
    # else:
    #     ret['edit_acc'] = None
    if 'locality' in js:
        if isinstance(js['locality']['zsre_acc'], list):
            # assert len(js['locality']['zsre_acc']) == 1
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
  
    # if 'para_lprobs' in js:
    #     ret['less_vs_more_para'] = compute_para_less_vs_more_all(js)
    #     ret['less_all_vs_more_all'] = compute_para_less_all_vs_more_all(js)
    # else:
    #     ret['less_vs_more_para'] = None
    return ret

def plot(metrics):
    df = pd.DataFrame({
        'subject_length': metrics["subject_length"],
        'original_prob_decrease': metrics["prob_decrease"],
        'shuffled_prob_decrease': metrics["shuffled_subject_prob_decrease"],
        'rephrase_prob_decrease': metrics["rephrase_subject_prob_decrease"]
    })

    def group_length(x):
        return 1 * (x // 1)
    
    grouped = df.groupby(df['subject_length'].apply(group_length)).agg({
        'original_prob_decrease': ['mean', 'std'],
        'shuffled_prob_decrease': ['mean', 'std'],
        'rephrase_prob_decrease': ['mean', 'std']
    })

    grouped = grouped.reset_index()

    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    for i, col in enumerate(['original_prob_decrease', 'shuffled_prob_decrease', 'rephrase_prob_decrease']):
        ax.errorbar(grouped['subject_length'], grouped[col]['mean'], 
                    yerr=grouped[col]['std'], label=col.replace('_', ' ').title(), 
                    fmt=f'{markers[i]}-', capsize=4, capthick=1.5, 
                    color=colors[i], ecolor=colors[i], alpha=0.7, 
                    markersize=8, linewidth=2, elinewidth=1.5)

    ax.set_title('Impact of Subject Manipulation on Probability', fontweight='bold', pad=20)
    ax.set_xlabel('Subject Length', fontweight='bold')
    ax.set_ylabel('Average Probability Decrease', fontweight='bold')

    ax.set_xticks(grouped['subject_length'])
    ax.set_xticklabels(grouped['subject_length'], rotation=45, ha='right')
    ax.legend( frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=-0.1, top=1.1)

    plt.tight_layout()

    plt.savefig('probability_decrease_vs_subject_length.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
    # compare_keys()