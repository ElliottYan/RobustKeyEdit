import json
import numpy as np
import matplotlib.pyplot as plt

with open("plots/keys_sim.json", "r") as f:
    keys = json.load(f)

original_rephrase_keys_sim = keys["original_rephrase_keys_sim"]
original_shuffle_keys_sim = keys["original_shuffle_keys_sim"]
original_long_keys_sim = keys["original_long_keys_sim"]
post_original_rephrase_keys_sim = keys["post_original_rephrase_keys_sim"]
post_original_shuffle_keys_sim = keys["post_original_shuffle_keys_sim"]
post_original_long_keys_sim = keys["post_original_long_keys_sim"]

# Calculate and print mean and standard deviation for each similarity list
def calculate_stats(data, label):
    mean = np.mean(data)
    std = np.std(data)
    print(f"{label} - {mean:.2f} ± {std:.2f}")
breakpoint()
calculate_stats(original_rephrase_keys_sim, "Original Rephrase Keys")
calculate_stats(post_original_rephrase_keys_sim, "Post Original Rephrase Keys")
calculate_stats(original_shuffle_keys_sim, "Original Shuffle Keys")
calculate_stats(post_original_shuffle_keys_sim, "Post Original Shuffle Keys")
calculate_stats(original_long_keys_sim, "Original Long Keys")
calculate_stats(post_original_long_keys_sim, "Post Original Long Keys")

# Correct calculation of overall mean and standard deviation
all_original = np.concatenate([original_rephrase_keys_sim, original_shuffle_keys_sim, original_long_keys_sim])
all_post = np.concatenate([post_original_rephrase_keys_sim, post_original_shuffle_keys_sim, post_original_long_keys_sim])

def calculate_overall_stats(data, label):
    mean = np.mean(data)
    std = np.std(data)
    print(f"{label} - Overall Mean: {mean:.2f} ± {std:.2f}")

calculate_overall_stats(all_original, "All Original Keys")
calculate_overall_stats(all_post, "All Post-Edited Keys")

bin_edges = np.arange(-0.05, 1.05, 0.1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Subplot for Rephrased Keys
axes[0].hist(original_rephrase_keys_sim, bins=bin_edges, alpha=0.7, label='Original', color='blue')
axes[0].hist(post_original_rephrase_keys_sim, bins=bin_edges, alpha=0.7, label='Edited', color='orange')
axes[0].set_title('Rephrased Keys', fontsize=20)
axes[0].set_xlabel('Whiten Dot Product', fontsize=16)
axes[0].set_ylabel('Frequency', fontsize=16)
axes[0].legend(fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Subplot for Shuffled Keys
axes[1].hist(original_shuffle_keys_sim, bins=bin_edges, alpha=0.7, label='Original', color='blue')
axes[1].hist(post_original_shuffle_keys_sim, bins=bin_edges, alpha=0.7, label='Edited', color='orange')
axes[1].set_title('Shuffled Keys', fontsize=20)
axes[1].set_xlabel('Whiten Dot Product', fontsize=16)
axes[1].legend(fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.6)

# Subplot for Long Keys
axes[2].hist(original_long_keys_sim, bins=bin_edges, alpha=0.7, label='Original', color='blue')
axes[2].hist(post_original_long_keys_sim, bins=bin_edges, alpha=0.7, label='Edited', color='orange')
axes[2].set_title('Long Keys', fontsize=20)
axes[2].set_xlabel('Whiten Dot Product', fontsize=16)
axes[2].legend(fontsize=14)
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("plots/comparison_distribution_keys.pdf")
plt.show()