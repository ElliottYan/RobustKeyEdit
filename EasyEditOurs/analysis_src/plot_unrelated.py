import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
import seaborn as sns

# Set the style for Seaborn
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

fn = "../unrelated/unrelated_collect.top10.pkl"
with open(fn, 'rb') as f:
    data = pickle.load(f)

all_values = [value for key_data in data.values() for value, _ in key_data]
fig, ax = plt.subplots(figsize=(10, 8))


sns.histplot(all_values, bins=20, kde=True)

# Adjust the tick parameters
ax.tick_params(axis='both', which='major', labelsize=18)

plt.title('Distribution of Top 10 Unrelated Keys', fontsize=22)
plt.xlabel('Whitening Similarity', fontsize=18)
plt.ylabel('Frequency', fontsize=18)

baseline_mean = np.mean(all_values)
ax.axvline(x=baseline_mean, color='r', linestyle='--', linewidth=2, label='Mean Similarity among Top-10 Unrelated Keys')

ax.text(baseline_mean, ax.get_ylim()[1]/2, f'{baseline_mean:.2f}', 
        horizontalalignment='center', verticalalignment='bottom', 
        color='r', fontsize=18)

plt.tight_layout()
plt.savefig('../plots/unrelated_keys.pdf', dpi=300, bbox_inches='tight')