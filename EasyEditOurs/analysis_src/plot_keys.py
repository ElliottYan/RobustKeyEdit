import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


# metrics = "whiten"
with open('./tmp.distances.pkl', 'rb') as f:
    mean_distances = pickle.load(f)
with open('./tmp.random.pkl', 'rb') as f:
    baseline_distances = pickle.load(f)

# Set the style for Seaborn
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the histogram using Seaborn
color = "steelblue"  # A muted blue

# Plot the histogram using Seaborn
sns.histplot(mean_distances, bins=20, kde=True, ax=ax)
# sns.histplot(mean_distances, bins=30, kde=True, color=color, edgecolor='black', ax=ax)

# Set the title and labels
ax.set_title("Distribution of $k_1C^{-1}k_2$ on Rephrased Keys", fontsize=22)

ax.set_xlabel(f'Whiten Dot Product', fontsize=20)
ax.set_ylabel('Frequency', fontsize=20)

# Add the vertical line for the baseline
baseline_mean = np.mean(baseline_distances)
ax.axvline(x=baseline_mean, color='r', linestyle='--', linewidth=2, label='Mean Value between Two Random Subjects')

# Customize the legend
ax.legend(loc='upper right', frameon=True, framealpha=0.7, fontsize=18)

# Adjust the tick parameters
ax.tick_params(axis='both', which='major', labelsize=18)

# Add a text annotation for the baseline value
ax.text(baseline_mean, ax.get_ylim()[1]/2, f'{baseline_mean:.2f}', 
        horizontalalignment='center', verticalalignment='bottom', 
        color='r', fontsize=18)

# Adjust layout and save
plt.tight_layout()
plt.savefig(f"plots/rephrase_key_distance_distribution_whiten_dot_product.pdf", dpi=300, bbox_inches='tight')
# plt.show()
