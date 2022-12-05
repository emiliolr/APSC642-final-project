import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Read in the data and prepare for plotting
base_path = '/Users/emiliolr/Desktop/APSC 642/final_project/final_results'

#  w/skip connections
all_results_w_skips = [pd.read_csv(os.path.join(base_path, f'results_w_skips/inlier_class={i}_results.csv')) for i in range(10)]
all_results_w_skips = pd.concat(all_results_w_skips, ignore_index = True)
print(f'Average AUROC w/skip connections: {all_results_w_skips["auroc"].mean()}')

#  w/o skip connections
all_results_wo_skips = [pd.read_csv(os.path.join(base_path, f'results_wo_skips/inlier_class={i}_results.csv')) for i in range(10)]
all_results_wo_skips = pd.concat(all_results_wo_skips, ignore_index = True)
print(f'Average AUROC w/o skip connections: {all_results_wo_skips["auroc"].mean()}')

# (1) Plotting AUROC scores for both methods
all_results_w_skips_AUROC = all_results_w_skips[['inlier_class', 'auroc']].copy()
all_results_wo_skips_AUROC = all_results_wo_skips[['inlier_class', 'auroc']].copy()

all_results_AUROC = pd.concat([all_results_w_skips_AUROC, all_results_wo_skips_AUROC])
all_results_AUROC['skip_connections'] = (['skips'] * 10) + (['no skips'] * 10)

plt.figure(figsize = (8, 6))

plt.axhline(y = 0.5, linestyle = 'dotted', color = 'black', zorder = 1)
sns.barplot(data = all_results_AUROC, x = 'inlier_class', y = 'auroc',
            hue = 'skip_connections', alpha = 0.85, linewidth = 1, edgecolor = 'black',
            zorder = 5)
plt.legend(title = '')

plt.xlabel('Inlier Class', fontweight = 'bold')
plt.ylabel('AUROC', fontweight = 'bold')

plt.savefig('figures/auroc_scores.png', dpi = 600)
plt.show()

# (2) Plotting distribution of reconstruction errors
