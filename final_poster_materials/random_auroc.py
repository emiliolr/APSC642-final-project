import numpy as np
from sklearn.metrics import roc_auc_score

# Calculating AUROC w/randomly assigned score (=reconstruction errors)
#   - repeating 1000 times to smooth over random variation
num_inlier_samples = 6000
num_outlier_samples = 700

auroc_scores = []
for i in range(1000):
    random_recon_error = np.random.uniform(0, 5000, size = num_inlier_samples + num_outlier_samples)
    labels_neg_pos = ([1] * num_inlier_samples) + ([0] * num_outlier_samples)
    labels_neg_pos = np.array(labels_neg_pos)

    random_auroc = roc_auc_score(labels_neg_pos, random_recon_error)
    auroc_scores.append(random_auroc)

mean_auroc = np.mean(auroc_scores)
print(f'AUROC from random score assignment: {round(mean_auroc, 2)}')
