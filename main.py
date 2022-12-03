import argparse
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from unet import UNet
from utils import mse_loss, get_inlier_outlier_dataset, save_images

def main(args):
    # Get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    print()

    # Instantiate UNet model & optimizer
    model = UNet(n_channels = 3, n_classes = 3, bilinear = True).to(device)
    optimizer = Adam(model.parameters(), lr = args.lr)

    # Get the CIFAR-10 dataset + putting into a dataloader
    print('PREPARING DATASET:')
    dataset = get_inlier_outlier_dataset(inlier_class = args.inlier_class,
                                         num_inlier_samples = args.inlier_samples,
                                         num_outlier_samples = args.outlier_samples)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    print()

    # Train the model for specified number of epochs
    print('TRAINING MODEL:')
    model.train()

    for e in range(args.epochs):
        for X, _ in dataloader:
            X = X.to(device)

            #  reconstruct image using model
            X_recon = model(X)

            #  calculate loss & update model params
            loss = mse_loss(X, X_recon, sigmoid = True, train = True)

            optimizer.zero_grad() # avoid accumulating gradient
            loss.backward() # backpropogate error
            optimizer.step() # update weights using Adam

        print(f'Done with epoch {e}')

    print()

    # Calculate reconstruction error on all images for evaluation
    print('EVALUATING MODEL:')
    model.eval() # turns off batchnorm in UNet

    dataloader = DataLoader(dataset, batch_size = 32, shuffle = False)
    full_dataloader = DataLoader(dataset, batch_size = len(dataset), shuffle = False)

    all_recon_error = []
    all_labels_neg_pos = []
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)

            #  predict on all images & save inlier/outlier reconstructions
            X_recon = model(X)
            if (i == 0) or (i == (len(dataloader) - 1)):
                X_recon_viz = X_recon.detach().cpu()
                X_recon_viz = torch.sigmoid(X_recon_viz)

                save_images(X_recon_viz, filename = f'figures/reconstructions_inlier_class={args.inlier_class}_batch={i}.png')

            #  extract reconstruction losses on each image - these are non-standardized scores!
            recon_error = mse_loss(X, X_recon, train = False).detach().cpu().tolist()
            all_recon_error.extend(recon_error)

            #  extract labels & convert to neg/pos (pos is inlier class)
            y = (y.detach().cpu() == args.inlier_class).numpy().astype(int)
            all_labels_neg_pos.extend(list(y))

    #  calculate AUROC
    auroc = roc_auc_score(all_labels_neg_pos, all_recon_error)
    print(f'AUROC score: {round(auroc, 2)}')

    #  calculate area under PR curve w/inlier & outlier as pos class - code taken from TIAE repo: github.com/wogong/pt-tiae
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(all_labels_neg_pos, all_recon_error)
    pr_auc_norm_inlier = auc(recall_norm, precision_norm)
    print(f'AUPR score w/inlier as positive class: {round(pr_auc_norm_inlier, 2)}')

    inverse_recon_error = [-1 * err for err in all_recon_error]
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(all_labels_neg_pos, inverse_recon_error, pos_label = 0)
    pr_auc_norm_outlier = auc(recall_norm, precision_norm)
    print(f'AUPR score w/outlier as positive class: {round(pr_auc_norm_outlier, 2)}')

    #  save results
    results_df = pd.DataFrame(columns = ['inlier_class', 'inlier_samples', 'outlier_samples',
                                         'auroc', 'aupr_inlier', 'aupr_outlier'])
    results_df.loc[0] = {'inlier_class' : args.inlier_class, 'inlier_samples' : args.inlier_samples,
                         'outlier_samples' : args.outlier_samples, 'auroc' : auroc,
                         'aupr_inlier' : pr_auc_norm_inlier, 'aupr_outlier' : pr_auc_norm_outlier}
    results_df.to_csv(f'results/inlier_class={args.inlier_class}_results.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--inlier_class', type = int, default = 0)
    parser.add_argument('--inlier_samples', type = int, default = 6000)
    parser.add_argument('--outlier_samples', type = int, default = 700)

    args = parser.parse_args()

    main(args)
