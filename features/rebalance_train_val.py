"""
Utils to rebalance train and val set from the mnist or typed dataset, i.e.
remake a train and val sets with the desired sizes
"""

import os
import argparse

import torch

parser = argparse.ArgumentParser(
    description='Take train/val.pt and rebalances it')

parser.add_argument('--data_dir', type=str,
                    help='directory where datasets are located')
parser.add_argument('--val_per_class', type=int, default=200,
                   help='number of validation data points per class')
parser.add_argument('--suffix', default='rebalanced', type=str,
                    help='suffix to add to the new pt files')

args = parser.parse_args()

assert len(args.suffix) > 0

os.chdir(args.data_dir)

x = torch.load('train.pt')
y = torch.load('val.pt')

features_cat = torch.cat([x['features'], y['features']])
targets_cat = torch.cat([x['targets'], y['targets']])
predictions_cat = torch.cat([x['predictions'], y['predictions']])

val_idx = torch.cat(
    [torch.where(targets_cat == j)[0][:args.val_per_class] for j in range(10)]
)

train_idx = torch.cat(
    [torch.where(targets_cat == j)[0][args.val_per_class:] for j in range(10)]
)

assert torch.all(torch.cat([val_idx, train_idx]).sort().values 
            == torch.arange(len(targets_cat))) # making sure we have all idx

x_new = {
    'features': features_cat[train_idx],
    'targets': targets_cat[train_idx],
    'predictions': predictions_cat[train_idx]
}

y_new = {
    'features': features_cat[val_idx],
    'targets': targets_cat[val_idx],
    'predictions': predictions_cat[val_idx],
}

reindexing = {
    'train': train_idx,
    'val': val_idx,
}

torch.save(x_new, 'train_%s.pt' % args.suffix)
torch.save(y_new, 'val_%s.pt' % args.suffix)
torch.save(reindexing, 'index_%s.pt' % args.suffix)
