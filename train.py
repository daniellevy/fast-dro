"""
Training loop for Digits and ImageNet experiments
"""

import os
import time
import logging
import argparse
import subprocess
import json
import pdb
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, Adagrad, RMSprop
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from datasets import (MNISTandTypedFeatures, ImageNetFeatures,
                      RandomBatchSizeSampler, CustomDistributionSampler)

from robust_losses import (RobustLoss, DualRobustLoss, PrimalDualRobustLoss,
                           MultiLevelRobustLoss, GEOMETRIES)

from utils import aggregate_by_group, average_step, average_step_ema, copy_state

MAX_ALLOWED_LOSS = 2000.  # value of the robust loss above which we kill the run

# ----------------------------- CONFIGURATION ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist_typed_100',
                    help='Dataset to work on')
parser.add_argument('--data_dir', type=str,
                    default='../data')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--subsample_seed', type=int,
                    default=123)
parser.add_argument('--features_mnist', type=str, default='lenet')
parser.add_argument('--seed', type=int, default=0,
                    help='seed for randomness of batches')

# base loss
parser.add_argument('--loss_type', type=str,
                    default='log', choices=('log', 'square'))

# uncertainty set parameters
parser.add_argument('--size', type=float, default=0.1)
parser.add_argument('--reg', type=float, default=0.01)
parser.add_argument('--geometry', type=str, default='cvar',
                    choices=GEOMETRIES)

# optimizer
parser.add_argument('--averaging', type=str, default='none',
                    help='"none" for no averaging, "constant" for standard,' +
                         '"epoch_X" to restart averaging every epoch, default to 1')

parser.add_argument('--algorithm', type=str, default='batch',
                    choices=('batch', 'dual', 'multilevel', 'eml', 'erm',
                             'primaldual'))
parser.add_argument('--doubling_probability', type=float, default=0.5,
                    help='Doubling probability for multi-level batch size')
parser.add_argument('--batch_size_max', type=int, default=-1,
                    help='Maximum batch size for multilevel (-1 is full batch)')

parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float)
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size (-1 is full batch)')

parser.add_argument('--init_scale', type=float, default=1.0)

parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=('sgd', 'adagrad', 'adam', 'rmsprop'))
parser.add_argument('--momentum', type=str, default='0.9')
parser.add_argument('--ada_eps', type=float, default=1e-8)
parser.add_argument('--heavy_ball', default=False, action='store_true')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_eta', type=float, default=0.1,
                    help='step size for Largrange dual variable')
parser.add_argument('--lr_dual', type=float, default=1e-3,
                    help='step size for dual block in primal-dual methods')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping for p in primal-dual methods')

parser.add_argument('--lr_schedule', type=str, default='constant',
                    help='Step size schedule. supporting poly_xx, constant, '
                         'cosine, step_xx_yy')

parser.add_argument('--output_dir', default='../results/test',
                    help='Directory of model for saving all outputs')
parser.add_argument('--log_interval', type=int, default=5,
                    help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=25, type=int,
                    help='Checkpoint save frequency (in epochs)')
parser.add_argument('--save_losses', default=False, type=bool,
                    help='Save individual losses on eval. Will also save output '
                         'in pickle format')
parser.add_argument('--batch_size_eval', default=-1, type=int,
                    help='Batch size for evaluation (-1 for full batch)')
parser.add_argument('--max_batches_eval', default=-1, type=int,
                    help='Number of eval batches (-1 for unlimited)')
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--max_examples_train', default=-1, type=int,
                    help='Number of training examples per epoch (-1 for unlimited)')

args = parser.parse_args()

# -------------------------- OUTPUT AND LOGGING --------------------------------
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Args: %s', args)
hash_cmd = subprocess.run('git rev-parse --short HEAD', shell=True, check=True,
                          stdout=subprocess.PIPE)
git_hash = hash_cmd.stdout.decode('utf-8').strip()
logging.info(f'Git commit: {git_hash}')

config = dict(args.__dict__)
config['_git_commit_'] = git_hash
with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config, f, sort_keys=True, indent=4)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# ----------------------------- DATASET SETUP ----------------------------------
use_bias = True
eval_train = True
if args.dataset.startswith('mnist_typed'):
    n_typed = int(args.dataset[len('mnist_typed_'):])
    dataset_train = MNISTandTypedFeatures(
        args.data_dir, features_name=args.features_mnist, train=True,
        group_by_class=True,
        subsample_to=(-1, n_typed),
        subsample_seed=args.subsample_seed)
    dataset_test = MNISTandTypedFeatures(
        args.data_dir, features_name=args.features_mnist, train=False,
        group_by_class=True,
        subsample_to=(-1, -1))
elif args.dataset.startswith('submnist_'):
    n_mnist, n_typed = map(int, args.dataset.split('_')[1::2])
    dataset_train = MNISTandTypedFeatures(
        args.data_dir, train=True, group_by_class=True,
        subsample_to=(n_mnist, n_typed),
        subsample_seed=args.subsample_seed)
    dataset_test = MNISTandTypedFeatures(
        args.data_dir, train=False, group_by_class=True,
        subsample_to=(-1, -1))
elif args.dataset == 'imagenet':
    dataset_train = ImageNetFeatures(args.data_dir, train=True, subsample_to=-1)
    dataset_test = ImageNetFeatures(args.data_dir, train=False, subsample_to=-1)
else:
    raise ValueError('Unknown dataset %s' % args.dataset)

data_loader_kwargs = dict(num_workers=0, pin_memory=True)
if args.algorithm in ('multilevel'):
    train_sampler = RandomBatchSizeSampler(
        dataset_train, batch_size_min=args.batch_size,
        batch_size_max=args.batch_size_max,
        doubling_probability=args.doubling_probability, replace=False,
        num_batches=None)

    loader_train = DataLoader(dataset_train, batch_sampler=train_sampler,
                              **data_loader_kwargs)
elif args.algorithm in ('batch', 'dual', 'erm'):
    loader_train = DataLoader(dataset_train,
                              batch_size=(args.batch_size if args.batch_size > 0
                                          else len(dataset_train)),
                              shuffle=True, **data_loader_kwargs)
elif args.algorithm == 'primaldual':
    train_sampler = CustomDistributionSampler(
        dataset_train, batch_size=args.batch_size)
    loader_train = DataLoader(dataset_train, batch_sampler=train_sampler,
                              **data_loader_kwargs)
else:
    raise ValueError('Unkown algorithm %s' % args.grad_est)

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

loader_eval_train = DataLoader(dataset_train, batch_size=(
    args.batch_size_eval if args.batch_size_eval > 0 else len(dataset_train)),
                               shuffle=True,
                               **data_loader_kwargs)
loader_eval_test = DataLoader(dataset_test, batch_size=(
    args.batch_size_eval if args.batch_size_eval > 0 else len(dataset_test)),
                              **data_loader_kwargs)

# ------------------------------- MODEL SETUP ----------------------------------
num_classes = dataset_train.num_classes
num_features = dataset_train.num_features

# linear model
model = torch.nn.Linear(num_features, num_classes, bias=use_bias).to(device)
model.weight.data.mul_(args.init_scale)
if use_bias:
    model.bias.data.mul_(args.init_scale)

if args.averaging != 'none':
    avg_model = copy.deepcopy(model)

# ------------------------------- LOSS SETUP -----------------------------------
if args.algorithm == 'dual':
    robust_loss = DualRobustLoss(args.size, args.reg, args.geometry).to(device)
elif args.algorithm == 'batch':
    robust_loss = RobustLoss(args.size, args.reg, args.geometry)
elif args.algorithm == 'erm':
    robust_loss = RobustLoss(0, 0, 'chi-square')
elif args.algorithm == 'multilevel':
    robust_layer = RobustLoss(args.size, args.reg, args.geometry)
    robust_loss = MultiLevelRobustLoss(
        robust_layer, train_sampler.batch_size_pmf, args.batch_size)
elif args.algorithm == 'primaldual':
    assert args.reg == 0.0  # currently not supporting regularization term

    if args.clip < 0:
        clip = None
    else:
        clip = args.clip

    robust_loss = PrimalDualRobustLoss(
        args.size, args.geometry, train_sampler, args.lr_dual, clip=clip)
else:
    raise ValueError('Unknown algorithm %s' % args.algorithm)

if args.size == float('inf'):
    robust_losses_eval = dict(
        robust_loss=RobustLoss(float('inf'), args.reg, args.geometry))
else:
    robust_losses_eval = dict(
        robust_loss=RobustLoss(args.size, 0., args.geometry),
        robust_loss_smoothed=RobustLoss(args.size, args.reg, args.geometry))

# ----------------------------- OPTIMIZER SETUP --------------------------------
param_list = [dict(params=model.weight),
              dict(params=robust_loss.parameters(),
                   weight_decay=0,
                   lr=args.lr_eta)]
if use_bias:
    param_list.append(dict(params=model.bias, weight_decay=0))
if args.optimizer == 'sgd':
    momentum = float(args.momentum)
    optimizer = SGD(param_list,
                    lr=args.lr,
                    momentum=momentum,
                    weight_decay=args.weight_decay,
                    nesterov=(not args.heavy_ball
                              and (momentum > 0)))
elif args.optimizer == 'adam':
    betas = tuple(map(float, args.momentum.split(',')))
    if len(betas) == 1:
        betas = (betas[0], 1 - (1 - betas[0]) / 10)
    optimizer = Adam(param_list, lr=args.lr,
                     betas=betas,
                     eps=args.ada_eps, weight_decay=args.weight_decay)
elif args.optimizer == 'rmsprop':
    optimizer = RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adagrad':
    optimizer = Adagrad(param_list, lr=args.lr, weight_decay=args.weight_decay)

logging.info('Set up optimizer: %s' % optimizer)

if args.lr_schedule == 'constant':
    lr_factor_func = lambda epoch: 1
elif args.lr_schedule.startswith('poly'):
    if args.lr_schedule == 'poly':
        power = 0.5
    else:
        power = float(args.lr_schedule.split('_')[-1])
    lr_factor_func = lambda epoch: (1 + epoch) ** (-power)
elif args.lr_schedule.startswith('step'):
    if args.lr_schedule == 'step':
        times, factor = 4, 0.1  # this will decrease the lr by 0.1, 4 times
    else:
        times, factor = map(float, args.lr_schedule.split('_')[1:])
    lr_factor_func = lambda epoch: factor ** int(epoch / args.epochs *
                                                 (times + 1))
elif args.lr_schedule == 'cosine':
    lr_factor_func = lambda epoch: 0.5 * (
            1 + np.cos(np.pi * epoch / args.epochs))
else:
    raise ValueError('Unknown LR schedule %s' % args.lr_schedule)
lr_scheduler = LambdaLR(optimizer, lr_factor_func)

if args.averaging == 'none':
    pass
elif args.averaging.startswith('epoch'):
    if args.averaging == 'epoch':
        average_every = 'auto'  # reset averaging every time we change LR
    else:
        averaging_reset_nb = int(args.averaging.split('_')[-1])
        average_every = args.epochs // (averaging_reset_nb + 1)
    eta = 0
elif args.averaging.startswith('ema'):
    gamma = float(args.averaging.split('_')[-1])
elif args.averaging.startswith('constant'):
    if args.averaging == 'constant':
        eta = 0
    else:
        eta = float(args.averaging.split('_')[-1])
else:
    raise ValueError('Unknown averaging scheme %s' % args.averaging)

steps_since_averaging = 1


# ----------------------------- TRAIN FUNCTION ---------------------------------
def train(args, model, robust_loss, device, loader,
          optimizer, epoch, avg_model=None):
    global steps_since_averaging

    model.train()

    train_metrics = []
    # note: this is only a prediction with multi-level, unless we explicitly set
    # num_batches
    nb_batches = len(loader)
    nb_seen_examples = 0

    for batch_idx, (x, y, g) in enumerate(loader):
        # TODO: in full batch mode with a large dataset, it will be wasteful
        #  to keep moving x to the gpu
        x, y, g = x.to(device), y.to(device), g.to(device)

        outputs = model(x)
        if args.loss_type == 'log':
            per_example_loss = F.cross_entropy(outputs, y, reduction='none')
            predicted = outputs.max(1)[1]
        else:  # args.loss_type == 'square'
            per_example_loss = (outputs.squeeze() - y) ** 2
            predicted = torch.sign(outputs.squeeze())
        loss = robust_loss(per_example_loss)

        accuracy = predicted.eq(y).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        weight_norm = torch.norm(model.weight.data).item()

        if avg_model is not None:
            if (args.averaging.startswith('constant')
                    or args.averaging.startswith('epoch')):
                average_step(model, avg_model,
                             steps_since_averaging + 1, eta=eta)
            elif args.averaging.startswith('ema'):
                if batch_idx == 0:
                    avg_model.load_state_dict(model.state_dict())
                else:
                    average_step_ema(model, avg_model, gamma=gamma)
            steps_since_averaging += 1

        train_metrics.append(dict(
            epoch=epoch,
            average_loss=per_example_loss.mean().item(),
            loss=loss.item(),
            accuracy=accuracy.item(),
            batch_size=x.shape[0],
            weight_norm=weight_norm
        ))

        if batch_idx % args.log_interval == 0:
            logger.info(f"Epoch {epoch}: {batch_idx}/{nb_batches}; "
                        f"Robust loss={loss.item():.4f}, "
                        f"Accuracy={100 * accuracy.item():.2f}%, "
                        f"||w||={weight_norm:.3f}")

        nb_seen_examples += x.shape[0]
        if 0 < args.max_examples_train < nb_seen_examples:
            break
    return train_metrics


# ------------------------------ EVAL FUNCTION ---------------------------------
def eval(epoch, model, robust_losses, device, set_name, loader):
    group_names = loader.dataset.group_names
    num_groups = len(group_names)

    loss_per_g = np.zeros(num_groups)
    total_per_g = np.zeros(num_groups)
    correct_per_g = np.zeros(num_groups)

    model.eval()
    individual_losses = []

    for batch_idx, (x, y, g) in enumerate(loader):
        if batch_idx == args.max_batches_eval:
            break

        x, y, g = x.to(device), y.cpu(), g.cpu().numpy().squeeze()

        with torch.no_grad():
            scores = model(x).cpu()

            if args.loss_type == 'log':
                pred = scores.max(1, keepdim=False)[1]
                batch_loss = F.cross_entropy(scores, y, reduction='none')
            else:  # args.loss_type == 'square'
                scores = scores.squeeze()
                pred = torch.sign(scores)
                batch_loss = (scores - y) ** 2
        individual_losses.append(batch_loss)

        batch_correct = pred.eq(y.view_as(pred)).float().squeeze()

        total_per_g += aggregate_by_group(
            np.ones_like(batch_loss.numpy()), g, num_groups)
        loss_per_g += aggregate_by_group(
            batch_loss.numpy(), g, num_groups)
        correct_per_g += aggregate_by_group(
            batch_correct.numpy(), g, num_groups)
    individual_losses = torch.cat(individual_losses)

    av_loss_per_g = loss_per_g / total_per_g
    accuracy_per_g = correct_per_g / total_per_g
    # av_loss = loss_per_g.sum() / total_per_g.sum()
    av_accuracy_per_g = np.nanmean(accuracy_per_g)
    accuracy = correct_per_g.sum() / total_per_g.sum()
    worst_group_accuracy = np.min(accuracy_per_g)
    worst_group_id = np.argmin(accuracy_per_g)
    worst_group_name = group_names[worst_group_id]

    eval_data = {name: loss(individual_losses).cpu().item() for
                 name, loss in robust_losses.items()}
    eval_data.update({'av_loss/' + group_names[i]: av_loss_per_g[i]
                      for i in range(num_groups)})
    eval_data.update({'accuracy/' + group_names[i]: accuracy_per_g[i]
                      for i in range(num_groups)})
    eval_data.update({'elements/' + group_names[i]: total_per_g[i]
                      for i in range(num_groups)})
    if args.save_losses:
        eval_data['individual_losses'] = individual_losses.cpu().numpy()

    log_str = 'EPOCH {} {:5s}'.format(epoch, set_name.upper())
    log_str += ' Acc: {:.2f}%,'.format(accuracy * 100)
    log_str += ' Av group acc: {:.2f}%,'.format(av_accuracy_per_g * 100)
    log_str += ' Worst group ({}) acc: {:.2f}%,'.format(
        worst_group_name, worst_group_accuracy * 100)
    log_str += ' Robust loss: {:.4f}'.format(eval_data['robust_loss'])

    logging.info(log_str)
    eval_data = {set_name + '/' + k: v for k, v in eval_data.items()}
    eval_data['weight_norm'] = torch.norm(model.weight.data).item()

    return eval_data


# -------------------------------- MAIN LOOP -----------------------------------
def main():
    global steps_since_averaging

    # clocktime
    time_spent_training = 0.
    time_spent_evaluating = 0.

    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    for epoch in range(1, args.epochs + 1):

        logging.info(120 * '=')
        logging.info('Starting epoch %d / %d' % (epoch, args.epochs))
        logging.info('Param group LR''s = %s' % (
            [g['lr'] for g in optimizer.param_groups],))
        logging.info(f'{steps_since_averaging} steps since averaging')

        if args.averaging.startswith('epoch'):
            if average_every == 'auto':
                cond = lr_factor_func(epoch - 1) != lr_factor_func(epoch - 2)
            else:
                cond = (epoch - 1) % average_every == 0
            if cond and epoch > 1:
                steps_since_averaging = 1
                copy_state(avg_model, model)
                logging.info('Resetting averaging now')

        time0 = time.time()
        train_data = train(args, model, robust_loss, device, loader_train,
                           optimizer, epoch,
                           avg_model=None if args.averaging == 'none'
                           else avg_model)
        train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)
        time1 = time.time()
        time_spent_training += time1 - time0

        logging.info(120 * '=')
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            time2 = time.time()
            eval_data = {'epoch': int(epoch),
                         'samples': train_df.batch_size.sum(),
                         'train_clocktime': 0.,
                         'val_clocktime': 0.,
                        }
            if eval_train:
                eval_data.update(
                    eval(epoch,
                         model if args.averaging == 'none' else avg_model,
                         robust_losses_eval, device, 'train',
                         loader_eval_train)
                )
            eval_data.update(
                eval(epoch, model if args.averaging == 'none' else avg_model,
                     robust_losses_eval, device, 'test',
                     loader_eval_test)
            )
            time3 = time.time()
            time_spent_evaluating += time3 - time2

            eval_data.update({
                'train_clocktime': time_spent_training,
                'val_clocktime': time_spent_evaluating,
            })
            eval_df = eval_df.append(
                pd.DataFrame([eval_data], index=[0]), ignore_index=True)

            if not args.save_losses:
                train_df.to_csv(os.path.join(output_dir, 'stats_train.csv'))
                eval_df.to_csv(os.path.join(output_dir, 'stats_eval.csv'))
            else:
                train_df.to_pickle(
                    os.path.join(output_dir, 'stats_train.pickle'))
                eval_df.to_pickle(os.path.join(output_dir, 'stats_eval.pickle'))

            if np.isnan(eval_data['weight_norm']):
                logging.error(
                    'Detected NaN value in weights; breaking training')
                break

            loss_value = (eval_data['test/robust_loss']
                          + 0.5 * args.weight_decay
                          * np.square(eval_data['weight_norm']))

            if np.isnan(loss_value) or loss_value > MAX_ALLOWED_LOSS:
                logging.error(
                    'Loss exceeded threshold value; breaking training')
                break

        lr_scheduler.step()

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(dict(num_classes=num_classes,
                            state_dict=model.state_dict()),
                       os.path.join(output_dir,
                                    'checkpoint-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(output_dir,
                                    'opt-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
