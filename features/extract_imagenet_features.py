"""
Extracts features for the ImageNet dataset provided by torchvision using the
pre-trained resnet specified in `resnet.py`
"""

import logging
import os

import argparse

import numpy as np

from torchvision import transforms
from imagenet_dataset import ImageNet
from resnet import resnet50

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset

import time

import pickle

import subprocess

import pdb

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Apply pretrained network on ImageNet dataset')

parser.add_argument('output_dir', type=str,
                    help='directory where datasets are located')
parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for training')
parser.add_argument('--data_dir', default='data/imagenet', type=str,
                    help='directory where datasets are located')
parser.add_argument('--model', default='resnet50', type=str,
                    help='name of the model')
parser.add_argument('--num_workers', type=int, default=30,
                    help='Number of workers for data loading')
parser.add_argument('--device', type=str, default='cuda',
                    choices=('cpu', 'cuda'),
                    help='Where to do the computation')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Whether to overwrite existing output files')


# parse args, etc.
args = parser.parse_args()
batch_size = args.batch_size

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir,
                                         'feature_extraction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('ImageNet feature extraction')
logging.info('Args: %s', args)

hash_cmd = subprocess.run('git rev-parse --short HEAD', shell=True, check=True,
    stdout=subprocess.PIPE)
git_hash = hash_cmd.stdout.decode('utf-8').strip()
logging.info(f'Git commit: {git_hash}')

# get model
if args.model == 'resnet50':
    model = resnet50(pretrained=True).to(args.device)
else:
    raise ValueError('Unkown model %s' % args.model)
if args.device == 'cuda':
    model = DataParallel(model)

# get input transform
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])


def extract_features(split='val'):
    # set output file names
    output_path = os.path.join(args.output_dir, split + '.pt')
    logging.info(f'Saving {split} set features to {output_path}')
    if not args.overwrite and os.path.exists(output_path):
        logging.info('Output file exists, skipping!')
        return

    # create dataset and dataloader objects
    dataset = ImageNet(args.data_dir, split=split, transform=transform)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # run validations loop
    model.eval()
    features = []
    predictions = []
    count = correct = 0
    start_time = time.time()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            scores, batch_features = model(x)
            batch_predictions = scores.argmax(1)
            batch_correct = (batch_predictions == y).float().sum()

        count += scores.shape[0]
        correct += batch_correct.item()
        features.append(batch_features.cpu().numpy())
        predictions.append(batch_predictions.cpu().numpy())

        elapsed = time.time() - start_time

        logging.info('Processed %d/%d images (%4.1f%%), %-4.0f images/sec, '
                     'Top-1 accuracy = %.3g%%' %
                     (count, len(dataset), 100 * count / len(dataset),
                      scores.shape[0] / elapsed, 100 * correct / count))
        start_time = time.time()

    # save output to file
    logging.info('Saving features to file')
    features = np.concatenate(features, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    torch.save(dict(features=torch.Tensor(features),
                    predictions=torch.Tensor(predictions),
                    targets=torch.LongTensor(dataset.targets)),
               output_path)


extract_features('val')
extract_features('train')
