"""
Trains a simple ConvNet on MNIST and extract features from MNIST and the
typed dataset
"""

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np
import argparse
import logging
import os
import subprocess
import pdb

parser = argparse.ArgumentParser(
    description='Train and extract features from MNIST and Typed Digits')

parser.add_argument('--output_name', type=str, default='lenet',
                    help='name of features output directory')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--data_root', default='data/', type=str,
                    help='directory where datasets are located')
parser.add_argument('--device', type=str, default='cpu',
                    choices=('cpu', 'cuda'),
                    help='Where to do the computation')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Whether to overwrite existing output files')

# parse args, etc.
args = parser.parse_args()
batch_size = args.batch_size
device = args.device

mnist_output_dir = os.path.join(args.data_root,
                                'MNIST', args.output_name + '_features')
typed_output_dir = os.path.join(args.data_root,
                                'typed_digits', args.output_name + '_features')
for directory in (mnist_output_dir, typed_output_dir):
    if not os.path.exists(directory):
        os.makedirs(directory)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(mnist_output_dir,
                                         'feature_extraction.log')),
        logging.FileHandler(os.path.join(typed_output_dir,
                                         'feature_extraction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Args: %s', args)

hash_cmd = subprocess.run('git rev-parse --short HEAD', shell=True, check=True,
    stdout=subprocess.PIPE)
git_hash = hash_cmd.stdout.decode('utf-8').strip()
logging.info(f'Git commit: {git_hash}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        z = F.relu(self.fc1(x))
        
        x = self.fc2(z)
        return F.log_softmax(x, dim=1), z
    
model = Net()

print(model)

model_save_path = os.path.join(mnist_output_dir, 'model.pt')
if not args.overwrite and os.path.exists(model_save_path):
    logging.info('Loading trained model from %s' % model_save_path)
    checkpoint = torch.load(model_save_path)
    print(model_save_path)
    model.load_state_dict(checkpoint)
    train_model = False
else:
    train_model = True
model.to(device)


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_root, train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])),
    batch_size=args.batch_size, shuffle=True)

# setting up objectives
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(args, model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    running_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, z = model(data)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    logging.info('\nTrain set: Average loss: %.4g, Accuracy: %.4g' % (running_loss / len(train_loader.dataset), correct * 100. / len(train_loader.dataset)))
    return running_loss / len(train_loader.dataset), correct * 100. / len(train_loader.dataset)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            model.eval()
            data = data
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logging.info('\nTest set: Average loss: %.4g, Accuracy: %.4g' 
                 % (test_loss, correct * 100. / len(test_loader.dataset)))

    return test_loss, correct * 100. / len(test_loader.dataset)


print(train_model)

if train_model:
    dev_accs = []
    dev_losses = []
    train_accs = []
    train_losses = []
    
    for epoch in range(10):
        print('start epoch %d' % epoch)
        train_loss, train_acc = train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                log_interval=100,
        )
        dev_loss, dev_acc = test(args, model, device, test_loader)

        dev_accs.append(dev_acc)
        dev_losses.append(dev_loss)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

    logging.info('--- finished training of the network, accuracy: %.3g ---' % dev_accs[-1])

    torch.save(model.to('cpu').state_dict(), model_save_path)
    model.to(device)


# --------------------------- FEATURE EXTRACTION -------------------------------

logging.info('--- extracting features from typeset ---')
model.eval()

type_dataset = np.load(os.path.join(args.data_root, 'typed_digits', 'data.npy'),
                       allow_pickle=True)[()]

X_train, y_train = type_dataset['X_train'] / 255, type_dataset['y_train']
X_test, y_test = type_dataset['X_test'] / 255, type_dataset['y_test']

T = transforms.Normalize((0.1307,), (0.3081,))

processed_X_train = T(torch.Tensor(X_train)).reshape(-1, 1, 28, 28)
processed_X_test = T(torch.Tensor(X_test)).reshape(-1, 1, 28, 28)

# logging.info(processed_X_train.min(), processed_X_train.max())

features = {}
predictions = {}
targets = dict(train=torch.LongTensor(y_train), val=torch.LongTensor(y_test))
with torch.no_grad():
    probs, features['train'] = model(processed_X_train.to(device))
    predictions['train'] = probs.argmax(1)
    probs, features['val'] = model(processed_X_test.to(device))
    predictions['val'] = probs.argmax(1)

for split in ('train', 'val'):
    out_path = os.path.join(typed_output_dir, split + '.pt')
    logging.info(f'Saving features to {out_path}')
    torch.save(dict(features=features[split].detach().cpu(),
                    targets=targets[split].detach().cpu(),
                    predictions=predictions[split].detach().cpu()),
               out_path)

logging.info('--- extracting features from MNIST ---')

for split in ('train', 'val'):
    logging.info(f'Extracting features from {split} set')
    loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=split=='train',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=60000 if split == 'train' else 10000, shuffle=False)

    for data, targets in loader:
        data = data.to(device)
        with torch.no_grad():
            probs, features = model(data)
        predictions = probs.argmax(1)

    out_path = os.path.join(mnist_output_dir, split + '.pt')
    logging.info(f'Saving features to {out_path}')
    torch.save(dict(features=features.detach().cpu(),
                    targets=targets.detach().cpu(),
                    predictions=predictions.detach().cpu()),
               out_path)

logging.info('--- done ---')
