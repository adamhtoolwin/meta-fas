import argparse
import random
import os
import yaml
import datetime
import glob

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter

import learn2learn as l2l

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'


def get_latest_version(root_dir: str):
    last_version = 1
    directories = glob.glob(root_dir + "/version_*/")
    for directory in directories:
        version = int(directory.split("/")[-2].split("_")[-1])
        if version >= last_version:
            last_version = version + 1

    return last_version


class Net(nn.Module):
    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def main(weights_directory, writer, lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, device=torch.device("cpu"),
         download_location='~/data'):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 28, 28),
    ])

    mnist_train = l2l.data.MetaDataset(MNIST(download_location,
                                             train=True,
                                             download=True,
                                             transform=transformations))

    train_tasks = l2l.data.TaskDataset(mnist_train,
                                       task_transforms=[
                                            l2l.data.transforms.NWays(mnist_train, ways),
                                            l2l.data.transforms.KShots(mnist_train, 2*shots),
                                            l2l.data.transforms.LoadData(mnist_train),
                                            l2l.data.transforms.RemapLabels(mnist_train),
                                            l2l.data.transforms.ConsecutiveLabels(mnist_train),
                                       ],
                                       num_tasks=1000)

    model = Net(ways)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction='mean')

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            data, labels = train_task
            data = data.to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Fast Adaptation
            for step in range(fas):
                train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error)

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, evaluation_labels)
            valid_error /= len(evaluation_data)
            valid_accuracy = accuracy(predictions, evaluation_labels)
            iteration_error += valid_error
            iteration_acc += valid_accuracy

        iteration_error /= tps
        iteration_acc /= tps

        writer.add_scalar('Loss (iteration)', iteration_error, iteration)
        writer.add_scalar('Accuracy', iteration_acc, iteration)

        print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()

        torch.save(meta_model.state_dict(), weights_directory + "epoch_" + str(iteration) + ".pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument("-c", "--config", required=True, help="Config file path.")
    parser.add_argument("-d", "--debug", required=False, type=bool, help="Checkpoint file path.", default=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        configs = yaml.safe_load(stream)

    root_dir = os.getcwd()
    log_dir = configs['log_dir']
    log_dir = os.path.join(root_dir, log_dir)

    version = get_latest_version(log_dir)

    version_directory = log_dir + "version_" + str(version)
    if not os.path.isdir(version_directory):
        os.makedirs(version_directory)

    weights_directory = version_directory + "/weights/"
    if not os.path.isdir(weights_directory):
        os.makedirs(weights_directory)

    debug = args.debug

    start = datetime.datetime.now()
    configs['start'] = start
    configs['version'] = version
    configs['debug'] = debug

    with open(version_directory + '/configs.yml', 'w') as outfile:
        yaml.dump(configs, outfile, default_flow_style=False)

    # ========================= End of DevOps ==========================
    # ========================= Start of ML ==========================

    use_cuda = not configs['no_cuda'] and torch.cuda.is_available()

    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    torch.manual_seed(configs['seed'])
    if use_cuda:
        torch.cuda.manual_seed(configs['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(configs['gpu']) if use_cuda else "cpu")

    print("Using", device)
    print("Version: ", version)
    print("Debug: ", debug)

    writer = SummaryWriter(log_dir=version_directory)

    main(weights_directory=weights_directory,
         writer=writer,
         lr=configs['lr'],
         maml_lr=configs['maml_lr'],
         iterations=configs['iterations'],
         ways=configs['ways'],
         shots=configs['shots'],
         tps=configs['tasks_per_step'],
         fas=configs['fast_adaption_steps'],
         device=device,
         download_location=configs['download_location'])
