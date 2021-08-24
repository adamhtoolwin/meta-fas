import argparse
import random
import os
import yaml
import datetime
import glob
import pickle
import math

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import Dataset, get_train_augmentations, get_test_augmentations
from loss import TripletLoss
from meta_training import calc_losses, construct_grid
from models.scan import SCAN, ResNet18Classifier
import metrics

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


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def main(configs, writer, lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5,
         device=torch.device("cpu"),
         download_location='~/data'):

    mean = (configs['mean']['r'], configs['mean']['g'], configs['mean']['b'])
    std = (configs['std']['r'], configs['std']['g'], configs['std']['b'])

    transforms = get_test_augmentations(configs['image_size'], mean=mean, std=std)

    df = pd.read_csv(configs['train_df'])

    dataset = Dataset(
        df, configs['dataset_root'], transforms, face_detector=None,
        bookkeeping_path=configs['bookkeeping_path'],
        # bookkeeping_path = configs['bookkeeping_path'] + "bookkeeping_" + configs['train_df'].split("/")[-1]
    )

    print("Generating meta-dataset using ", dataset.bookkeeping_path)
    meta_test = l2l.data.MetaDataset(dataset)

    print("Generating taskset...")
    val_tasks = l2l.data.TaskDataset(meta_test,
                                     task_transforms=[
                                         l2l.data.transforms.NWays(meta_test, ways),
                                         l2l.data.transforms.KShots(meta_test, shots * configs['sample_count_factor'], replacement=True),
                                         l2l.data.transforms.LoadData(meta_test),
                                         # l2l.data.transforms.RemapLabels(meta_test),
                                         # l2l.data.transforms.ConsecutiveLabels(meta_test),
                                     ],
                                     num_tasks=-1)

    model = SCAN(pretrained=False)
    meta_model = l2l.algorithms.MAML(model, lr=lr, allow_nograd=False, first_order=True)

    print("Loading model from ", configs['weights'])
    meta_model.load_state_dict(torch.load(configs['weights'], map_location='cpu'))
    model.to(device)

    opt = optim.Adam(meta_model.parameters(), lr=maml_lr)
    triplet_loss = TripletLoss()
    clf_criterion = nn.CrossEntropyLoss()

    total_metrics = {
        'accuracy': 0.0,
        'acer': 0.0,
        'apcer': 0.0,
        'npcer': 0.0
    }

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_clf_loss = 0.0
        iteration_triplet_loss = 0.0
        iteration_reg_loss = 0.0

        iteration_acc = 0.0
        iteration_acer = 0.0
        iteration_apcer = 0.0
        iteration_npcer = 0.0

        for task in range(tps):
            learner = meta_model.clone()
            train_task = val_tasks.sample()
            data, labels = train_task
            data = data.to(device)
            labels = labels.to(device)

            live_count = len(labels[labels == 1])
            spoof_count = len(labels[labels == 0])
            assert live_count == spoof_count

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)

            # first 5 of class 1 chosen
            adaptation_indices[:shots] = True
            length = adaptation_indices.shape[0]
            # first 5 of class 2 chosen
            adaptation_indices[math.floor(length / 2):math.floor(length / 2 + shots)] = True

            # adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Fast Adaptation
            if not configs['no_adaptation']:
                for step in range(fas):
                    outs, clf_out = learner(adaptation_data)
                    train_loss, clf_loss, reg_loss, trip_loss = calc_losses(configs,
                                                                            clf_criterion,
                                                                            triplet_loss,
                                                                            outs,
                                                                            clf_out,
                                                                            adaptation_labels
                                                                            )

                    if configs['plot_inner_loop_loss'] and iteration % configs['plot_inner_loop_interval'] == 0:
                        writer.add_scalar('Adaptation Loss/Iteration ' + str(iteration) + ' Task ' + str(task),
                                          train_loss, step)
                    learner.adapt(train_loss)
            else:
                evaluation_data = data
                evaluation_labels = labels

            # Compute validation loss
            outs, clf_out = learner(evaluation_data)
            val_loss, clf_loss, reg_loss, trip_loss = calc_losses(configs, clf_criterion, triplet_loss, outs, clf_out,
                                                                  evaluation_labels)

            scores = []
            cues = outs[-1]
            for i in range(cues.shape[0]):
                score = 1.0 - cues[i,].mean().cpu().item()
                scores.append(score)

            metrics_, best_thr, val_acc = metrics.eval_from_scores(np.array(scores),
                                                                   evaluation_labels.cpu().long().numpy())
            acer, apcer, npcer = metrics_

            iteration_error += float(val_loss)
            iteration_clf_loss += float(clf_loss)
            iteration_triplet_loss += float(trip_loss)
            iteration_reg_loss += float(reg_loss)

            iteration_acc += val_acc
            iteration_acer += acer
            iteration_apcer += apcer
            iteration_npcer += npcer

        iteration_error /= tps
        iteration_clf_loss /= tps
        iteration_triplet_loss /= tps
        iteration_reg_loss /= tps

        iteration_acc /= tps
        iteration_acer /= tps
        iteration_apcer /= tps
        iteration_npcer /= tps

        writer.add_scalar('Losses (meta-testing)/Total', iteration_error, iteration)
        writer.add_scalar('Losses (meta-testing)/classifier loss', iteration_clf_loss, iteration)
        writer.add_scalar('Losses (meta-testing)/triplet loss', iteration_triplet_loss, iteration)
        writer.add_scalar('Losses (meta-testing)/regression loss', iteration_reg_loss, iteration)

        writer.add_scalar('Metrics (meta-testing)/Accuracy', iteration_acc, iteration)
        writer.add_scalar('Metrics (meta-testing)/acer', iteration_acer, iteration)
        writer.add_scalar('Metrics (meta-testing)/apcer', iteration_apcer, iteration)
        writer.add_scalar('Metrics (meta-testing)/npcer', iteration_npcer, iteration)

        print('Version: {:d} Iteration: {:d} Loss : {:.3f} Acc : {:.3f}'.format(configs['version'],
                                                                                iteration,
                                                                                iteration_error,
                                                                                iteration_acc)
              )

        if configs['log_tasks'] and iteration % configs['log_tasks_interval'] == 0:
            adaptation_images = construct_grid(adaptation_data, nrow=2 * shots)
            evaluation_images = construct_grid(evaluation_data, nrow=10)

            writer.add_image("Last task (adapt)", adaptation_images, iteration)
            writer.add_image("Last task (evaluation)", evaluation_images, iteration)

        total_metrics['accuracy'] += iteration_acc
        total_metrics['acer'] += iteration_acer
        total_metrics['apcer'] += iteration_apcer
        total_metrics['npcer'] += iteration_npcer

        # # Take the meta-learning step
        # opt.zero_grad()
        # iteration_error.backward()
        # opt.step()

    avg_acc = total_metrics['accuracy'] / iterations
    avg_acer = total_metrics['acer'] / iterations
    avg_apcer = total_metrics['apcer'] / iterations
    avg_npcer = total_metrics['npcer'] / iterations

    print('Averages - Acc: {:.3f} ACER: {:.3f} APCER: {:.3f} NPCER: {:.3f}'
          .format(avg_acc, avg_acer, avg_apcer, avg_npcer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Oulu Training')

    parser.add_argument("-c", "--config", required=True, help="Config file path.")
    parser.add_argument("-d", "--debug", required=False, type=bool, help="Checkpoint file path.", default=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        configs = yaml.safe_load(stream)

    root_dir = os.getcwd()
    log_dir = configs['log_dir']
    log_dir = os.path.join(root_dir, log_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    version = get_latest_version(log_dir)

    version_directory = log_dir + "version_" + str(version)
    if not os.path.isdir(version_directory):
        os.makedirs(version_directory)

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
    print("Debug: ", debug)
    print("Version: ", version)

    writer = SummaryWriter(log_dir=version_directory)

    main(configs=configs,
         writer=writer,
         lr=configs['lr'],
         maml_lr=configs['maml_lr'],
         iterations=configs['iterations'],
         ways=configs['ways'],
         shots=configs['shots'],
         tps=configs['tasks_per_step'],
         fas=configs['fast_adaption_steps'],
         device=device)
