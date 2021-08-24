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
import torchvision
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from datasets.datasets import Dataset, get_train_augmentations, get_test_augmentations
from models.scan import SCAN, ResNet18Classifier
from loss import TripletLoss
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


def construct_grid(batch, nrow: int = 8):
    images = torchvision.utils.make_grid(batch, nrow=nrow)
    images = images.detach().cpu().numpy()
    return images


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def calc_losses(configs, clf_criterion, triplet_loss, outs, clf_out, target):
    clf_loss = (
            clf_criterion(clf_out, target)
            * configs['loss_coef']["clf_loss"]
    )
    cue = outs[-1]
    cue = target.reshape(-1, 1, 1, 1) * cue
    num_reg = (
            torch.sum(target) * cue.shape[1] * cue.shape[2] * cue.shape[3]
    ).type(torch.float)
    reg_loss = (
                       torch.sum(torch.abs(cue)) / (num_reg + 1e-9)
               ) * configs['loss_coef']['reg_loss']

    trip_loss = 0
    bs = outs[-1].shape[0]
    for feat in outs[:-1]:
        feat = F.adaptive_avg_pool2d(feat, [1, 1]).view(bs, -1)
        trip_loss += (
                triplet_loss(feat, target)
                * configs['loss_coef']['trip_loss']
        )
    total_loss = clf_loss + reg_loss + trip_loss

    return total_loss, clf_loss, reg_loss, trip_loss


def main(configs, writer, lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, val_fas=5,
         device=torch.device("cpu"),
         download_location='~/data'):
    mean = (configs['mean']['r'], configs['mean']['g'], configs['mean']['b'])
    std = (configs['std']['r'], configs['std']['g'], configs['std']['b'])

    transforms = get_train_augmentations(configs['image_size'], mean=mean, std=std)

    print("Reading CSV...")
    train_df = pd.read_csv(configs['train_df'])
    val_df = pd.read_csv(configs['val_df'])

    train_dataset = Dataset(
        train_df, configs['dataset_root'], transforms, face_detector=None,
        bookkeeping_path=configs['train_bookkeeping_path'],
    )

    validation_dataset = Dataset(
        val_df, configs['dataset_root'], transforms, face_detector=None,
        bookkeeping_path=configs['val_bookkeeping_path'],
    )

    print("Generating meta-training dataset using ", train_dataset.bookkeeping_path)
    meta_train = l2l.data.MetaDataset(train_dataset)

    print("Generating meta-training tasks...")
    train_tasks = l2l.data.TaskDataset(meta_train,
                                       task_transforms=[
                                           l2l.data.transforms.NWays(meta_train, ways),
                                           l2l.data.transforms.KShots(meta_train, shots * configs['sample_count_factor'], replacement=True),
                                           l2l.data.transforms.LoadData(meta_train),
                                           # l2l.data.transforms.RemapLabels(meta_train),
                                           # l2l.data.transforms.ConsecutiveLabels(meta_train),
                                       ],
                                       num_tasks=20000)

    print("Generating meta-validation dataset using ", validation_dataset.bookkeeping_path)
    meta_validation = l2l.data.MetaDataset(validation_dataset)

    print("Generating meta-validation tasks...")
    val_tasks = l2l.data.TaskDataset(meta_validation,
                                     task_transforms=[
                                         l2l.data.transforms.NWays(meta_validation, ways),
                                         l2l.data.transforms.KShots(meta_validation, shots * configs['sample_count_factor'], replacement=True),
                                         l2l.data.transforms.LoadData(meta_validation),
                                     ],
                                     num_tasks=1000)
    if configs['pretrained']:
        model = ResNet18Classifier(pretrained=True)
    else:
        model = ResNet18Classifier(pretrained=False)

    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=lr, allow_nograd=False, first_order=True)

    if 'checkpoint' in configs:
        print("Loading model from ", configs['checkpoint'])
        meta_model.load_state_dict(torch.load(configs['checkpoint']))

    opt = optim.Adam(meta_model.parameters(), lr=maml_lr)
    scheduler = MultiStepLR(opt, milestones=configs['milestones'], gamma=configs['gamma'])
    clf_criterion = nn.CrossEntropyLoss()

    print("Starting meta-training...")
    for iteration in range(iterations):
        opt.zero_grad()

        iteration_error = 0.0
        iteration_clf_loss = 0.0
        iteration_triplet_loss = 0.0
        iteration_reg_loss = 0.0
        iteration_acc = 0.0
        iteration_acer = 0.0
        iteration_apcer = 0.0
        iteration_npcer = 0.0

        val_iteration_error = 0.0
        val_iteration_clf_loss = 0.0
        val_iteration_triplet_loss = 0.0
        val_iteration_reg_loss = 0.0
        val_iteration_acc = 0.0
        val_iteration_acer = 0.0
        val_iteration_apcer = 0.0
        val_iteration_npcer = 0.0

        # Meta-training loop
        for task in range(tps):
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            data, labels = train_task
            data = data.to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[:shots] = True
            length = adaptation_indices.shape[0]
            adaptation_indices[math.floor(length / 2):math.floor(length / 2 + shots)] = True

            # adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Fast Adaptation
            for step in range(fas):
                outs = learner(adaptation_data)
                train_loss = clf_criterion(outs, adaptation_labels)

                if configs['plot_inner_loop_loss'] and iteration % configs['plot_inner_loop_interval'] == 0:
                    writer.add_scalar('Adaptation Loss (training)/Iteration ' + str(iteration) + ' Task ' + str(task), train_loss,
                                      step)

                # train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_loss)

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = clf_criterion(predictions, evaluation_labels)
            valid_error /= len(evaluation_data)
            valid_accuracy = accuracy(predictions, evaluation_labels)
            acer, apcer, npcer = metrics.get_metrics(predictions.argmax(dim=1).cpu().numpy(), evaluation_labels.cpu())

            valid_error.backward()

            iteration_error += valid_error
            iteration_acc += valid_accuracy
            iteration_acer += acer
            iteration_apcer += apcer
            iteration_npcer += npcer

        if configs['log_tasks'] and iteration % configs['log_tasks_interval'] == 0:
            adaptation_images = construct_grid(adaptation_data, nrow=2 * shots)
            evaluation_images = construct_grid(evaluation_data, nrow=10)

            writer.add_image("Sample tasks (meta-training)/Adaptation", adaptation_images, iteration)
            writer.add_image("Sample tasks (meta-training)/Evaluation", evaluation_images, iteration)

        # Meta-training metrics
        iteration_error /= tps
        iteration_clf_loss /= tps
        iteration_triplet_loss /= tps
        iteration_reg_loss /= tps
        iteration_acc /= tps
        iteration_acer /= tps
        iteration_apcer /= tps
        iteration_npcer /= tps

        # Take the meta-learning step
        # opt.zero_grad()
        # iteration_error.backward()

        for p in meta_model.parameters():
            p.grad.data.mul_(1.0 / tps)

        opt.step()

        scheduler.step()

        # Meta-validation loop
        for task in range(tps):
            learner = meta_model.clone()
            val_task = val_tasks.sample()
            data, labels = val_task
            data = data.to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[:shots] = True
            length = adaptation_indices.shape[0]
            adaptation_indices[math.floor(length / 2):math.floor(length / 2 + shots)] = True

            # adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            val_adaptation_data, val_adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            val_evaluation_data, val_evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Fast Adaptation
            if not configs['no_adaptation']:
                for step in range(val_fas):
                    outs = learner(val_adaptation_data)
                    train_loss = clf_criterion(outs, val_adaptation_labels)

                    if configs['plot_inner_loop_loss'] and iteration % configs['plot_inner_loop_interval'] == 0:
                        writer.add_scalar('Adaptation Loss (validation)/Iteration ' + str(iteration) + ' Task ' + str(task),
                                          train_loss, step)
                    learner.adapt(train_loss)

            # Compute validation loss
            predictions = learner(val_evaluation_data)
            valid_error = clf_criterion(predictions, val_evaluation_labels)
            valid_error /= len(val_evaluation_data)
            valid_accuracy = accuracy(predictions, val_evaluation_labels)
            acer, apcer, npcer = metrics.get_metrics(predictions.argmax(dim=1).cpu().numpy(),
                                                     val_evaluation_labels.cpu())

            val_iteration_error += valid_error
            val_iteration_acc += valid_accuracy
            val_iteration_acer += acer
            val_iteration_apcer += apcer
            val_iteration_npcer += npcer


        if configs['log_tasks'] and iteration % configs['log_tasks_interval'] == 0:
            val_adaptation_images = construct_grid(val_adaptation_data, nrow=2 * shots)
            val_evaluation_images = construct_grid(val_evaluation_data, nrow=10)

            writer.add_image("Sample tasks (meta-validation)/Adaptation", val_adaptation_images, iteration)
            writer.add_image("Sample tasks (meta-validation)/Evaluation", val_evaluation_images, iteration)

        # Meta-validation metrics
        val_iteration_error /= tps
        val_iteration_clf_loss /= tps
        val_iteration_triplet_loss /= tps
        val_iteration_reg_loss /= tps
        val_iteration_acc /= tps
        val_iteration_acer /= tps
        val_iteration_apcer /= tps
        val_iteration_npcer /= tps

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, iteration)

        # Plotting meta-training metrics
        writer.add_scalar('Losses (training)/Total', iteration_error, iteration)
        writer.add_scalar('Metrics (training)/Accuracy', iteration_acc, iteration)
        writer.add_scalar('Metrics (training)/acer', iteration_acer, iteration)
        writer.add_scalar('Metrics (training)/apcer', iteration_apcer, iteration)
        writer.add_scalar('Metrics (training)/npcer', iteration_npcer, iteration)

        # Plotting meta-validation metrics
        writer.add_scalar('Losses (validation)/Total', val_iteration_error, iteration)
        writer.add_scalar('Metrics (validation)/Accuracy', val_iteration_acc, iteration)
        writer.add_scalar('Metrics (validation)/acer', val_iteration_acer, iteration)
        writer.add_scalar('Metrics (validation)/apcer', val_iteration_apcer, iteration)
        writer.add_scalar('Metrics (validation)/npcer', val_iteration_npcer, iteration)

        print('Version: {:d} Iteration: {:d} Loss : {:.3f} Acc : {:.3f} Val Loss : {:.3f} Val Acc : {:.3f}'.format(
            configs['version'],
            iteration,
            iteration_error.item(),
            iteration_acc,
            val_iteration_error.item(),
            val_iteration_acc)
        )

        if iteration % configs['save_weight_interval'] == 0:
            torch.save(meta_model.state_dict(), weights_directory + "epoch_" + str(iteration) + ".pth")


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

    version = get_latest_version(log_dir)

    version_directory = log_dir + "version_" + str(version)
    if not os.path.isdir(version_directory):
        os.makedirs(version_directory)

    if 'checkpoint' in configs:
        checkout = open(version_directory + '/CHECKPOINT', 'w').close()

    weights_directory = version_directory + "/weights/"
    if not os.path.isdir(weights_directory):
        os.makedirs(weights_directory)

    debug = args.debug

    start = datetime.datetime.now()
    configs['start'] = start
    configs['version'] = version
    configs['debug'] = debug
    configs['weights_directory'] = weights_directory

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

    main(configs=configs,
         writer=writer,
         lr=configs['lr'],
         maml_lr=configs['maml_lr'],
         iterations=configs['iterations'],
         ways=configs['ways'],
         shots=configs['shots'],
         tps=configs['tasks_per_step'],
         fas=configs['fast_adaptation_steps'],
         val_fas=configs['val_fast_adaptation_steps'],
         device=device)
