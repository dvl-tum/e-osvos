# Package Includes
from __future__ import division

import os
import random
import socket
import timeit
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import DAVIS, VOC2012, YouTube
from data import custom_transforms as tr
from layers.osvos_layers import class_balanced_cross_entropy_loss, dice_loss
from mypath import Path
from networks.drn_seg import DRNSeg
from networks.fpn import FPN
from networks.unet import Unet
from networks.vgg_osvos import OSVOSVgg
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from util import visualize as viz
from util.helper_func import eval_loader, run_loader

# Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Setting of parameters
# Parameters in p are used for the name of the model

# # Setting other parameters
resume_epoch = False  # Default is False, change if want to resume
nEpochs = 500  # Number of epochs for training (nAveGrad * (50000)/(2079/train_batch))
useTest = True  # See evolution of the test set when training?
train_batch = 8  # Testing Batch
test_batch = 8  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
vis_net = False  # Visualize the network?
snapshot = 5  # Store a model every snapshot epochs
nAveGrad = 1
seed = 123
log_to_tb = True

save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# DAVIS
db_root_dir = 'data/DAVIS-2016'
# db_root_dir = 'data/DAVIS-2017'

# train_dataset = 'train_seqs'
# test_dataset = 'val_seqs'

# train_dataset = 'train_split_3_train'
# test_dataset = 'train_split_3_val'

train_dataset = 'train_split_balanced_train'
test_dataset = 'train_split_balanced_val'

# PASCAL VOC
# db_root_dir = 'data/VOC2012'
# train_dataset = 'pascal_voc'


# YoutTube VOS
# db_root_dir = 'data/YouTube-VOS'
# train_dataset = 'train'

# Network definition
# model_name = 'VGG'
# model_name = 'DRN_D_22'
# model_name = 'UNET_ResNet18_dice_loss'
# model_name = 'UNET_ResNet34'
model_name = 'FPN_ResNet34'
# loss_func = 'cross_entropy'
loss_func = 'dice'

if 'VGG' in model_name:
    load_caffe_vgg = True
    num_losses = 5
    lr = 1e-8

    if not resume_epoch:
        if load_caffe_vgg:
            net = OSVOSVgg(pretrained=2)
        else:
            net = OSVOSVgg(pretrained=1)
    else:
        net = OSVOSVgg(pretrained=0)
        file_name = f'{model_name}_epoch-{resume_epoch - 1}.pth'
        print("Updating weights from: {}".format(os.path.join(save_dir, file_name)))
        net.load_state_dict(torch.load(os.path.join(save_dir, file_name),
                            map_location=lambda storage, loc: storage))
elif 'DRN_D_22' in model_name:
    num_losses = 1
    lr = 1e-6

    net = DRNSeg('DRN_D_22', 1, pretrained=True, use_torch_up=False)
elif 'UNET_ResNet18' in model_name:
    num_losses = 1
    lr = 1e-3

    net = Unet('resnet18', classes=1, activation='softmax')
elif 'UNET_ResNet34' in model_name:
    num_losses = 1
    lr = 1e-7

    net = Unet('resnet34', classes=1, activation='softmax')
elif 'FPN_ResNet34' in model_name:
    num_losses = 1
    lr = 1e-5

    net = FPN('resnet34', classes=1, activation='softmax')
elif 'FPN_ResNet101' in model_name:
    num_losses = 1
    lr = 1e-5

    net = FPN('resnet101', classes=1, activation='softmax')

log_dir = os.path.join(model_name, db_root_dir.split('/')[-1], train_dataset)

if resume_epoch:
    parent_state_dict = torch.load(
        os.path.join(save_dir, log_dir, f"{model_name}_epoch-{resume_epoch}.pth"),
        map_location=lambda storage, loc: storage)
    net.load_state_dict(parent_state_dict)

if not os.path.exists(os.path.join(save_dir, model_name)):
    os.makedirs(os.path.join(save_dir, model_name))
if not os.path.exists(os.path.join(save_dir, model_name, db_root_dir.split('/')[-1])):
    os.makedirs(os.path.join(save_dir, model_name, db_root_dir.split('/')[-1]))
if not os.path.exists(os.path.join(save_dir, model_name, db_root_dir.split('/')[-1], train_dataset)):
    os.makedirs(os.path.join(save_dir, model_name, db_root_dir.split('/')[-1], train_dataset))

# parentModelName = 'parent'
# parentEpoch = 240
# net = OSVOSVgg(pretrained=0)
# net.load_state_dict(torch.load(os.path.join(save_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
#                                 map_location=lambda storage, loc: storage))

print(log_dir)
print(f'NUM MODEL PARAMS - {model_name}: {sum([p.numel() for p in net.parameters()])}')

# Logging into Tensorboard
if log_to_tb:
    tf_log_dir = os.path.join('log/tf_runs', log_dir)
    writer = SummaryWriter(log_dir=tf_log_dir)

net.to(device)

# Visualize the network
if vis_net:
    x = torch.randn(1, 3, 480, 854)
    x.requires_grad_()
    x = x.to(device)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()

# Use the following optimizer
wd = 0.0002
if 'VGG' in model_name:
    optimizer = optim.SGD([
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd,
         'initial_lr': lr},
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': 2 * lr, 'initial_lr': 2 * lr},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd,
         'initial_lr': lr},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': 2 * lr,
         'initial_lr': 2 * lr},
        {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'weight' in pr[0]], 'lr': lr / 10,
         'weight_decay': wd, 'initial_lr': lr / 10},
        {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'bias' in pr[0]], 'lr': 2 * lr / 10,
         'initial_lr': 2 * lr / 10},
        {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
        {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
        {'params': net.fuse.weight, 'lr': lr / 100, 'initial_lr': lr / 100, 'weight_decay': wd},
        {'params': net.fuse.bias, 'lr': 2 * lr / 100, 'initial_lr': 2 * lr / 100},
    ], lr=lr, momentum=0.9)
else:
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                        tr.RandomScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                        tr.ToTensor()])
train_crop_size = None
train_multi_object = False

if 'DAVIS' in db_root_dir:
    # train is cropped. but for davis 2017 test batch has changing heights and widths
    if 'DAVIS-2017' in db_root_dir:
        train_crop_size = (480, 854)
    db_train = DAVIS(seqs_key=train_dataset,
                     root_dir=db_root_dir,
                     crop_size=train_crop_size,
                     transform=composed_transforms,
                     multi_object=train_multi_object)

    db_test = DAVIS(seqs_key=test_dataset,
                    root_dir=db_root_dir,
                    transform=tr.ToTensor())

elif 'YouTube-VOS' in db_root_dir:
    train_batch = 16
    test_batch = 16
    nTestInterval = 2
    train_crop_size = (720, 960)
    train_multi_object = False#'single_id'

    # Testing dataset and its iterator
    db_train = YouTube(seqs_key=train_dataset,
                       root_dir=db_root_dir,
                       crop_size=train_crop_size,
                       transform=composed_transforms,
                       multi_object=train_multi_object)

    if train_multi_object == 'single_id':
        db_train.multi_object_id = 0

    # validate YouTube-VOS with DAVIS-16 val
    db_test = DAVIS(seqs_key='val_seqs',
                    root_dir='data/DAVIS-2016',
                    transform=tr.ToTensor())

    # Training dataset and its iterator

elif 'VOC2012' in db_root_dir:
    db_train = VOC2012(split='train')
    # db_test = VOC2012(split='val')
    db_test = DAVIS(seqs_key='val_seqs',
                    root_dir='data/DAVIS-2016',
                    transform=tr.ToTensor())
else:
    raise NotImplementedError

print(f"DATA - TRAIN LENGTH: {len(db_train)} - TEST LENGTH: {len(db_test)}")

train_loader = DataLoader(db_train, batch_size=train_batch, shuffle=True, num_workers=2)
test_loader = DataLoader(db_test, batch_size=test_batch, shuffle=False, num_workers=2)

num_img_tr = len(train_loader)
num_img_ts = len(test_loader)
running_loss_tr = [0] * num_losses
running_loss_ts = [0] * num_losses
loss_tr = []
loss_ts = []
aveGrad = 0

print("Training Network")
# Main Training and Testing Loop
for epoch in range(resume_epoch, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(train_loader):
        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs.requires_grad_()
        inputs, gts = inputs.to(device), gts.to(device)

        net.train()
        outputs = net.forward(inputs)

        # Compute the losses
        losses = [0] * num_losses
        for i in range(num_losses):
            if loss_func == 'cross_entropy':
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts)
            elif loss_func == 'dice':
                losses[i] = dice_loss(outputs[i], gts)
            else:
                raise NotImplementedError

            running_loss_tr[i] += losses[i].item()
        loss = (1 - epoch / nEpochs)*sum(losses[:-1]) + losses[-1]

        # Print stuff

        # if ii % num_img_tr == num_img_tr - 1:
        if (ii + 1) % 25 == 0:
            running_loss_tr = [x / num_img_tr for x in running_loss_tr]
            loss_tr.append(running_loss_tr[-1])
            # writer.add_scalar('data/total_loss_epoch', running_loss_tr[-1], epoch)
            if log_to_tb:
                writer.add_scalar('total_loss_epoch', loss,
                                (ii + 1) + num_img_tr * epoch)
            print(f'[EPOCH {epoch + 1} ITER {ii + 1} LOSS {loss:.2f}]')
            # for l in range(0, len(running_loss_tr)):
            #     print(f'LOSS {l}: {running_loss_tr[l]:.2f}')
            #     running_loss_tr[l] = 0

            # stop_time = timeit.default_timer()
            # print("Execution time: " + str(stop_time - start_time))

        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

        # if ii + 1 == 100:
        #     break

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        torch.save(net.state_dict(), os.path.join(
            save_dir, model_name, db_root_dir.split('/')[-1], train_dataset, model_name + '_epoch-' + str(epoch + 1) + '.pth'))

    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        with torch.no_grad():
            metrics_names = ['test_loss', 'test_acc', 'test_J', 'test_F']
            metrics = {n: [] for n in metrics_names}

            net.eval()
            if isinstance(db_test, VOC2012):
                test_loss_batches, test_acc_batches = run_loader(net, test_loader, loss_func)
                metrics['test_loss'].append(test_loss_batches.mean())
                metrics['test_acc'].append(test_acc_batches.mean())
                metrics['test_J'].append(0.0)
                metrics['test_F'].append(0.0)
            else:
                for seq_name in db_test.seqs_names:
                    db_test.set_seq(seq_name)
                    test_loss_batches, test_acc_batches, test_J, test_F = eval_loader(
                        net, test_loader, loss_func)
                    metrics['test_loss'].append(test_loss_batches.mean())
                    metrics['test_acc'].append(test_acc_batches.mean())
                    metrics['test_J'].append(test_J)
                    metrics['test_F'].append(test_F)

            metrics = {n: torch.tensor(m).mean() for n, m in metrics.items()}

            if log_to_tb:
                writer.add_scalar('test_metrics/loss', metrics['test_loss'], epoch)
                writer.add_scalar('test_metrics/acc', metrics['test_acc'], epoch)
                writer.add_scalar('test_metrics/J_mean', metrics['test_J'], epoch)
                writer.add_scalar('test_metrics/F_mean', metrics['test_F'], epoch)

if log_to_tb:
    writer.close()
