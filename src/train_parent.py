# Package Includes
from __future__ import division

import os
import random
import socket
import timeit
from datetime import datetime

import networks.vgg_osvos as vo
from networks.drn_seg import DRNSeg
import numpy as np

import torch
import torch.optim as optim
from dataloaders import custom_transforms as tr
from dataloaders import davis_2016 as db
from layers.osvos_layers import class_balanced_cross_entropy_loss
from mypath import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from util import visualize as viz

# Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 5,  # Number of Images in each mini-batch
}

# # Setting other parameters
resume_epoch = False  # Default is False, change if want to resume
nEpochs = 240  # Number of epochs for training (nAveGrad * (50000)/(2079/trainBatch))
useTest = True  # See evolution of the test set when training?
testBatch = 5  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
db_root_dir = Path.db_root_dir()
vis_net = False  # Visualize the network?
snapshot = 5  # Store a model every snapshot epochs
nAveGrad = 2
seed = 123
log_to_tb = True
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Network definition
# model_name = 'VGG'
model_name = 'DRN_D_22'
if model_name == 'VGG':
    load_caffe_vgg = True
    num_losses = 5
    lr = 1e-8

    if not resume_epoch:
        if load_caffe_vgg:
            net = vo.OSVOS(pretrained=2)
        else:
            net = vo.OSVOS(pretrained=1)
    else:
        net = vo.OSVOS(pretrained=0)
        file_name = f'{model_name}_epoch-{resume_epoch - 1}.pth'
        print("Updating weights from: {}".format(os.path.join(save_dir, file_name)))
        net.load_state_dict(torch.load(os.path.join(save_dir, file_name),
                            map_location=lambda storage, loc: storage))
else:
    num_losses = 1
    lr = 1e-6

    net = DRNSeg('DRN_D_22', 1, pretrained=True, use_torch_up=False)
    if resume_epoch:
        parent_state_dict = torch.load(
        os.path.join(save_dir, 'DRN_D_22', f"DRN_D_22_epoch-{resume_epoch - 1}.pth"),
                     map_location=lambda storage, loc: storage)
        net.load_state_dict(parent_state_dict)

if not os.path.exists(os.path.join(save_dir, model_name)):
    os.makedirs(os.path.join(save_dir, model_name))

# parentModelName = 'parent'
# parentEpoch = 240
# net = vo.OSVOS(pretrained=0)
# net.load_state_dict(torch.load(os.path.join(save_dir, parentModelName+'_epoch-'+str(parentEpoch-1)+'.pth'),
#                                 map_location=lambda storage, loc: storage))

print(
    f'NUM MODEL PARAMS - {model_name}: {sum([p.numel() for p in net.parameters()])}')

# Logging into Tensorboard
if log_to_tb:
    log_dir = os.path.join(save_dir, 'runs', model_name)
    writer = SummaryWriter(log_dir=log_dir)

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

if model_name == 'VGG':
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
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.DAVIS2016(seqs='train_seqs', input_res=None, db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)

# Testing dataset and its iterator
db_test = db.DAVIS2016(seqs='test_seqs', db_root_dir=db_root_dir, transform=tr.ToTensor())
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=2)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
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
    for ii, sample_batched in enumerate(trainloader):
        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs.requires_grad_()
        inputs, gts = inputs.to(device), gts.to(device)

        outputs = net.forward(inputs)

        # Compute the losses
        losses = [0] * num_losses
        for i in range(num_losses):
            losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
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
            save_dir, model_name, model_name + '_epoch-' + str(epoch + 1) + '.pth'))

    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        with torch.no_grad():
            for ii, sample_batched in enumerate(testloader):
                inputs, gts = sample_batched['image'], sample_batched['gt']

                # Forward pass of the mini-batch
                inputs, gts = inputs.to(device), gts.to(device)

                outputs = net.forward(inputs)

                # Compute the losses, side outputs and fuse
                losses = [0] * num_losses
                for i in range(num_losses):
                    losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                    running_loss_ts[i] += losses[i].item()
                loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:
                    running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                    loss_ts.append(running_loss_ts[-1])

                    print(f'[TEST LOSS {loss:.2f}]')
                    if log_to_tb:
                        writer.add_scalar('test_loss_epoch', loss, epoch)
                    # for l in range(0, len(running_loss_ts)):
                    #     print('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    #     running_loss_ts[l] = 0
if log_to_tb:
    writer.close()
