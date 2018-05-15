import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import datetime
import pdb
import numpy as np


def train_model(train_data, dev_data, model, args):
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad] , lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        # Report Epoch Losses
        train_loss, train_accuracy = run_epoch(train_data, True, model, optimizer, args)
        print('Train  MSE loss: {:.6f}  acc: {:.6f}\n'.format(train_loss, train_accuracy))
        val_loss, val_accuracy = run_epoch(dev_data, False, model, optimizer, args)
        print('Val MSE loss: {:.6f}  acc: {:.6f}\n'.format(val_loss, val_accuracy))
        # Save model
        torch.save(model, args.save_path)


def eval_model(dev_data, model, args):
    if args.cuda:
        model = model.cuda()
    test_loss, test_accuracy = run_epoch(dev_data, False, model, None, args)
    print('Train  MSE loss: {:.6f}  acc: {:.6f}\n'.format(test_loss, test_accuracy))


def compute_accuracy(out, y):
    return np.mean(np.equal(out, y))


def run_epoch(data, is_training, model, optimizer, args):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)

    losses = []
    accuracies = []
    if is_training:
        model.train()
    else:
        model.eval()
    for batch in tqdm(data_loader):
        x, y = autograd.Variable(batch['x']), autograd.Variable(batch['y'])
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        if is_training:
            optimizer.zero_grad()
        out = model(x)
        if args.loss == "MSE":
            loss = F.mse_loss(out, y.float())
        elif args.loss == "BCE":
            loss = F.binary_cross_entropy_with_logits(out, y.float())
        else:
            raise Exception("invalid loss parameter")
        if is_training:
            loss.backward()
            optimizer.step()
        losses.append(loss.cpu().item())
        predictions = np.rint(out.cpu().detach().numpy())
        accuracy = compute_accuracy(predictions, y.cpu().detach().numpy())
        accuracies.append(accuracy)
    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    accuracy = np.mean(accuracies)
    return avg_loss, accuracy
