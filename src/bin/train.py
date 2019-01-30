#!/usr/bin/env python

# Created on 2019/01
# Author: Kaituo XU (NPU-ASLP)

import argparse

import torch
from torch.utils.data import DataLoader

import hyperparams as hparams
from audio_process import spectrogram
from data import LJSpeechDataset, RandomBucketBatchSampler, TextAudioCollate
from loss import FeaturePredictNetLoss
from model import FeaturePredictNet
from solver import Solver
from text_process import text_to_sequence


parser = argparse.ArgumentParser("Tacotron2 FeaturePredictNet Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, required=True, help='dir including wav')
parser.add_argument('--train_csv', type=str, default='metadata.csv', help='csv file such metadata.csv')
# Network architecture
# We use paper config right now. If need to adjust model config, then add options.
# Training config
parser.add_argument('--use_cuda', type=int, default=1)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--max_norm', default=1, type=float, help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--batch_size', default=32, type=int)
# optimizer
parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
parser.add_argument('--l2', default=0.0, type=float, help='weight decay (L2)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp', help='Dir to save models')
parser.add_argument('--checkpoint', default=0, type=int, help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar', help='model name')
# logging
parser.add_argument('--print_freq', default=10, type=int, help='Frequency of printing training infomation')
parser.add_argument('--visdom', type=int, default=0, help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', type=int, default=0, help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='Taco2 training', help='Identifier for visdom run')


def main(args):
    # Build data loader
    dataset = LJSpeechDataset(args.train_dir, args.train_csv,
                              text_transformer=text_to_sequence,
                              audio_transformer=spectrogram)
    batch_sampler = RandomBucketBatchSampler(dataset,
                                             batch_size=args.batch_size,
                                             drop_last=False)
    collate_fn = TextAudioCollate()
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                            collate_fn=collate_fn, num_workers=1)
    # Build model
    model = FeaturePredictNet(hparams.num_chars, hparams.padding_idx,
                              hparams.feature_dim)
    # print(model)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # Build criterion
    criterion = FeaturePredictNetLoss()
    # Build optimizer
    optimizier = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.l2,
                                  betas=(0.9, 0.999), eps=1e-6)
    # Build Solver
    solver = Solver(data_loader, model, criterion, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
