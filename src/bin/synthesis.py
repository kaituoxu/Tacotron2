#!/usr/bin/env python

# Created on 2019/01
# Author: Kaituo XU (NPU-ASLP)

import argparse
import os

import librosa
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from audio_process import inv_spectrogram, save_wav
from model import FeaturePredictNet
from text_process import text_to_sequence

plt.switch_backend('agg')


parser = argparse.ArgumentParser('Synthesis speech using Tacotron2')
parser.add_argument('--model_path', type=str, required=True, help='Path to model file created by training')
parser.add_argument('--text_file', type=str, default=None, help='Text file each line is one utterance')
parser.add_argument('--csv_file', type=str, default=None, help='Csv file each line is one utterance')
parser.add_argument('--out_dir', type=str, default='exp/result', help='Directory putting synthesis wav files')
parser.add_argument('--show_spect', type=int, default=0, help='Whether show spectrogram')
parser.add_argument('--show_attn', type=int, default=0, help='Whether show attention distribution')
parser.add_argument('--use_cuda', type=int, default=0, help='Whether use GPU to synthesis speech')


def synthesis(args):
    if args.text_file is None and args.csv_file is None:
        print("Must provide text_file or csv_file! When providing csv_file, "
              "text_file is ignored.")
    if args.csv_file:
        args.text_file = args.csv_file

    # Load model
    model = FeaturePredictNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    os.makedirs(args.out_dir, exist_ok=True)

    # Synthesis
    with torch.no_grad():
        with open(args.text_file, 'r') as text_file:
            for i, text in enumerate(text_file.readlines()):
                # Prepare input
                filename = str(i)
                if args.csv_file:
                    filename, text_unnorm, text = text.split('|')
                print(i, text)
                text = torch.LongTensor(text_to_sequence(text)).unsqueeze(0)
                input_length = torch.LongTensor([text.size(-1)])
                if args.use_cuda:
                    text, input_length = text.cuda(), input_length.cuda()
                # Inference
                outputs = model.inference(text, input_length)
                feat_outputs, feat_residual_outputs, _, attention_weights = outputs
                feat_pred = feat_outputs + feat_residual_outputs
                # Visualize spectrogram
                if args.show_spect:
                    spect_file = os.path.join(args.out_dir, f"{filename}_spec")
                    show_spectrogram(feat_pred, spect_file)
                # Visualize attention
                if args.show_attn:
                    attn = attention_weights[0].cpu().numpy().T # [Ti, To]
                    attn_file = os.path.join(args.out_dir, f"{filename}_attn")
                    show_attention(attn, attn_file)
                # Reconstruct audio using Griffin-Lim & Save
                audio = inv_spectrogram(feat_pred[0].cpu().numpy().T)
                audio_path = os.path.join(args.out_dir, f"{filename}.wav")
                save_wav(audio, audio_path)
                # if i == 1: break


def show_spectrogram(feat_pred, spect_file):
    plt.clf()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.imshow(feat_pred[0].cpu().numpy().T, aspect='equal', origin='lower')
    plt.savefig(spect_file)#, dpi=1000)


def show_attention(attn, attn_file):
    # To make colorbar same height with attention figure:
    plt.clf()
    plt.xlabel('Output')
    plt.ylabel('Input')
    plt.title('Attention')
    ax = plt.gca()
    im = ax.imshow(attn, aspect='equal', origin='lower')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(attn_file)#, dpi=1000)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    synthesis(args)
