"""
Logic:
- Dataset includes a lot of samples, where each sample has an index.
- Sampler decide how to use samples' indexes, e.g. sequentially or randomly.
- BatchSampler use Sampler to generate indexes of a minibatch.
"""
import csv

import librosa
import pandas as pd
import torch
import torch.utils.data as data
from torch._six import int_classes as _int_classes
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler

from audio_process import load_wav


class LJSpeechDataset(data.Dataset):
    
    def __init__(self, path, text_transformer=None, audio_transformer=None,
                 sample_rate=22050, sort=True):
        self.path = path
        self.metadata = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                    names=['wav', 'text', 'norm_text'],
                                    usecols=['wav', 'norm_text'],
                                    quoting=csv.QUOTE_NONE)  # not ignore quote in string
        self.metadata.dropna(inplace=True)  # Actually, nothing to drop
        self.text_transformer = text_transformer
        self.audio_transformer = audio_transformer
        self.sample_rate = sample_rate
        if sort:
            self.metadata['length'] = self.metadata['wav'].apply(
                    lambda x: librosa.get_duration(filename=f'{path}/wavs/{x}.wav'))
            self.metadata.sort_values(by=['length'], inplace=True, ascending=False)

    def __getitem__(self, index):
        """
        Returns:
            text (torch.IntTensor): an id sequence, [T]
            audio (torch.FloatTensor): a feature sequence, [D, T]
        """
        text = self._get_text(index)
        audio = self._get_audio(index)
        # print(text.size(), audio.size())
        return text, audio

    def __len__(self):
        return len(self.metadata)

    def _get_text(self, index):
        text = self.metadata.iloc[index]['norm_text']
        if self.text_transformer:
            text = self.text_transformer(text)
            text = torch.IntTensor(text)
        return text

    def _get_audio(self, index):
        filename = self.metadata.iloc[index]['wav']
        audio = load_wav(f'{self.path}/wavs/{filename}.wav', self.sample_rate)
        if self.audio_transformer:
            audio = self.audio_transformer(audio)
        return torch.FloatTensor(audio)


class RandomBucketBatchSampler(object):
    """Yields of mini-batch of indices, sequential within the batch, random between batches.
    
    I.e. it works like bucket, but it also supports random between batches.
    Helpful for minimizing padding while retaining randomness with variable length inputs.

    Args:
        data_source (Dataset): dataset to sample from.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = SequentialSampler(data_source) # impl sequential within the batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches() # impl random between batches

    def _make_batches(self):
        indices = [i for i in self.sampler]
        batches = [indices[i:i+self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and len(self.sampler) % self.batch_size > 0:
            random_indices = torch.randperm(len(batches)-1).tolist() + [len(batches)-1]
        else:
            random_indices = torch.randperm(len(batches)).tolist()
        return [batches[i] for i in random_indices]

    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)


class TextAudioCollate(object):
    """Another way to implement collate_fn passed to DataLoader.
    Use class but not function because this is easier to pass some parameters.
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Process one mini-batch samples, such as sorting and padding.
        Args:
            batch: a list of (text sequence, audio feature sequence)
        Returns:
            text_padded: [N, Ti]
            input_lengths: [N]
            mel_padded: [N, To, D]
            gate_padded: [N, To]
            output_lengths: [N]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)]= text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
        mel_padded = mel_padded.transpose(1, 2)  # [N, To, D]

        # print(text_padded.size(), input_lengths.size(), mel_padded.size(), gate_padded.size(), output_lengths.size())
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


if __name__ == '__main__':
    # Test LJSpeechDataset
    import sys
    path = sys.argv[1]
    dataset = LJSpeechDataset(path)
    print(len(dataset))
    for i, data in enumerate(dataset):
        text, audio = data
        print(i)
        print(text)
        print(audio)
        if i == 10:
            break

    # Test RandomBucketBatchSampler
    torch.manual_seed(123)
    print(list(RandomBucketBatchSampler(range(10), 2, False)))
    print(list(RandomBucketBatchSampler(range(10), 2, True)))
    print(list(RandomBucketBatchSampler(range(10), 3, False)))
    print(list(RandomBucketBatchSampler(range(10), 3, True)))
    # output:
    # [[4, 5], [0, 1], [2, 3], [6, 7], [8, 9]]
    # [[0, 1], [6, 7], [4, 5], [8, 9], [2, 3]]
    # [[9], [6, 7, 8], [0, 1, 2], [3, 4, 5]]
    # [[3, 4, 5], [0, 1, 2], [6, 7, 8], [9]]
