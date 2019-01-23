#!/usr/bin/env python
from torch.utils.data import DataLoader

from audio_process import melspectrogram, spectrogram
from data import LJSpeechDataset, RandomBucketBatchSampler, TextAudioCollate
from text_process import sequence_to_text, text_to_sequence


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    shuffle = True

    dataset = LJSpeechDataset(path, text_transformer=text_to_sequence,
                              audio_transformer=spectrogram)
    batch_sampler = RandomBucketBatchSampler(dataset, batch_size=5, drop_last=False)
    collate_fn = TextAudioCollate()
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                            collate_fn=collate_fn)

    print(len(dataset))
    for i, data in enumerate(dataset):
        text, audio = data
        print(i, len(text), audio.shape)
        print(text)
        print(audio)
        if i == 100:
            break

    print("*"*80)
    for i, batch in enumerate(dataloader):
        print(i, batch)
        if i == 10:
            break
