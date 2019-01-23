# encoding: utf-8
import audio_process as audio
import hyperparams as hparams
import numpy as np
import os

if __name__ == '__main__':
    data_foler = "data"
    wavs = [os.path.join(data_foler, filename) for filename in ["000001", "000002"]]
    outputs_linear = [file + ".linear.wav" for file in wavs]
    outputs_mel = [file + ".mel.wav" for file in wavs]
    wavs = [audio.load_wav(wav_path + ".wav", hparams.sample_rate) for wav_path in wavs]

    spectrogram = [audio.spectrogram(wav).astype(np.float32) for wav in wavs]
    print("Linear spectrograms dim: ")
    print(spectrogram[0].shape)
    gens = [audio.inv_spectrogram(s) for s in spectrogram]
    for gen, output in zip(gens, outputs_linear):
        audio.save_wav(gen, output)

    mel_spectrogram = [audio.melspectrogram(wav).astype(np.float32) for wav in wavs]
    print("Mel spectrograms dim: ")
    print(mel_spectrogram[0].shape)
    gens = [audio.inv_melspectrogram(s) for s in mel_spectrogram]
    for gen, output in zip(gens, outputs_mel):
        audio.save_wav(gen, output)

    print("Done!")
