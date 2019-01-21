# encoding: utf-8
import audio
import audio_hparams as hparams
import numpy as np
import os

if __name__ == '__main__':
    data_foler = "data"
    wavs = [os.path.join(data_foler, file[:-4]) for file in os.listdir(data_foler) if file.endswith(".wav")]
    outputs_py = [file + ".py.gen.wav" for file in wavs]
    outputs_mel = [file + ".mel.wav" for file in wavs]
    wavs = [audio.load_wav(wav_path + ".wav", hparams.sample_rate) for wav_path in wavs]
    spectrogram = [audio.spectrogram(wav).astype(np.float32) for wav in wavs]
    print("Linear spectrograms dim: ")
    print(spectrogram[0].shape)
    # --------------------------------- librosa Version ---------------------------------
    # convert back
    gens = [audio.inv_spectrogram(s) for s in spectrogram]

    for gen, output in zip(gens, outputs_py):
        audio.save_wav(gen, output)

    # --------------------------------- librosa Version ---------------------------------
    mel_spectrogram = [audio.melspectrogram(wav).astype(np.float32) for wav in wavs]
    # convert back
    gens = [audio.inv_melspectrogram(s) for s in mel_spectrogram]

    for gen, output in zip(gens, outputs_mel):
        audio.save_wav(gen, output)

    print("Done!")
