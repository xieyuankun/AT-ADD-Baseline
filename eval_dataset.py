#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import os
import librosa
from torch.utils.data.dataloader import default_collate
from typing import Tuple

torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, str]


def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


def pad_dataset(wav, audio_length=64600):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = audio_length

    if waveform_len >= cut:
        waveform = waveform[:cut]
    else:
        num_repeats = int(cut / waveform_len) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    return waveform


class atadd_eval_dataset(Dataset):
    def __init__(self, path_to_audio, audio_length=64600, exts=(".flac", ".wav")):
        super(atadd_eval_dataset, self).__init__()

        self.path_to_audio = path_to_audio
        self.audio_length = audio_length

        self.all_files = sorted([
            f for f in os.listdir(self.path_to_audio)
            if f.lower().endswith(exts)
        ])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)

        return waveform, filename

    def collate_fn(self, samples):
        return default_collate(samples)


if __name__ == "__main__":
    dataset = atadd_eval_dataset(
        path_to_audio="yourpath/atadd/T2/eval"
    )
    print("dataset size:", len(dataset))
    print(dataset[0][0].size(), dataset[0][1])
