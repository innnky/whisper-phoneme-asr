import time
import os
import random

import librosa
import numpy as np
import torch
import torch.utils.data

from modules import commons
from modules.symbols import phone_to_int
from utils import load_filepaths_and_text


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        audiopaths_sid_text_new = []
        lengths = []
        skip_num = 0
        for _id, spk, ph, tone, duration in self.audiopaths_sid_text:
            unit_path = f'dataset/wav/{spk}/{_id}.unit.pt'
            if not os.path.exists(unit_path):
                skip_num+=1
                continue
            audiopaths_sid_text_new.append([unit_path, ph, tone, duration])
            lengths.append(torch.load(unit_path).shape[-1])
        print("skip:", skip_num, "samples！")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths


    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        unit_path, ph, tone, duration = audiopath_sid_text
        unit = torch.load(unit_path)
        duration = self.get_duration(duration, unit)
        text, tone = self.get_text(ph, tone, duration)
        return (text, tone, unit)

    def get_duration(self, duration, unit):
        duration = [int(i) for i in duration.split(" ")]
        sum_dur = sum(duration)
        sub = sum_dur - unit.shape[-1]
        assert abs(sub) < 2
        duration[-1] -= sub
        assert sum(duration) == unit.shape[-1]
        return duration

    def get_text(self, ph, tone, duration):
        text_norm = [phone_to_int[i] for i in ph.split(" ")]
        tone = [int(i) for i in tone.split(" ")]
        text_norm_res = []
        tone_res = []
        for idx, du in enumerate(duration):
            text_norm_res += [text_norm[idx]] * du
            tone_res += [tone[idx]] * du
        text_norm = torch.LongTensor(text_norm_res)
        tone = torch.LongTensor(tone_res)
        assert text_norm.shape[0] == sum(duration)

        return text_norm, tone

    def get_sid(self, sid):
        sid = self.spk_map[sid]
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].size(2) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])

        text_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        unit_padded = torch.FloatTensor(len(batch), batch[0][2].size(1), max_text_len)
        text_padded.zero_()
        tone_padded.zero_()
        unit_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            tone = row[1]
            tone_padded[i, :tone.size(0)] = tone

            unit = row[2]
            unit_padded[i, :, :unit.size(2)] = unit[0, :, :]



        return text_padded, text_lengths, tone_padded, unit_padded

