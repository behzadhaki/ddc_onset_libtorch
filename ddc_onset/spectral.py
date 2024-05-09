from typing import Optional
import pathlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SAMPLE_RATE, FRAME_RATE
from .paths import WEIGHTS_DIR

_FEATS_HOP = SAMPLE_RATE // FRAME_RATE
_FFT_FRAME_LENGTHS = [1024, 2048, 4096]
_NUM_MEL_BANDS = 80
_LOG_EPS = 1e-16


class SpectrogramExtractor(nn.Module):
    """Extracts log-Mel spectrograms from waveforms."""

    def __init__(self):
        super().__init__()

        self._FEATS_HOP = SAMPLE_RATE // FRAME_RATE
        self._LOG_EPS = 1e-16

        for i in _FFT_FRAME_LENGTHS:
            window = nn.Parameter(
                torch.zeros(i, dtype=torch.float32, requires_grad=False),
                requires_grad=False,
            )
            if i == 1024:
                self.window_1024 = window
            elif i == 2048:
                self.window_2048 = window
            elif i == 4096:
                self.window_4096 = window

            mel = nn.Parameter(
                torch.zeros(
                    ((i // 2) + 1, 80), dtype=torch.float32, requires_grad=False
                ),
                requires_grad=False,
            )

            if i == 1024:
                self.mel_1024 = mel
            elif i == 2048:
                self.mel_2048 = mel
            elif i == 4096:
                self.mel_4096 = mel

        self.load_state_dict(
            torch.load(pathlib.Path(WEIGHTS_DIR, "spectrogram_extractor.bin"))
        )

    @torch.jit.export
    def forward(
        self, x: torch.Tensor, frame_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Extracts an 80-bin log-Mel spectrogram from a waveform with 3 different FFT frame lengths.

        Args:
            x: 44.1kHz waveforms as float32 [batch_size, num_samples].
            frame_chunk_size: Number of frames to process at a time. If None, process all frames at once.
        Returns:
            Log mel spectrograms as float32 [batch_size, num_frames, num_mel_bands (80), num_fft_frame_lengths (3)].
        """
        # NOTE: This was originally implemented as [samps, batch] but [batch, samps] is better type signature.
        waveform = x.transpose(1, 0)
        feats = []

        feats_num_timesteps = 0

        for i in range(3):
            # _FFT_FRAME_LENGTHS = [1024, 2048, 4096]
            fft_frame_length = int(1024 * 2 ** i)
            # Pad waveform to center spectrogram
            fft_frame_length_half = fft_frame_length // 2
            waveform_padded = F.pad(waveform, (0, 0, fft_frame_length_half, 0))
            waveform_padded_len = waveform_padded.shape[0]




            # Chunk up waveform to save memory at cost of some efficiency
            if frame_chunk_size is None:
                chunk_hop = waveform_padded_len
            else:
                chunk_hop = frame_chunk_size * self._FEATS_HOP

            chunk_feats = []
            for c in range(0, waveform_padded_len, chunk_hop):
                frames = []
                for s in range(c, min(c + chunk_hop, waveform_padded_len), self._FEATS_HOP):
                    # Slice waveform into frames
                    # TODO: Change this to range(0, waveform.shape[0], self._FEATS_HOP) to make num feats = ceil(num_samps / 441)? Make sure frames are equal after doing this
                    frame = waveform_padded[s : s + fft_frame_length]
                    padding_amt = fft_frame_length - frame.shape[0]
                    if padding_amt > 0:
                        frame = F.pad(frame, (0, 0, padding_amt, 0))
                    frames.append(frame)
                frames = torch.stack(frames, dim=0)

                # Apply window
                # Params for this FFT size
                if fft_frame_length == 1024:
                    frames *= self.window_1024.view(1, -1, 1)
                    # mel_w = self.mel_1024
                elif fft_frame_length == 2048:
                    frames *= self.window_2048.view(1, -1, 1)
                    # mel_w = self.mel_2048
                elif fft_frame_length == 4096:
                    frames *= self.window_4096.view(1, -1, 1)
                    # mel_w = self.mel_4096

                # Copying questionable "zero phase" windowing nonsense from essentia
                # https://github.com/MTG/essentia/blob/master/src/algorithms/standard/windowing.cpp#L85
                frames_half_one = frames[:, :fft_frame_length_half]
                frames_half_two = frames[:, fft_frame_length_half:]
                frames = torch.cat([frames_half_two, frames_half_one], dim=1)

                # Perform FFT
                frames = torch.transpose(frames, 1, 2)
                spec = torch.fft.rfft(frames)

                # Compute power spectrogram
                spec_r = torch.real(spec)
                spec_i = torch.imag(spec)
                pow_spec = torch.pow(spec_r, 2) + torch.pow(spec_i, 2)

                # Compute mel spectrogram
                if fft_frame_length == 1024:
                    mel_spec = torch.matmul(pow_spec, self.mel_1024)
                    # Compute log mel spectrogram
                    log_mel_spec = torch.log(mel_spec + self._LOG_EPS)
                    log_mel_spec = torch.transpose(log_mel_spec, 1, 2)
                    chunk_feats.append(log_mel_spec)
                elif fft_frame_length == 2048:
                    mel_spec = torch.matmul(pow_spec, self.mel_2048)
                    # Compute log mel spectrogram
                    log_mel_spec = torch.log(mel_spec + self._LOG_EPS)
                    log_mel_spec = torch.transpose(log_mel_spec, 1, 2)
                    chunk_feats.append(log_mel_spec)
                elif fft_frame_length == 4096:
                    mel_spec = torch.matmul(pow_spec, self.mel_4096)
                    # Compute log mel spectrogram
                    log_mel_spec = torch.log(mel_spec + self._LOG_EPS)
                    log_mel_spec = torch.transpose(log_mel_spec, 1, 2)
                    chunk_feats.append(log_mel_spec)
                else:
                    raise ValueError("fft_frame_length must be 1024, 2048, or 4096")

            chunk_feats = torch.cat(chunk_feats, dim=0)

            # Cut off extra chunk_feats for larger FFT lengths
            # TODO: Don't slice these chunk_feats to begin with
            if i == 0:
                feats_num_timesteps = chunk_feats.shape[0]
            else:
                assert chunk_feats.shape[0] > feats_num_timesteps
                chunk_feats = chunk_feats[:feats_num_timesteps]

            feats.append(chunk_feats)

        feats = torch.stack(feats, dim=2)

        # [feats..., batch] -> [batch, feats...]
        return feats.permute(3, 0, 1, 2)

    @torch.jit.ignore
    def serialize(self, save_folder, filename=None):

        os.makedirs(save_folder, exist_ok=True)

        if filename is None:
            filename = f'SpectrogramExtractor.pt'
        is_train = self.training
        self.eval()
        save_path = os.path.join(save_folder, filename)

        scr = torch.jit.script(self)
        # save model
        with open(save_path, "wb") as f:
            torch.jit.save(scr, f)

        if is_train:
            self.train()