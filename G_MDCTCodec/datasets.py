import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import librosa
from torch import nn, view_as_real, view_as_complex
import scipy

def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):

    mel = librosa_mel_fn(sr=sampling_rate,n_fft= n_fft,n_mels= num_mels,fmin= fmin,fmax= fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec #[batch_size,n_fft/2+1,frames]

def amp_pha_specturm(y, n_fft, hop_size, win_size):

    hann_window=torch.hann_window(win_size).to(y.device)

    stft_spec=torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,center=True) #[batch_size, n_fft//2+1, frames, 2]

    rea=stft_spec[:,:,:,0] #[batch_size, n_fft//2+1, frames]
    imag=stft_spec[:,:,:,1] #[batch_size, n_fft//2+1, frames]

    log_amplitude=torch.log(torch.abs(torch.sqrt(torch.pow(rea,2)+torch.pow(imag,2)))+1e-5) #[batch_size, n_fft//2+1, frames]
    phase=torch.atan2(imag,rea) #[batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, rea, imag

# def get_dataset_filelist(input_training_wav_list,input_validation_wav_list):

#     with open(input_training_wav_list, 'r') as fi:
#         training_files = [x for x in fi.read().split('\n') if len(x) > 0]

#     with open(input_validation_wav_list, 'r') as fi:
#         validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

#     return training_files, validation_files

def get_dataset_filelist(input_training_wav_list, input_validation_wav_list):
    train_files=[]
    filelist=os.listdir(input_training_wav_list)
    for files in filelist:

        src=os.path.join(input_training_wav_list,files)
        train_files.append(src)

    valid_files=[]
    filelist=os.listdir(input_validation_wav_list)
    for files in filelist:
        src=os.path.join(input_validation_wav_list,files)
        valid_files.append(src)

    return train_files, valid_files

class MDCT(nn.Module):
    """
    Modified Discrete Cosine Transform (MDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.windows.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(-1j * torch.pi * torch.arange(frame_len) / frame_len)
        post_twiddle = torch.exp(-1j * torch.pi * n0 * (torch.arange(N) + 0.5) / N)
        # view_as_real: NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply the Modified Discrete Cosine Transform (MDCT) to the input audio.

        Args:
            audio (Tensor): Input audio waveform of shape (B, T), where B is the batch size
                and T is the length of the audio.

        Returns:
            Tensor: MDCT coefficients of shape (B, L, N), where L is the number of output frames
                and N is the number of frequency bins.
        """
        if self.padding == "center":
            audio = torch.nn.functional.pad(audio, (self.frame_len // 2, self.frame_len // 2))
        elif self.padding == "same":
            # hop_length is 1/2 frame_len
            audio = torch.nn.functional.pad(audio, (self.frame_len // 4, self.frame_len // 4))
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        x = audio.unfold(-1, self.frame_len, self.frame_len // 2)
        N = self.frame_len // 2
        x = x * self.window.expand(x.shape)
        X = torch.fft.fft(x * view_as_complex(self.pre_twiddle).expand(x.shape), dim=-1)[..., :N]
        res = X * view_as_complex(self.post_twiddle).expand(X.shape) * np.sqrt(1 / N)
        return torch.real(res) * np.sqrt(2)


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.windows.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, L, N), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(Y * view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1)
        y = torch.real(y * view_as_complex(self.post_twiddle).expand(y.shape)) * np.sqrt(N) * np.sqrt(2)
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels_for_loss,
                 hop_size, win_size, sampling_rate, ratio, split=True, shuffle=True, n_cache_reuse=1,
                 device=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.ratio = ratio
        self.split = split
        self.n_fft = n_fft
        self.num_mels_for_loss = num_mels_for_loss
        self.hop_size = hop_size
        self.win_size = win_size
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio = load_wav(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio) #[T]
        audio = audio.unsqueeze(0) #[1,T]

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start: audio_start + self.segment_size] #[1,T]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        else:
            if audio.size(1) - (audio.size(1)//self.hop_size) * self.hop_size > 0:
                audio = audio[:, 0:-(audio.size(1) - (audio.size(1)//self.hop_size) * self.hop_size)]

            if audio.size(1) - (audio.size(1)//self.hop_size) * self.hop_size < 0:
                audio = audio[:, 0:-(audio.size(1) - (audio.size(1)//self.hop_size) * self.hop_size + self.hop_size)]

            if (audio.size(1)//self.hop_size + 1) % self.ratio > 0:
            	audio = audio[:,0: (((audio.size(1)//self.hop_size + 1) // self.ratio) * self.ratio - 1) * self.hop_size]

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels_for_loss,
                                   self.sampling_rate, self.hop_size, self.win_size, 0, None,
                                   center=True) #[1,n_fft/2+1,frames]
        MDCT_operation=MDCT(self.hop_size*2)
        MDCT_coff=MDCT_operation(audio)
        # print('audio')
        # print(audio.shape)
        # print(MDCT_coff.shape)
        MDCT_coff=MDCT_coff.permute(0,2,1)
        #log_amplitude, phase, rea, imag = amp_pha_specturm(audio, self.n_fft, self.hop_size, self.win_size) #[1,n_fft/2+1,frames]


        return (MDCT_coff.squeeze(), audio.squeeze(0), mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)


def preprocess(data, device):
    MDCT_operation=MDCT(80).to(device)
    mdct = MDCT_operation(data)
    mdct = mdct.squeeze(1)
    mdct=mdct.permute(0,2,1)
    return mdct