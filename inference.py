from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import sys
sys.path.append('G_MDCTCodec')
import argparse
import json
import torch
import torchaudio
from models_codec import Encoder,Decoder
from datasets import preprocess
import torch.nn.functional as F
from utils import AttrDict
from dataset import mel_spectrogram, mag_pha_stft
from models import ParaGSE
import soundfile as sf
import librosa
import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt
import time

h = None
device = None

def same_len(clean, rir):
    min_len = len(clean)
    clean = clean[:min_len]
    if len(rir) < min_len:
        rir = np.pad(rir, (0,(min_len - len(rir))), mode='constant', constant_values=0)
    rir = rir[:min_len]
    return clean, rir 

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def get_dataset_filelist(input_training_wav_list):
    training_files=[]
    filelist=os.listdir(input_training_wav_list)
    for files in filelist:

        src=os.path.join(input_training_wav_list,files)
        training_files.append(src)
    return training_files

def inference(a,h):
    generator = ParaGSE(code_size=h.codebook_size,num_quantize=h.num_quantize, num_blocks=1).to(device)
    mdct_encoder = Encoder(h).to(device)
    state_dict_encoder = load_checkpoint('G_MDCTCodec/checkpoint/encoder', device)
    mdct_encoder.load_state_dict(state_dict_encoder['encoder'])
    mdct_decoder = Decoder(h).to(device)
    state_dict_decoder = load_checkpoint('G_MDCTCodec/checkpoint/decoder', device)
    mdct_decoder.load_state_dict(state_dict_decoder['decoder'])

    state_dict = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict['generator'])

    test_indexes = get_dataset_filelist(a.test_noise_wav)
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for index in track(test_indexes):
            
            noisy_wav, sr = librosa.load(index, sr=None)
            mix = noisy_wav
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            mix_in = preprocess(noisy_wav.unsqueeze(0), device)
            
            _, noisy_token, _, _, _ = mdct_encoder(mix_in)
            stft_fea = mag_pha_stft(noisy_wav.unsqueeze(0))
            B, _, T = noisy_token.size()

            prob = generator(noisy_token, stft_fea)
            token_g = torch.cat(prob, dim=-1)
            token_g = token_g.reshape(B, T, -1, h.codebook_size)
            token_g = F.softmax(token_g, dim=-1)
            token_g = torch.argmax(token_g, dim=-1).permute(0,2,1)
            
            z, _, _ = mdct_encoder.quantizer.from_codes(token_g)
            audio_g,_ = mdct_decoder(z)
            audio_g = audio_g.cpu().numpy()
            mix, audio_g = same_len(mix.squeeze(), audio_g.squeeze())

            name = os.path.basename(index)
            output_file = os.path.join(a.output_dir, name)

            sf.write(output_file, audio_g.squeeze(), sr, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_noise_wav', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--checkpoint_file', default='')
    a = parser.parse_args()

    config_file = 'config.json'
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    

    inference(a,h)


if __name__ == '__main__':
    main()


