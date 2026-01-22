from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import glob
import os
import argparse
import json
import torch
from utils import AttrDict
from datasets import amp_pha_specturm, load_wav
from models_500bps import Encoder, Decoder
import soundfile as sf
import librosa
import numpy as np
from datasets import MDCT,IMDCT
from time import time
from rich.progress import track
h = None
device = torch.device('cuda:{:d}'.format(0))


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

def plot_spec(spec, label):
    # 绘制语谱图
    # plt.figure(figsize=(8, 8))
    # plt.imshow(spec, aspect='auto', origin='lower')
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(spec, aspect="auto", origin="lower",
                   interpolation='none')

    # fig.canvas.draw()
    # librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='hz')
    # plt.colorbar(format='%+2.0f dB')

    # plt.xticks([0, 480, 960, 1440, 1920, 2400, 2880], ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'], fontproperties = 'Times New Roman', size = 28)
    # plt.yticks([0, 170, 256, 341, 512, 1024], ['0', '4', '6', '8', '12', '24'], fontproperties = 'Times New Roman', size = 28)
    # 22k
    # plt.xticks([0, 440, 880, 1320, 1760, 2200, 2640, 3080, 3520], ['0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'], fontproperties = 'Times New Roman', size = 28)
    # plt.yticks([0, 373, 745, 1024], ['0', '4', '8', '11'], fontproperties = 'Times New Roman', size = 28)
    # 48k
    plt.xticks([0, 600, 1200,1800, 2400], ['0', '0.5', '1.0','1.5', '2.0'], fontproperties = 'Times New Roman', size = 18)
    plt.yticks([0, 6.6, 10, 13.6, 20, 40], ['0', '4', '6', '8', '12', '24'], fontproperties = 'Times New Roman', size = 18)
    plt.xlabel('Time(s)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Frequency(kHz)', fontdict={'family': 'Times New Roman', 'size': 18})
    # 16k
    # plt.xticks([0, 320, 640, 960, 1280, 1600, 1920, 2240, 2560], ['0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'], fontproperties = 'Times New Roman', size = 28)
    # plt.yticks([0, 256, 512, 1024], ['0', '2', '4', '8'], fontproperties = 'Times New Roman', size = 28)
    # plt.xlabel('Time(s)', fontdict={'family': 'Times New Roman', 'size': 36})
    # plt.ylabel('Frequency(kHz)', fontdict={'family': 'Times New Roman', 'size': 36})
    # plt.xticks([])  # 去掉x轴
    # plt.yticks([])  # 去掉y轴
    # plt.axis('off')
    plt.tight_layout()
    plt.savefig(label + '.png', dpi=500)

def plot_and_save_waveform(audio_file, save_path):
    y, sr = librosa.load(audio_file)
    plt.figure(figsize=(10, 8))
    librosa.display.waveshow(y)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlabel('Time(s)', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Amplitude', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.savefig(save_path)

def tensor_to_text(tensor, filename="output.txt"):
    # 将Tensor转换为numpy数组
    if isinstance(tensor, torch.Tensor):
        array = tensor.cpu().detach().numpy()
    elif isinstance(tensor, np.ndarray):
        array = tensor
    else:
        raise TypeError("Input should be either PyTorch Tensor or Numpy array.")

    # 只保留第一维
    array = array[3]

    # 设置numpy打印选项，禁止省略大数组
    np.set_printoptions(threshold=np.inf)

    # 将数组转换为字符串，移除 []
    str_array = str(array).replace('[', '').replace(']', '')

    # 按行分割，然后写入文件，每行末尾添加换行符
    with open(filename, 'a') as f:
        for line in str_array.split('\n'):
            f.write(line + '\n')

# def get_dataset_filelist(input_validation_wav_list):

#     with open(input_validation_wav_list, 'r') as fi:
#         validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

#     return validation_files

def get_dataset_filelist(input_training_wav_list):
    training_files=[]
    filelist=os.listdir(input_training_wav_list)
    for files in filelist:

        src=os.path.join(input_training_wav_list,files)
        training_files.append(src)
    return training_files

def inference(h):
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)

    state_dict_encoder = load_checkpoint(h.checkpoint_file_load_Encoder, device)
    encoder.load_state_dict(state_dict_encoder['encoder'])
    state_dict_decoder = load_checkpoint(h.checkpoint_file_load_Decoder, device)
    decoder.load_state_dict(state_dict_decoder['decoder'])

    # filelist = sorted(os.listdir(h.test_input_wavs_dir))
    filelist = get_dataset_filelist(h.test_input_wavs_dir)

    # os.makedirs(h.test_latent_output_dir, exist_ok=True)
    os.makedirs(h.test_wav_output_dir, exist_ok=True)

    encoder.eval()
    decoder.eval()

    total_len=0.
    total_time=0.

    MDCT_operation=MDCT(80).to(device)

    with torch.no_grad():
        for filename in track(filelist):

            raw_wav, _ = librosa.load(filename, sr=h.sampling_rate, mono=True)
            raw_wav = torch.FloatTensor(raw_wav).to(device)
           # print(raw_wav.shape)
            mdct = MDCT_operation(raw_wav.unsqueeze(0))
            mdct=mdct.permute(0,2,1)
            #logamp, pha, _, _ = amp_pha_specturm(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)
            total_time-=time()
            latent,z,_,_,_ = encoder(mdct)
            # z1,_,_ = encoder.quantizer.from_codes(z)
            y_g,mdct_g = decoder(latent)
            #print('mdct',mdct.shape)
            # mdct_spec_1 = librosa.amplitude_to_db(mdct.squeeze().cpu())
            # mdct_spec_2 = librosa.amplitude_to_db(mdct_g.squeeze().cpu())
            # plot_spec(mdct_spec_1, '/disk1/xhjiang/pic/gtmdct48k_new')
            # plot_spec(mdct_spec_2, '/disk1/xhjiang/pic/mdct48k_new')
            # plot_and_save_waveform('/disk1/xhjiang/MDCT2_noMPD_MDCTD/6kbps_best_output_wav/p360_001.wav','/disk1/xhjiang/pic/synwaveform_new.png')
            # plot_and_save_waveform('/disk1/xhjiang/VCTK48K/wav48/test/p360_001.wav','/disk1/xhjiang/pic/gtwaveform_new.png')
            # total_time+=time()
            # latent = latent.squeeze()
            audio = y_g.squeeze()
            # latent = latent.cpu().numpy()
            audio = audio.cpu().numpy()
            total_len += len(audio)
            # np.save(os.path.join(h.test_latent_output_dir, filename.split('.')[0]+'.npy'), latent)
            name = os.path.basename(filename)
            sf.write(os.path.join(h.test_wav_output_dir, name), audio, h.sampling_rate,'PCM_16')
    # total_len = total_len/48000.

    # generation_time_per_second = total_time/total_len

    # log="The time of generating 1s speech is {} seconds.\n"
    # log=log.format(generation_time_per_second)
    # print(log)

def main():
    print('Initializing Inference Process..')

    config_file = 'config_codec.json'

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

    inference(h)


if __name__ == '__main__':
    main()

