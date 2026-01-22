import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import sys
sys.path.append('G_MDCTCodec')
import time
import torchaudio
import argparse
from models_codec import Encoder,Decoder
from datasets import preprocess
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import Dataset, get_dataset_filelist, mel_spectrogram, mag_pha_stft, get_all_audio_files
from models import ParaGSE
from utils import AttrDict, build_env, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from scheduler import WarmupCosineScheduler

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = ParaGSE(input_dim=513*2, code_size=h.codebook_size, num_quantize=h.num_quantize, num_blocks=1)
    mdct_encoder = Encoder(h).to(device)
    state_dict_encoder = load_checkpoint('G_MDCTCodec/checkpoint/encoder', device)
    mdct_encoder.load_state_dict(state_dict_encoder['encoder'])
    mdct_decoder = Decoder(h).to(device)
    state_dict_decoder = load_checkpoint('G_MDCTCodec/checkpoint/decoder', device)
    mdct_decoder.load_state_dict(state_dict_decoder['decoder'])

    for p in mdct_encoder.parameters():
        p.requires_grad = False
    for p in mdct_decoder.parameters():
        p.requires_grad = False


    if rank == 0:
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        print('Total Parameters: {:.3f}M'.format(num_params/1e6))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    generator = generator.to(device)

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)

    training_clean_indexes, training_noise_indexes = get_dataset_filelist(a.input_train_clean_list, a.input_train_noise_list)
    validation_clean_indexes, validation_noise_indexes = get_dataset_filelist(a.input_validation_clean_list, a.input_validation_noise_list)

    trainset = Dataset(training_clean_indexes, training_noise_indexes, h.segment_size,
                       split=True, n_cache_reuse=0, shuffle=False , device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    scheduler_g = WarmupCosineScheduler(
        optim_g,
        warmup_steps=h.warmup_steps,
        total_steps=h.training_epochs * len(train_loader),
        base_lr=h.learning_rate,
        min_lr=1e-5,         
        num_cycles=0.5,      
        last_epoch=steps
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
    
    if rank == 0:
        validset = Dataset(validation_clean_indexes, validation_noise_indexes, h.segment_size,
                           split=False, shuffle=False, n_cache_reuse=0, device=device)
        
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    
    generator.train()
    mdct_encoder.eval()
    mdct_decoder.eval()
    torch.cuda.empty_cache()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            if rank == 0:
                start_b = time.time()
            clean_audio, mix_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            mix_audio = mix_audio.to(device, non_blocking=True)

            clean_audio = clean_audio.unsqueeze(1)
            mix_audio = mix_audio.unsqueeze(1)

            clean_in = preprocess(clean_audio, device)
            _, clean_token_dac, _, _, _ = mdct_encoder(clean_in)    # (B,Q,T)
            clean_token = clean_token_dac.permute(0,2,1)
            clean_token_loss = [clean_token[:,:,k].reshape(-1) for k in range(h.num_quantize)]
            
            mix_in = preprocess(mix_audio, device)
            _, noisy_token, _, _, _ = mdct_encoder(mix_in)
            stft_fea = mag_pha_stft(mix_audio.squeeze(1))
            prob = generator(noisy_token, stft_fea)

            # Generator
            optim_g.zero_grad()
            loss_list = []
            for i in range(h.num_quantize):
                loss_c = F.cross_entropy(prob[i].reshape(-1, h.codebook_size) / 1.0, clean_token_loss[i])
                loss_list.append(loss_c)
            loss = sum(loss_list) / h.num_quantize
            loss.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % h.stdout_interval == 0:
                    with torch.no_grad():
                        Q1_error = loss_list[0].item()
                        Q2_error = loss_list[1].item()
                        Q3_error = loss_list[2].item()
                        Q4_error = loss_list[3].item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Q1 Loss: {:4.3f}, Q2 Loss: {:4.3f}, Q3 Loss: {:4.3f}, Q4 Loss: {:4.3f}, s/b : {:4.3f}'.
                           format(steps, loss, Q1_error, Q2_error, Q3_error, Q4_error, time.time() - start_b))

                # checkpointing
                if steps % h.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {
                                     'optim_g': optim_g.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % h.summary_interval == 0:
                    # visualizer.update_step(steps)
                    sw.add_scalar("Training/Generator Loss", loss, steps)
                    sw.add_scalar("Training/Q1 Loss", Q1_error, steps)
                    sw.add_scalar("Training/Q2 Loss", Q2_error, steps)
                    sw.add_scalar("Training/Q3 Loss", Q3_error, steps)
                    sw.add_scalar("Training/Q4 Loss", Q4_error, steps)


                # Validation
                if steps % h.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_cross_err_tot = 0
                    val_dnsmos_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, mix_audio = batch
                            clean_audio = clean_audio.to(device, non_blocking=True)
                            mix_audio = mix_audio.to(device, non_blocking=True)
                
                            clean_audio = clean_audio.unsqueeze(1)
                            mix_audio = mix_audio.unsqueeze(1)

                            clean_in = preprocess(clean_audio, device)
                            
                            _, clean_token_dac, _, _, _ = mdct_encoder(clean_in)    # (B,Q,T)
                            clean_token = clean_token_dac.permute(0,2,1)
                            clean_token_loss = [clean_token[:,:,k].reshape(-1) for k in range(h.num_quantize)]
                            mask = torch.ones_like(clean_token, dtype=torch.bool)

                            mix_in = preprocess(mix_audio, device)
                            _, noisy_token, _, _, _ = mdct_encoder(mix_in)
                            stft_fea = mag_pha_stft(mix_audio.squeeze(1))
                            prob = generator(noisy_token, stft_fea)

                            loss_list = []
                            for i in range(h.num_quantize):
                                loss_c = F.cross_entropy(prob[i].reshape(-1, h.codebook_size) / 1.0, clean_token_loss[i])
                                loss_list.append(loss_c)
                            loss_cross = sum(loss_list) / h.num_quantize

                            val_cross_err_tot += loss_cross.item()
                            

                        val_cross_err = val_cross_err_tot / (j+1)
                        sw.add_scalar("Validation/Cross Loss", val_cross_err, steps)
                    generator.train()
            steps += 1
            scheduler_g.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--gpu_num', default=0)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--input_train_clean_list', default='../data')
    parser.add_argument('--input_train_noise_list', default='../data')
    parser.add_argument('--input_validation_clean_list', default='../data')
    parser.add_argument('--input_validation_noise_list', default='../data')
    parser.add_argument('--checkpoint_path', default='../checkpoint/ParaGSE')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
