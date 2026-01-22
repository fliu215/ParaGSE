import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from datasets import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from models_codec import Encoder, Decoder, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, amplitude_loss, phase_loss, STFT_consistency_loss, MultiResolutionDiscriminator
from utils import AttrDict, build_env, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
torch.backends.cudnn.benchmark = True


def train(h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(0))
    encoder = Encoder(h).to(device)
    decoder = Decoder(h).to(device)
    # mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    print("Encoder: ")
    print(encoder)
    print("Decoder: ")
    print(decoder)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_encoder = scan_checkpoint(h.checkpoint_path, 'encoder_')
        cp_decoder = scan_checkpoint(h.checkpoint_path, 'decoder_')
        cp_do = scan_checkpoint(h.checkpoint_path, 'do_')

    steps = 0
    if cp_encoder is None or cp_decoder is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_encoder = load_checkpoint(cp_encoder, device)
        state_dict_decoder = load_checkpoint(cp_decoder, device)
        state_dict_do = load_checkpoint(cp_do, device)
        encoder.load_state_dict(state_dict_encoder['encoder'])
        decoder.load_state_dict(state_dict_decoder['decoder'])
        # mpd.load_state_dict(state_dict_do['mpd'])
        mrd.load_state_dict(state_dict_do['mrd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    optim_g = torch.optim.AdamW(itertools.chain(encoder.parameters(), decoder.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    # for group in optim_g.param_groups:
    #     group['capturable'] = True
 
    # for group in optim_d.param_groups:
    #     group['capturable'] = True

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    # for group in optim_g.param_groups:
    #     group['capturable'] = True
 
    # for group in optim_d.param_groups:
    #     group['capturable'] = True

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                       h.hop_size, h.win_size, h.sampling_rate, h.ratio, n_cache_reuse=0,
                       shuffle=True, device=device)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    validset = Dataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels_for_loss,
                       h.hop_size, h.win_size, h.sampling_rate, h.ratio, False, False, n_cache_reuse=0,
                       device=device)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    encoder.train()
    decoder.train()
    # mpd.train()
    mrd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):

        start = time.time()
        print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            MDCT_coff, y, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            MDCT_coff = torch.autograd.Variable(MDCT_coff.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            latent,_,_,commitment_loss,codebook_loss = encoder(MDCT_coff)

            y_g, MDCT_g_coff = decoder(latent)
            y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                      0, None)
            

            optim_d.zero_grad()

            # y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
            # loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            L_D = loss_disc_s*0.1# + loss_disc_f

            L_D.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # Losses defined on log amplitude spectra
            # L_A = amplitude_loss(logamp, logamp_g)

            # L_IP, L_GD, L_PTD = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
            # # Losses defined on phase spectra
            # L_P = L_IP + L_GD + L_PTD

            # _, _, rea_g_final, imag_g_final = amp_pha_specturm(y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size)
            # L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)
            # L_R = F.l1_loss(rea, rea_g)
            # L_I = F.l1_loss(imag, imag_g)
            # # Losses defined on reconstructed STFT spectra
            # L_S = L_C + 2.25 * (L_R + L_I)

            # y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)
            # loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            # loss_gen_f, losses_gen_f = generator_loss(y_df_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_g)
            L_GAN_G = loss_gen_s*0.1 #+ loss_gen_f
            L_FM = loss_fm_s*0.1# + loss_fm_f
            L_Mel = F.l1_loss(y_mel, y_g_mel)
            L_Mel_L2 = amplitude_loss(y_mel, y_g_mel)         

            L_MDCT = amplitude_loss(MDCT_coff,MDCT_g_coff)

            L_W = L_GAN_G + L_FM + 45 * L_Mel + 45 * L_Mel_L2 + 250 * L_MDCT

            L_G = L_W + codebook_loss*10 +commitment_loss*2.5

            L_G.backward()
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    Mel_error = F.l1_loss(y_mel, y_g_mel).item()
                    Mel_L2_error = amplitude_loss(y_mel, y_g_mel).item()
                    commit_loss = commitment_loss.item()
                    MDCT_loss = amplitude_loss(MDCT_coff,MDCT_g_coff).item()
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, MDCT Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, Mel Spectrogram L2 Loss : {:4.3f}, Commit Loss : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, L_G, MDCT_loss, Mel_error, Mel_L2_error, commit_loss, time.time() - start_b))

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/encoder_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'encoder': encoder.state_dict()})
                checkpoint_path = "{}/decoder_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'decoder': decoder.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, 
                                {'mrd': mrd.state_dict(),
                                 'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                 'epoch': epoch})
                # if steps > h.checkpoint_interval:
                #     checkpoint_path_old = "{}/encoder_{:08d}".format(h.checkpoint_path, steps - h.checkpoint_interval)
                #     os.remove(checkpoint_path_old)
                #     checkpoint_path_old = "{}/decoder_{:08d}".format(h.checkpoint_path, steps - h.checkpoint_interval)
                #     os.remove(checkpoint_path_old)
                #     checkpoint_path_old = "{}/do_{:08d}".format(h.checkpoint_path, steps - h.checkpoint_interval)
                #     os.remove(checkpoint_path_old)

            # Tensorboard summary logging
            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if steps % h.validation_interval == 0 and steps != 0:
                encoder.eval()
                decoder.eval()
                torch.cuda.empty_cache()
                val_Mel_err_tot = 0
                val_Mel_L2_err_tot = 0
                val_MDCT_loss = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        MDCT_coff, y, y_mel = batch
                        latent,_,_,_,_ = encoder(MDCT_coff.to(device))
                        y_g,MDCT_g_coff = decoder(latent)
                        MDCT_g_coff=MDCT_g_coff.to(device)
                        MDCT_coff=MDCT_coff.to(device)
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                        y_g_mel = mel_spectrogram(y_g.squeeze(1), h.n_fft, h.num_mels_for_loss, h.sampling_rate,h.hop_size, h.win_size, 0, None)
                        
                        val_Mel_err_tot += F.l1_loss(y_mel, y_g_mel).item()
                        val_Mel_L2_err_tot += amplitude_loss(y_mel, y_g_mel).item()
                        val_MDCT_loss += amplitude_loss(MDCT_coff,MDCT_g_coff).item()
                        if j <= 4:
                            if steps == 0:
                                sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)

                            sw.add_audio('generated/y_g_{}'.format(j), y_g[0], steps, h.sampling_rate)

                    val_Mel_err = val_Mel_err_tot / (j+1)
                    val_Mel_L2_err = val_Mel_L2_err_tot / (j+1)
                    val_MDCT_loss = val_MDCT_loss / (j+1)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_L2_loss", val_Mel_L2_err, steps)
                    sw.add_scalar("Validation/val_MDCT_loss", val_MDCT_loss, steps)

                encoder.train()
                decoder.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    config_file = 'config_codec.json'

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, 'config.json', h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == '__main__':
    main()
