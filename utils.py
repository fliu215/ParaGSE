import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import shutil
import torch.nn.functional as F

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
        
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def compute_log_probs_from_logits(logits, target_tokens):
    """
    logits: (B, T, C)
    target_tokens: (B, T)
    return: log-likelihood per sample, shape (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, C)
    gathered = torch.gather(log_probs, 2, target_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return gathered.sum(dim=-1)  # sum over time


def dpo_loss_from_logits(logits, ref_logits, preferred_tokens, dispreferred_tokens, beta=0.1):
    """
    logits, ref_logits: (B, T, C)
    preferred_tokens, dispreferred_tokens: (B, T)
    """
    logp_pref = compute_log_probs_from_logits(logits, preferred_tokens)
    logp_disp = compute_log_probs_from_logits(logits, dispreferred_tokens)

    with torch.no_grad():
        logq_pref = compute_log_probs_from_logits(ref_logits, preferred_tokens)
        logq_disp = compute_log_probs_from_logits(ref_logits, dispreferred_tokens)

    logits_diff = (logp_pref - logq_pref) - (logp_disp - logq_disp)
    return -F.logsigmoid(beta * logits_diff).mean()

