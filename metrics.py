import torch
import torchaudio
import numpy as np
import scipy.signal as signal

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def safe_log(x, eps=1e-7):
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)

"""
Setting up STFT
"""
def get_stft(x, n_fft, hop_length=None):
    return torch.stft(x, n_fft=n_fft, hop_length = hop_length, window=torch.hann_window(n_fft).to(device), return_complex=False)

"""
Log L1 Losses
"""
def L1_and_Log(x,y, n_fft=512, eps=1e-6,hop_length=None):
    """Computes spectral L1 and log spectral L1 loss"""
    est_stft = get_stft(x, n_fft=n_fft,hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft,hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)

    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp))) + torch.mean(torch.abs(est_amp-ref_amp))
    return result

def mason_special_DDSP_loss(x,y,cutoff=9000, eps=1e-6, tiny_hop=False):
    """Training Loss"""
    loss1 = L1_and_Log(x,y,eps=eps)
    loss2 = L1_and_Log(x,y, n_fft=1024, eps=eps)
    loss3 = L1_and_Log(x,y, n_fft=2048, eps=eps)
    loss4 = L1_and_Log(x,y, n_fft=4096, eps=eps)

    if tiny_hop:
        loss5 = L1_and_Log(x[...,:cutoff], y[...,:cutoff], n_fft=256, eps=eps, hop_length=1)
    else:
        loss5 = 0
    return loss1 + loss2 + loss3 + loss4 + loss5



"""
Evaluation Metrics
"""


def log_L1_STFT(x,y, n_fft=512, eps=1e-6, hop_length=None):
    """
    Helper function - 
    For a given window sizes, log-L1 spectral loss
    """
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)
    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp)))

    return result


def multiscale_log_l1(x,y, eps=1e-6):
    """Spectral Evaluation Metric"""
    loss = 0
    loss += log_L1_STFT(x,y,eps=eps)
    loss += log_L1_STFT(x,y, n_fft=1024, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=2048, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=4096, eps=eps)
    loss += log_L1_STFT(x,y,n_fft=256,eps=eps)
    loss += log_L1_STFT(x,y,n_fft=128,eps=eps)
    loss += log_L1_STFT(x,y,n_fft=64,eps=eps)
    return loss

def env_loss(x, y, envelope_size=32):
    """Envelope Evaluation Metric"""
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+1e-6
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+1e-6

    loss =  (np.mean(np.abs(np.log(env1) - np.log(env2))))
    
    return loss

baseline_metrics = [multiscale_log_l1, env_loss]


"""
Binaural Evaluation
"""
def LRE(x, y, n_fft = 1024, hop_length=None, eps=1e-06):
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)

    assert est_stft.shape == ref_stft.shape    
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + 1e-06)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + 1e-06)
    dif = torch.sum(est_amp[1])/torch.sum(est_amp[0]) - torch.sum(ref_amp[1])/torch.sum(ref_amp[0])
    dif = dif ** 2

    return dif.item()