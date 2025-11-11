import numpy as np
import torch
import scipy.signal as signal
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import window2d
from utils.filterbank import FilterBank
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def safe_log(x, eps=1e-7):
    """
    Avoid taking the log of a non-positive number
    """
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)


def get_stft(x, n_fft, hop_length=None):
    """
    Returns the stft of x.
    """
    return torch.stft(x,
                      n_fft=n_fft,
                      hop_length = hop_length,
                      window=torch.hann_window(n_fft).to(device),
                      return_complex=False)




"""
Training Losses
"""
def L1_and_Log(x,y, n_fft=512, hop_length=None, eps=1e-6):
    """
    Computes spectral L1 plus log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    n_fft: n_fft for stft
    hop_length: stft hop length
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss (float)
    """
    est_stft = get_stft(x, n_fft=n_fft,hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft,hop_length=hop_length)
    
    assert est_stft.shape == ref_stft.shape
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)

    result = torch.mean(torch.abs(safe_log(est_amp)-safe_log(ref_amp))) + torch.mean(torch.abs(est_amp-ref_amp))
    return result

def training_loss(x,y,cutoff=9000, eps=1e-6):
    """
    Training Loss

    Computes spectral L1 and log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss: float tensor
    """
    loss1 = L1_and_Log(x,y, n_fft=512, eps=eps)
    loss2 = L1_and_Log(x,y, n_fft=1024, eps=eps)
    loss3 = L1_and_Log(x,y, n_fft=2048, eps=eps)
    loss4 = L1_and_Log(x,y, n_fft=4096, eps=eps)
    tiny_hop_loss = L1_and_Log(x[...,:cutoff], y[...,:cutoff], n_fft=256, eps=eps, hop_length=1)
    return loss1 + loss2 + loss3 + loss4 + tiny_hop_loss


"""
Evaluation Metrics
"""

def log_L1_STFT(x,y, n_fft=512, eps=1e-6, hop_length=None):
    """
    Computes log spectral L1 loss

    Parameters
    ----------
    x: first audio waveform(s), torch.tensor
    y: second audio waveform(s), torch.tensor
    n_fft: n_fft for stft
    hop_length: stft hop length
    eps: added to the magnitude stft before taking the square root. Limits dynamic range of spectrogram.

    Returns
    -------
    loss, float tensor
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
    loss += log_L1_STFT(x,y, n_fft=64, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=128, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=256, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=512, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=1024, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=2048, eps=eps)
    loss += log_L1_STFT(x,y, n_fft=4096, eps=eps)
    return loss

def env_loss(x, y, envelope_size=32, eps=1e-6):
    """Envelope Evaluation Metric. x,y are tensors representing waveforms."""
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps

    loss =  (np.mean(np.abs(np.log(env1) - np.log(env2))))
    
    return loss

# Direct-to-Reverberant Ratio (DRR) metric
@torch.no_grad()
def DRR(x, y, fs=48000, window_ms=2.5):
    """
    Compute Direct-to-Reverberant Ratio (DRR) between estimated (x) and reference (y) RIRs.

    Parameters
    ----------
    x: torch.Tensor, estimated RIR (1D)
    y: torch.Tensor, reference RIR (1D)
    fs: int, sampling rate in Hz
    window_ms: float, half-window size around the direct sound in milliseconds

    Returns
    -------
    DRR error (squared difference of DRRs), float
    """
    # compute sample window
    n = x.shape[-1]
    window_samples = int(fs * window_ms / 1000)

    # find peak in each signal (direct sound)
    peak_x = torch.argmax(torch.abs(x)).item()
    peak_y = torch.argmax(torch.abs(y)).item()

    # compute DRR for x
    direct_x = x[max(0, peak_x - window_samples): min(n, peak_x + window_samples + 1)]
    reverb_x = x[min(n, peak_x + window_samples + 1):]
    drr_x = 10 * torch.log10(torch.sum(direct_x ** 2) / (torch.sum(reverb_x ** 2) + 1e-10))

    # compute DRR for y
    direct_y = y[max(0, peak_y - window_samples): min(n, peak_y + window_samples + 1)]
    reverb_y = y[min(n, peak_y + window_samples + 1):]
    drr_y = 10 * torch.log10(torch.sum(direct_y ** 2) / (torch.sum(reverb_y ** 2) + 1e-10))

    # return absolute error
    return abs(drr_x - drr_y).item()

@torch.no_grad()
def C50(x, y, fs=48000):
    """
    Compute C50 (Clarity) between estimated (x) and reference (y) RIRs.

    Parameters
    ----------
    x: torch.Tensor, estimated RIR (1D)
    y: torch.Tensor, reference RIR (1D)
    fs: int, sampling rate in Hz

    Returns
    -------
    C50 error (squared difference of C50 values), float
    """
    def compute_c50(signal):
        n_50 = int(0.05 * fs)
        energy_total = torch.sum(signal ** 2)
        energy_early = torch.sum(signal[:n_50] ** 2)
        D_50 = energy_early / (energy_total + 1e-10)
        return 10 * torch.log10(D_50 / (1 - D_50 + 1e-10))
    
    c50_x = compute_c50(x)
    c50_y = compute_c50(y)
    return abs(c50_x - c50_y).item()

@torch.no_grad()
def AveragePower(x,y): # x , y // pred, target
    '''compute the Average Power convergence of two RIRs'''
    x = torch.tensor(x)
    y = torch.tensor(y)
    # compute the magnitude spectrogram
    S1 = torch.abs(torch.stft(x, n_fft=1024, hop_length=256, return_complex=True))
    S2 = torch.abs(torch.stft(y, n_fft=1024, hop_length=256, return_complex=True))
    
    # create 2d window
    win = window2d(torch.hann_window(64, dtype=S1.dtype, device=S1.device))
    # convove spectrograms with the window
    S1_win = F.conv2d(S1.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), stride=(4, 4)).squeeze()
    S2_win = F.conv2d(S2.unsqueeze(0).unsqueeze(0), win.unsqueeze(0).unsqueeze(0), stride=(4, 4)).squeeze()
    # compute the normalized difference between the two windowed spectrograms
    return torch.norm(S2_win - S1_win, p="fro") / torch.norm(S2_win, p="fro") / torch.norm(S1_win, p="fro")

@torch.no_grad()
class EDCLoss(nn.Module):
    '''compute the Energy Decay Convergence of two RIRs'''
    def __init__(self, backend = 'torch',sr = 48000,nfft=None):
        super().__init__()
        self.sr = sr
        self.filterbank = FilterBank(fraction=3,
                                order = 5,
                                fmin = 60,
                                fmax = 15000,
                                sample_rate= self.sr,
                                backend=backend,
                                nfft=nfft)
        self.mse = nn.MSELoss(reduction='mean')

    def discard_last_n_percent(self, edc, n_percent):
        # Discard last n%
        last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
        out = edc[..., 0:last_id]

        return out
    
    def backward_int(self, x):
        # Backwards integral on last dimension
        x = torch.flip(x, [-1])
        x = (1 / x.shape[-1]) * torch.cumsum(x ** 2, -1)
        return torch.flip(x, [-1])


    def forward(self, y_pred, y_true):
        # Remove filtering artefacts (last 5 permille)
        y_pred = self.discard_last_n_percent(y_pred, 0.5)
        y_true = self.discard_last_n_percent(y_true, 0.5)
        # compute EDCs
        y_pred_edr = self.backward_int(self.filterbank(y_pred))
        y_true_edr = self.backward_int(self.filterbank(y_true))
        y_pred_edr = 10*torch.log10(y_pred_edr + 1e-32)
        y_true_edr = 10*torch.log10(y_true_edr + 1e-32)
        # Trim both signals to the same length after filtering
        min_len = min(y_pred_edr.shape[-1], y_true_edr.shape[-1])
        y_pred_edr = y_pred_edr[..., :min_len]
        y_true_edr = y_true_edr[..., :min_len]
        level_pred = y_pred_edr[:,:,0]
        level_true = y_true_edr[:,:,0]
        # compute normalized mean squared error on the EDCs
        num = self.mse(y_pred_edr - level_pred.unsqueeze(-1), y_true_edr - level_true.unsqueeze(-1))
        den = torch.mean(torch.pow(y_true_edr - level_true.unsqueeze(-1), 2))
        return  num / den

@torch.no_grad()
def edc_loss_single(x, y):
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    return EDCLoss()(x, y)

baseline_metrics = [multiscale_log_l1, env_loss, DRR, C50, AveragePower, edc_loss_single]

def LRE(x, y, n_fft = 1024, hop_length=None, eps=1e-6):
    """LRE - Binaural Evaluation."""
    est_stft = get_stft(x, n_fft=n_fft, hop_length=hop_length)
    ref_stft = get_stft(y, n_fft=n_fft, hop_length=hop_length)

    assert est_stft.shape == ref_stft.shape
    est_amp = torch.sqrt(est_stft[..., 0]**2 + est_stft[..., 1]**2 + eps)
    ref_amp = torch.sqrt(ref_stft[..., 0]**2 + ref_stft[..., 1]**2 + eps)
    dif = torch.sum(est_amp[1])/torch.sum(est_amp[0]) - torch.sum(ref_amp[1])/torch.sum(ref_amp[0])
    dif = dif ** 2

    return dif.item()
