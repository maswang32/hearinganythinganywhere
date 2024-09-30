import numpy as np
import torch
import torchaudio.functional as F
import metrics

def compute_error(predicted_audios, gt_audios, metric, device='cuda:0'):
    """
    Computes errors between a stack of two audio recordings.

    Parameters
    ----------
    predicted_audios: 2-dim array (N,T)
    gt_audios: 2-dim array (N,T)
    metric: function, evaluation function that compares 2 audio files

    Returns
    -------
    errors: (N,) - errors for each of the N pairs of recordings.
    """

    errors = np.zeros((gt_audios.shape[0]))

    assert (predicted_audios.shape == gt_audios.shape) or (len(predicted_audios) == len(gt_audios))

    for i in range(errors.shape[0]):
        errors[i] = metric(torch.tensor(predicted_audios[i]).to(device), torch.tensor(gt_audios[i]).to(device))
            
    return errors


def render_music(pred_rirs, music_sources, rir_length=96000, length=10*48000, device='cuda:0'):
    """
    Renders music given a stack of RIRs music sources through convolution.

    Parameters
    ----------
    pred_rirs: 2-dim array (N,T)
    music_sources: music source files, usually the direct-line recordings. (N,T2)?????????and the number of songs????
    rir_length: length to truncate RIRs to
    length: length to truncate music to after rendering

    Returns
    -------
    pred_musics: (N, T3) array of predicted music recordings
    """

    pred_rirs = torch.Tensor(pred_rirs)
    music_sources = torch.Tensor(music_sources)    
    pred_musics = torch.zeros((music_sources.shape[0], music_sources.shape[1], length))

    for i in range(music_sources.shape[0]):
        pred_musics[i] = F.fftconvolve(pred_rirs[i,:rir_length].unsqueeze(0).to(device),
                                        music_sources[i,:].to(device))[...,:length].cpu()

    torch.cuda.empty_cache()
    return pred_musics.detach().numpy()#############?????????non c'era detach


def eval_music(pred_music, gt_music, metric, length=10*48000, device='cuda:0'):
    """
    Evaluates rendered music against ground truth music.

    Parameters
    ----------
    pred_music: 2-dim array (N,K,T1), where K is the number of songs
    gt_music: music source files, usually the direct-line recordings. (N,K,T2)
    metric: function comparing 2 audio files
    length: length to truncate music to before evaluation.

    Returns
    -------
    errors: (N,K) errors for each datapoint, song pair.
    """
    gt_music = torch.Tensor(gt_music[...,:length])
    pred_music = torch.Tensor(pred_music[...,:length])

    assert gt_music.shape == pred_music.shape

    errors = np.zeros((pred_music.shape[0], pred_music.shape[1]))

    for i in range(pred_music.shape[0]):
        for song_idx in range(pred_music.shape[1]):
            errors[i, song_idx] = metric(gt_music[i, song_idx].to(device), pred_music[i,song_idx].to(device))
    
    torch.cuda.empty_cache()
    return errors