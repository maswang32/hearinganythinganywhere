import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import librosa
import librosa.display
import plotly.graph_objects as go
import argparse
import rooms.dataset
import torchaudio.functional as F
import os
import pickle
import pandas as pd
import glob

fs = 48000

def compute_error(predicted_audios, gt_audios, cuda=True, metric=metrics.diffimpactLoss):
    """
    Parameters
    ----------
    predicted_audios: 2-dim array (N,T)
    gt_audios: 2-dim array (N,T)
    cuda: if we're using cuda
    metric: evaluation function, compares 2 audio files
    """
    device = "cuda:0" if cuda else "cpu"

    errors = np.zeros((gt_audios.shape[0]))

    assert (predicted_audios.shape == gt_audios.shape) or (len(predicted_audios) == len(gt_audios))

    for i in range(errors.shape[0]):
        errors[i] = metric(torch.tensor(predicted_audios[i]).to(device), torch.tensor(gt_audios[i]).to(device))
            
    return errors


"""
Rendering Music
"""
def render_music(pred_rirs, music_dls, save_path=None, length=10*48000, cuda=True):

    device = "cuda:0" if cuda else "cpu"
    pred_rirs = torch.Tensor(pred_rirs)
    music_dls = torch.Tensor(music_dls)

    
    pred_musics = torch.zeros((music_dls.shape[0], music_dls.shape[1], length))

    for i in range(music_dls.shape[0]):
        pred_musics[i] = F.fftconvolve(pred_rirs[i,:96000].unsqueeze(0).to(device), music_dls[i,:].to(device))[...,:length].cpu()

    if save_path is not None:
        np.save(os.path.join(save_path,"pred_musics.npy"), pred_musics.numpy())    

    torch.cuda.empty_cache()
    return pred_musics.numpy()


"""
Evaluating Rendered
"""
def eval_music(pred_music, gt_music, metric, length=10*48000,cuda=True):

    device = "cuda:0" if cuda else "cpu"
    gt_music = torch.Tensor(gt_music[...,116:length+116])
    pred_music = torch.Tensor(pred_music[...,:length])

    assert gt_music.shape == pred_music.shape

    errors = np.zeros((pred_music.shape[0], pred_music.shape[1]))

    for i in range(pred_music.shape[0]):
        for song_idx in range(pred_music.shape[1]):
            errors[i, song_idx] = metric(gt_music[i, song_idx].to(device), pred_music[i,song_idx].to(device))
    
    torch.cuda.empty_cache()
    return errors