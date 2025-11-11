import torch 
from torch import nn
import scipy 
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
from scipy.signal import sosfreqz
from utils.utils import get_device

class FilterBank(nn.Module):
    """Generates a filterbank and filters tensors.

    This is gpu compatible if using torch backend, but it is super slow and should not be used at all.
    The octave filterbanks is created using cascade Buttwerworth filters, which then are processed using
    the biquad function native to PyTorch.

    This is useful to get the decay curves of RIRs.
    """

    def __init__(self, fraction=3, order=5, fmin = 20, fmax = 18000, sample_rate=48000, nfft = None, backend='scipy'):
        super(FilterBank, self).__init__()

        assert fraction == 1 | fraction == 3,  "At the moment only fractions 1 and 3 are supported"

        # nominal frequencies 
        nom_freq_f1 = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
        nom_freq_f3 = [16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000, 25000, 32000]

        if fraction == 1:
            index = [0, len(nom_freq_f1)]
            for i, f in enumerate(nom_freq_f1):
                if fmin > f:
                    index[0] = i
                    break 
            for i, f in enumerate(nom_freq_f1):
                if f > fmax:
                    index[1] = i
                    break   
            center_frequencies = nom_freq_f1[index[0]:index[1]]

        if fraction == 3:
            index = [0, len(nom_freq_f3)]
            for i, f in enumerate(nom_freq_f3):
                if fmin > f:
                    index[0] = i+1
                    break 
            for i, f in enumerate(nom_freq_f3):
                if f > fmax:
                    index[1] = i
                    break    
            center_frequencies = nom_freq_f3[index[0]:index[1]]

        self._center_frequencies = center_frequencies
        self._order = order
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(center_frequencies, self._sample_rate, self._order)
        self.backend = backend

        self.nfft = nfft
        self.freqz = None  # will be initialized in _forward_torch based on x

    def _forward_scipy(self, x):
        out = []
        for this_sos in self._sos:
            tmp = torch.clone(x).cpu().numpy()
            tmp = scipy.signal.sosfilt(this_sos, tmp, axis=-1)
            out.append(torch.from_numpy(tmp.copy()))
        out = torch.stack(out, dim=-2)  # Stack over frequency bands

        return out

    def _forward_torch(self, x, filt_type='conv'):
        out = []
        if filt_type == 'conv':
            nfft = x.shape[-1] if self.nfft is None else self.nfft
            X = torch.fft.rfft(x, n=nfft)
            if self.freqz is None or self.freqz.shape[1] != X.shape[-1]:
                self.freqz = np.zeros((len(self._sos), X.shape[-1]), dtype=np.complex128)
                for i, sos in enumerate(self._sos):
                    _, self.freqz[i, :] = sosfreqz(sos, X.shape[-1], fs=self._sample_rate)
            out = []
            for i, this_freq in enumerate(self.freqz):
                this_freq = torch.tensor(this_freq, device=get_device())
                tmp = torch.fft.irfft(X * this_freq, n=nfft)
                out.append(tmp)
        else:
            for this_sos in self._sos:
                # loop over the sections 
                tmp = x
                for i, sos in enumerate(this_sos):
                    tmp =  torchaudio.functional.biquad(tmp, sos[0], sos[1], sos[2], sos[3], sos[4], sos[5])
                out.append(tmp)
                # b, a = scipy.signal.sos2tf(this_sos)
                # out.append(torchaudio.functional.lfilter(x, torch.tensor(a).float(), torch.tensor(b).float(), clamp = False))
                # out.append(sosfilt(torch.tensor(this_sos), x, axis=-1))
        out = torch.stack(out, dim=-2)  # Stack over frequency bands
        return out 

    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(self._center_frequencies, self._sample_rate, self._order)

    def set_order(self, order):
        self._order = order
        self._sos = self._get_octave_filters(self._center_frequencies, self._sample_rate, self._order)

    def set_center_frequencies(self, center_freqs):
        center_freqs_np = np.asarray(center_freqs)
        assert not np.any(center_freqs_np < 0) and not np.any(center_freqs_np > self._sample_rate / 2), \
            'Center Frequencies must be greater than 0 and smaller than fs/2. Exceptions: exactly 0 or fs/2 ' \
            'will give lowpass or highpass bands'
        self._center_frequencies = np.sort(center_freqs_np).tolist()
        self._sos = self._get_octave_filters(center_freqs, self._sample_rate, self._order)

    def get_center_frequencies(self):
        return self._center_frequencies

    def forward(self, x):
        if self.backend == 'scipy':
            out = self._forward_scipy(x)
        if self.backend == 'torch':
            out = self._forward_torch(x)
        else:
            raise NotImplementedError('No good implementation relying solely on the pytorch backend has been found yet')
        return out

    def get_filterbank_impulse_response(self):
        """Returns the impulse response of the filterbank."""
        impulse = torch.zeros(1, self._sample_rate * 20)
        impulse[0, self._sample_rate] = 1
        response = self.forward(impulse)
        return response

    @staticmethod
    def _get_octave_filters(center_freqs, fs, order):
        """
        Design octave band filters (butterworth filter).
        Returns a tensor with the SOS (second order sections) representation of the filter
        """
        sos = []
        for band_idx in range(len(center_freqs)):
            center_freq = center_freqs[band_idx]
            if abs(center_freq) < 1e-6:
                # Lowpass band below lowest octave band
                f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='lowpass', analog=False, output='sos')
            elif abs(center_freq - fs / 2) < 1e-6:
                f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='highpass', analog=False,
                                               output='sos')
            else:
                f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='bandpass', analog=False,
                                               output='sos')

            sos.append(torch.from_numpy(this_sos))

        return sos

def discard_last_n_percent(edc, n_percent):
    # Discard last n%
    last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
    out = edc[..., 0:last_id]

    return out

if __name__ == '__main__':
    # load impulse response
    filename = "/Users/dalsag1/Dropbox (Aalto)/aalto/projects/diff-delay-net/shungo-dar/website-rirs/as_0_rir.wav"
    rir, sr = sf.read(filename, dtype='float32')
    rir = torch.tensor(rir[:sr//2])

    filterbank = FilterBank(backend='torch')
    rir_filter = filterbank(rir)
    '''
    for i, sos in enumerate(filterbank._sos):
        w, h = scipy.signal.sosfreqz(sos, 512*4)
        plt.plot(w, np.abs(h))
        sf.write('rir_band'+str(i)+'.wav', rir_filter[i,:], sr)
    plt.show()
    '''
    # Remove filtering artefacts (last 5 permille)
    out = discard_last_n_percent(rir_filter, 0.5)

    # Backwards integral
    out = torch.flip(out, [1])
    out = (1 / out.shape[1]) * torch.cumsum(out ** 2, 1)
    out = torch.flip(out, [1])

    for i in range(out.shape[0]):
        plt.plot(out[i,:])
        # sf.write('rir_band'+str(i)+'.wav', rir_filter[i,:], sr)
    plt.show()
