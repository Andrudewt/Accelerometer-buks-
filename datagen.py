import numpy as np
from keras.utils import Sequence # type: ignore
from scipy.signal import medfilt, spectrogram, detrend
from librosa.feature import melspectrogram, mfcc # type: ignore
from librosa import power_to_db # type: ignore
from librosa.display import specshow # type: ignore
import matplotlib.pyplot as plt

FFT = 512


class DataGen(Sequence):
    def __init__(self, X, y, batch_size, smpl_r):
        super().__init__()
        self.batch_size = batch_size
        # X_scl = X * 0.1
        self.X = medfilt(X, kernel_size=5)
        self.y = y
        self.smpl_r = smpl_r
        print(f'All data shape {X.shape} {y.shape}')
        plt.plot(self.X)

    def __len__(self):
        return int(np.floor((self.y.shape[0]) / self.batch_size))

    def __getitem__(self, index, plot_mode=False):
        batch_X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        return self.data_proc(batch_X, batch_y, plot_mode)

    def data_proc(self, X, y, plot_mode):
        # flt = np.ones(2501, np.float64)/2501      # на новых данных ~60м/с (Веларо) такую фильтрацию не прим
        # X_hf = np.convolve(X, flt, mode='valid')

        _, _, Sxx = spectrogram(
            X, fs=self.smpl_r,
            nperseg=FFT, noverlap=32,
            detrend='linear', mode='psd')

        mel_spec = melspectrogram(
            S=Sxx, sr=self.smpl_r,
            n_fft=FFT, n_mels=128,
            power=2, center=False)

        mfcc_spec = mfcc(
            S=power_to_db(mel_spec),
            sr=self.smpl_r, n_mfcc=128,
            dct_type=2, norm="ortho",)

        if plot_mode:
            _, ax = plt.subplots(3, 1)
            specshow(power_to_db(Sxx), y_axis='hz', sr=self.smpl_r, ax=ax[0])
            specshow(power_to_db(mel_spec), sr=self.smpl_r, x_axis='time', y_axis='mel', ax=ax[1])
            specshow(power_to_db(mfcc_spec), y_axis='hz', sr=self.smpl_r, ax=ax[2])

        data_way_interp = np.interp(np.linspace(0, 1, mfcc_spec.shape[1]), np.linspace(0, 1, y.shape[0]), y)

        return mfcc_spec.T, data_way_interp.reshape((-1, 1))
