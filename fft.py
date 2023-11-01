from option import Option
import numpy as np
option=Option()

class FFT :
    def fft_3d(self, x):
        fft_data_3d = np.zeros(x.shape)
        for i in range(x.shape[0]):
            fft_temp = np.fft.fft(x[i:i + 1, :, :], axis=1)
            fft_data_3d[i:i + 1, :, :] = fft_temp
        return fft_data_3d
