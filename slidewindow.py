from option import Option
from tsai.all import SlidingWindow
import numpy as np

option=Option()

class Slidewindow :
    def __init__(self, Option: option):
        self.slidewindow_length = option.slidewindow_length
        self.slidewindow_stride = option.slidewindow_stride
        self.num_row_perpage=option.num_row_perpage
        self.num_person=option.num_person
        self.num_gesture=option.num_gesture

    def window_3d(self, x):
        x_length = int((self.num_row_perpage + self.slidewindow_stride - self.slidewindow_length) / self.slidewindow_stride)

        x_new_samples = x_length * self.num_person * self.num_gesture
        x_new_features = x.shape[1]
        x_new_steps = self.slidewindow_length

        x_new = np.zeros((x_new_samples, x_new_features, x_new_steps))
        y_new = np.zeros((x_new_samples,))

        for i in range(int(x.shape[0])):
            x_temp = x[i, :, :]
            x_slide, y_slide = SlidingWindow(window_len=self.slidewindow_length, stride=self.slidewindow_stride, seq_first=False)(x_temp)
            x_new[i * x_length:(i + 1) * x_length, :, :] = x_slide
            y_new[i * x_length:(i + 1) * x_length, ] = int(i % self.num_gesture)

        return x_new, y_new
