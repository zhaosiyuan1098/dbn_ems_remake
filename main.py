from loader import Loader
from option import Option
from  fft import  FFT
from slidewindow import Slidewindow
from compare import Compare
from xceptiontime import Xceptiontime

if __name__ == "__main__":

    option=Option()
    loader=Loader(option)
    fft=FFT()
    slidewindow=Slidewindow(option)
    compare=Compare(option)
    xceptiontime=Xceptiontime()

    x_3d, _ = loader.load_3d()
    fft_x_3d=fft.fft_3d(x_3d)
    slidewindow_x_3d, slidewindow_y_3d=slidewindow.window_3d(fft_x_3d)
    # xceptiontime_model=xceptiontime.train(slidewindow_x_3d,slidewindow_y_3d)
    compare.model_compare(slidewindow_x_3d,slidewindow_y_3d)
    print(111)