from loader import Loader
from option import Option
from  fft import  FFT
from slidewindow import Slidewindow
from compare import Compare


if __name__ == "__main__":

    option=Option()
    loader=Loader(option)
    fft=FFT()
    slidewindow=Slidewindow(option)
    compare=Compare(option)

    x_3d, y_3d = loader.load_3d()
    fft_x_3d=fft.fft_3d(x_3d)
    fft_x_3d, y=slidewindow.window_3d(fft_x_3d)
    compare.model_compare(x_3d,y_3d)
    print(111)