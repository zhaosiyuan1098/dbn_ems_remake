from compare import Compare
from fft import FFT
from ssa import SSA
from loader import Loader
from omniscalecnn import Ominiscalecnn
from option import Option
from slidewindow import Slidewindow
from xceptiontime import Xceptiontime
from dbn import DBN
from dbn import DBN_last_layer


def switch(a='train'):
    # 对象初始化

    option = Option()
    loader = Loader(option)
    fft = FFT()
    ssa = SSA(option)
    slidewindow = Slidewindow(option)
    compare = Compare(option)
    xceptiontime = Xceptiontime(option)
    ominiscalecnn = Ominiscalecnn(option)
    dbn = DBN(option)
    dbn_last_layer = DBN_last_layer(option)

    # 比较各模型
    if a == 'compare':
        print('compare')
        x_3d, _ = loader.load_3d()
        slidewindow_x_3d, slidewindow_y_3d = slidewindow.window_3d(x_3d)
        compare.model_compare(slidewindow_x_3d, slidewindow_y_3d)

    # 训练两个模型
    if a == 'train':
        print('train')
        x_3d, _ = loader.load_3d()
        ssa_x_3d = ssa.ssa_3d(x_3d)
        fft_x_3d = fft.fft_3d(x_3d)
        slidewindow_x_3d, slidewindow_y_3d = slidewindow.window_3d(fft_x_3d)
        xceptiontime_model = xceptiontime.train(slidewindow_x_3d, slidewindow_y_3d)
        ominiscalecnn_model = ominiscalecnn.train(slidewindow_x_3d, slidewindow_y_3d)
        x_train, x_valid, y_train, y_valid = loader.load_for_dbn()
        dbn_input_size = x_train
        dbn.pretrain(dbn_input_size)
        dbn_last_layer.train(x_train, y_train, x_train, x_valid, y_train, y_valid)

    # 加载xception和ominiscale结果，供后续dbn训练
    if a == 'load':
        print("load")
        x_train, x_valid, y_train, y_valid = loader.load_for_dbn()

    if a == 'plot':
        print('plot')


if __name__ == "__main__":
    switch('train')
