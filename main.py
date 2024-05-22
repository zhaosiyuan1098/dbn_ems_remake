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
import numpy as np

def merge_arrays(array1, array2):
    # 检查输入数组的形状是否符合要求
    if array1.shape[0] != array2.shape[0] or array1.shape[2] != array2.shape[2]:
        raise ValueError("输入数组的形状不符合要求")

    # 使用np.concatenate沿着第二个维度合并数组
    merged_array = np.concatenate((array1, array2), axis=1)
    
    return merged_array

def load_data_from_npy(directory="./data"):
    """
    Load the EMG data, labels, and repetitions from .npy files.
    Args:
        directory (str): Directory where the .npy files are stored.

    Returns:
        tuple: Tuple containing numpy arrays for emgs, labels, and repetitions.
    """
    directory = "./data"
    emgs_flattened = np.load(f"{directory}/emgs_flattened.npy")
    labels_flattened = np.load(f"{directory}/labels_flattened.npy")
    repetitions_flattened = np.load(f"{directory}/repetitions_flattened.npy")
    return emgs_flattened, labels_flattened, repetitions_flattened

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

    if a=='ninapro':
        print("ninapro")
        emgs_flattened, labels_flattened, repetitions_flattened = load_data_from_npy()
        print(emgs_flattened.shape)
        print(labels_flattened.shape)
        print(repetitions_flattened.shape)
        compare.model_compare(emgs_flattened, labels_flattened)

    # 训练两个模型
    if a == 'train':
        print('train')
        x_3d, _ = loader.load_3d()
        time_x,time_y=slidewindow.window_3d(x_3d)
        freq_x=fft.fft_transform_multidimensional(time_x)
        ssa_x=ssa.ssa_3d(time_x)
        time_ssa_x=merge_arrays(time_x,ssa_x)
        compare.model_compare(time_ssa_x, time_y)
        compare.model_compare(freq_x, time_y)

        # xceptiontime_model = xceptiontime.train(time_ssa_x, time_y)
        
        # ominiscalecnn_model = ominiscalecnn.train(freq_x, time_y)
        
        
        
        
        
        # ssa_x_3d = ssa.ssa_3d(x_3d)
        # fft_x_3d = fft.fft_3d(x_3d)
        # time_input_temp=merge_arrays(x_3d,ssa_x_3d)
        # time_input_x,time_input_y=slidewindow.window_3d(time_input_temp)
        # freq_input_x,freq_input_y=slidewindow.window_3d(fft_x_3d)
        # xceptiontime_model = xceptiontime.train(time_input_x, time_input_y)
        # ominiscalecnn_model = ominiscalecnn.train(freq_input_x, freq_input_y)


        # # slidewindow_x_3d, slidewindow_y_3d = slidewindow.window_3d(fft_x_3d)
        # # xceptiontime_model = xceptiontime.train(slidewindow_x_3d, slidewindow_y_3d)
        # # ominiscalecnn_model = ominiscalecnn.train(slidewindow_x_3d, slidewindow_y_3d)
        # x_train, x_valid, y_train, y_valid = loader.load_for_dbn()
        # dbn_input_size = x_train
        # dbn.pretrain(dbn_input_size)
        # dbn_last_layer.train(x_train, y_train, x_train, x_valid, y_train, y_valid)

    # 加载xception和ominiscale结果，供后续dbn训练
    if a == 'load':
        print("load")
        x_train, x_valid, y_train, y_valid = loader.load_for_dbn()

    if a == 'plot':
        print('plot')


if __name__ == "__main__":
    switch('ninapro')

