import numpy as np
from compare import Compare
from dbn import DBN
from dbn import DBN_last_layer
from fft import FFT
from loader import Loader
from omniscalecnn import Omniscalecnn
from option import Option
from preprocessor import Preprocessor
from ssa import SSA
from xceptiontime import Xceptiontime


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
    compare = Compare(option)
    xceptiontime = Xceptiontime(option)
    ominiscalecnn = Omniscalecnn(option)
    dbn = DBN(option)
    dbn_last_layer = DBN_last_layer(option)
    prepro = Preprocessor()

    # 比较各模型
    if a == 'compare':
        print('compare')

        x_train, x_test, y_train, y_test = prepro.load_ang_split()
        x_train_flattern, x_test_flattern, y_train_flattern, y_test_flattern = prepro.flattern(x_train, y_train, x_test,
                                                                                               y_test)
        compare.model_compare(x_train_flattern, y_train_flattern)

    if a == 'ninapro':
        print("ninapro")
        emgs_flattened, labels_flattened, repetitions_flattened = load_data_from_npy()
        print(emgs_flattened.shape)
        print(labels_flattened.shape)
        print(repetitions_flattened.shape)
        # compare.model_compare(emgs_flattened, labels_flattened)
        freq_x = fft.fft_transform_multidimensional(emgs_flattened)
        ssa_x = ssa.ssa_3d(emgs_flattened)
        time_ssa_x = merge_arrays(emgs_flattened, ssa_x)
        xceptiontime_model = xceptiontime.train(emgs_flattened, labels_flattened)

        # ominiscalecnn_model = ominiscalecnn.train(freq_x, labels_flattened)

    # 训练两个模型
    if a == 'train':
        print('train')
        # 新写了一个preprocess类，用于处理数据
        x_train, x_test, y_train, y_test = prepro.load_ang_split()
        x_train_flattern, x_test_flattern, y_train_flattern, y_test_flattern = prepro.flattern(x_train, y_train, x_test,
                                                                                               y_test)

        # 下面的六行可以合并到一起
        ssa_x_train = ssa.ssa_3d(x_train_flattern)
        ssa_x_test = ssa.ssa_3d(x_test_flattern)
        time_ssa_x_train = merge_arrays(x_train_flattern, ssa_x_train)
        time_ssa_x_test = merge_arrays(x_test_flattern, ssa_x_test)
        xceptiontime_model = xceptiontime.train(time_ssa_x_train, y_train_flattern, time_ssa_x_test, y_test_flattern)

        freq_x_train = fft.fft_transform_multidimensional(x_train_flattern)
        freq_x_test = fft.fft_transform_multidimensional(x_test_flattern)
        # time_fft_x_train = merge_arrays(x_train_flattern, freq_x_train)
        # time_fft_x_test = merge_arrays(x_test_flattern, freq_x_test)
        ominiscalecnn_model = ominiscalecnn.train(freq_x_train, y_train_flattern, freq_x_test, y_test_flattern)

    # 加载xception和ominiscale结果，供后续dbn训练
    if a == 'load':
        print("load")
        x_train, x_valid, y_train, y_valid = loader.load_for_dbn()

    if a == 'plot':
        print('plot')


if __name__ == "__main__":
    switch('train')
