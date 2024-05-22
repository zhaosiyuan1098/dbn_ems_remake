import os
import glob
import numpy as np
from scipy import signal
from yacs.config import CfgNode as CN

def load_and_process_data(path_s):
    emgs = np.loadtxt(os.path.join(path_s, 'emg.txt'))
    labels = np.loadtxt(os.path.join(path_s, 'restimulus.txt'))
    repetitions = np.loadtxt(os.path.join(path_s, 'rerepetition.txt'))

    # Perform 1-order 1Hz low-pass filter on EMG data
    order = 1
    fs = 100  # sample rate: 100Hz
    cutoff = 1  # cutoff frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, 'lowpass')
    emgs = np.array([signal.filtfilt(b, a, emg) for emg in emgs.T]).T

    # u-law normalization for EMG data
    u = 256
    emgs = np.sign(emgs) * np.log(1 + u * abs(emgs)) / np.log(1 + u)
    return emgs, labels, repetitions

def get_all_data(cfg):
    paths_s = glob.glob(os.path.join(cfg.root_path, 's*'))
    emg_list = []
    label_list = []
    repetition_list = []
    for path in sorted(paths_s):
        emgs, labels, repetitions = load_and_process_data(path)
        emg_list.append(emgs)
        label_list.append(labels)
        repetition_list.append(repetitions)
    
    # Find the minimum length of data across all subjects
    min_length = min(min(data.shape[0] for data in emg_list), 
                     min(data.shape[0] for data in label_list),
                     min(data.shape[0] for data in repetition_list))
    
    # Truncate all data to the minimum length and stack them
    emgs_array = np.stack([data[:min_length] for data in emg_list])
    labels_array = np.stack([data[:min_length] for data in label_list])
    repetitions_array = np.stack([data[:min_length] for data in repetition_list])
    
    return emgs_array, labels_array, repetitions_array

def sliding_window(data, window_length, step):
    """
    Apply sliding window to the data.
    Args:
        data (numpy.ndarray): The data to apply the sliding window, shape (n_subjects, n_samples, n_features).
        window_length (int): Length of the sliding window.
        step (int): Step size of the sliding window.

    Returns:
        numpy.ndarray: The transformed data with shape (n_windows, n_subjects, n_features, window_length).
    """
    n_subjects, n_samples, n_features = data.shape
    n_windows = (n_samples - window_length) // step + 1
    # Correctly initialize the window array with proper dimensions
    windows = np.zeros((n_windows, n_subjects, n_features, window_length))

    # Correct the indexing to handle the dimensions properly
    for i in range(n_windows):
        start_index = i * step
        end_index = start_index + window_length
        for j in range(n_subjects):
            for k in range(n_features):
                windows[i, j, k, :] = data[j, start_index:end_index, k]

    return windows


def apply_sliding_window(all_emgs, all_labels, all_repetitions, window_length, step):
    """
    Apply sliding window to emg data, labels, and repetitions.
    Args:
        all_emgs (numpy.ndarray): EMG data array, shape (27, n, 10).
        all_labels (numpy.ndarray): Label array, shape (27, n).
        all_repetitions (numpy.ndarray): Repetition array, shape (27, n).
        window_length (int): Length of the sliding window.
        step (int): Step size of the sliding window.

    Returns:
        tuple: Tuple containing transformed emgs, labels, and repetitions.
    """
    emgs_windows = sliding_window(all_emgs, window_length, step)
    labels_windows = sliding_window(all_labels[:, :, np.newaxis], window_length, step)  # Add an extra dimension for consistency
    repetitions_windows = sliding_window(all_repetitions[:, :, np.newaxis], window_length, step)  # Add an extra dimension for consistency

    # We need to decide on the label for each window. A common practice is to take the mode of the labels in the window.
    # Here we will just take the first label of the window for simplicity.
    labels_windows = labels_windows[:, :, 0, 0]
    repetitions_windows = repetitions_windows[:, :, 0, 0]

    return emgs_windows, labels_windows, repetitions_windows

def flatten_windows(emgs_windows, labels_windows, repetitions_windows):
    """
    Flatten the first two dimensions of the windowed data arrays.
    Args:
        emgs_windows (numpy.ndarray): Windowed EMG data, shape (n_windows, n_subjects, n_features, window_length).
        labels_windows (numpy.ndarray): Windowed labels data, shape (n_windows, n_subjects).
        repetitions_windows (numpy.ndarray): Windowed repetitions data, shape (n_windows, n_subjects).

    Returns:
        tuple: Tuple containing flattened emgs, labels, and repetitions.
    """
    n_windows, n_subjects, n_features, _ = emgs_windows.shape
    # Reshape EMG data
    emgs_flattened = emgs_windows.transpose(1, 0, 2, 3).reshape(n_windows * n_subjects, n_features, window_length)
    # Flatten labels and repetitions
    labels_flattened = labels_windows.transpose(1, 0).flatten()
    repetitions_flattened = repetitions_windows.transpose(1, 0).flatten()

    return emgs_flattened, labels_flattened, repetitions_flattened

def save_data_to_npy(emgs_flattened, labels_flattened, repetitions_flattened):
    """
    Save the EMG data, labels, and repetitions as .npy files in the './data' directory.
    """
    directory = "./data"
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(f"{directory}/emgs_flattened.npy", emgs_flattened)
    np.save(f"{directory}/labels_flattened.npy", labels_flattened)
    np.save(f"{directory}/repetitions_flattened.npy", repetitions_flattened)
    print("Data successfully saved to:", directory)

def load_data_from_npy():
    """
    Load the EMG data, labels, and repetitions from .npy files in the './data' directory.
    Returns:
        tuple: Tuple containing numpy arrays for emgs, labels, and repetitions.
    """
    directory = "./data"
    emgs_flattened = np.load(f"{directory}/emgs_flattened.npy")
    labels_flattened = np.load(f"{directory}/labels_flattened.npy")
    repetitions_flattened = np.load(f"{directory}/repetitions_flattened.npy")
    return emgs_flattened, labels_flattened, repetitions_flattened



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



if __name__ == "__main__":
    with open('./cfgs/db1.yaml') as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        print('Successfully loading the config file...')
        dataCfg = cfg['DatasetConfig']
        all_emgs, all_labels, all_repetitions = get_all_data(dataCfg)
        print('Shapes of the arrays:', all_emgs.shape, all_labels.shape, all_repetitions.shape)
        window_length = 200  # 窗口长度
        step = 200  # 步长

        # 应用滑动窗口函数
        emgs_windows, labels_windows, repetitions_windows = apply_sliding_window(
            all_emgs, all_labels, all_repetitions, window_length, step
        )

        # 打印输出窗口数据的形状
        print("EMG windows shape:", emgs_windows.shape)
        print("Labels windows shape:", labels_windows.shape)
        print("Repetitions windows shape:", repetitions_windows.shape)

        emgs_flattened, labels_flattened, repetitions_flattened = flatten_windows(emgs_windows, labels_windows, repetitions_windows)

        print("Flattened EMG windows shape:", emgs_flattened.shape)
        print("Flattened Labels shape:", labels_flattened.shape)
        print("Flattened Repetitions shape:", repetitions_flattened.shape)

        save_data_to_npy(emgs_flattened, labels_flattened, repetitions_flattened)

        emgs_loaded, labels_loaded, repetitions_loaded = load_data_from_npy()
        print("Loaded EMG data shape:", emgs_loaded.shape)
        print("Loaded labels shape:", labels_loaded.shape)
        print("Loaded repetitions shape:", repetitions_loaded.shape)