import numpy as np
import pandas as pd

class Preprocessor():
    def __init__(self):
        self.num_person=4
        self.num_gesture=12
        self.num_channel=6
        self.num_row_perpage=4165
        self.folder_path = './data'
    def load(self):
        num_samples = self.num_person
        num_features = self.num_row_perpage * self.num_gesture
        num_steps = self.num_channel
        x = np.zeros((num_samples, num_features, num_steps))
        y = np.zeros((self.num_person, self.num_row_perpage * self.num_gesture))  # Change the shape of y
        for i in range(1, self.num_person + 1):
            for j in range(1, self.num_gesture + 1):
                dftemp = pd.read_excel(self.folder_path + '/{}{}.xls'.format(i, j))
                x_index = (i - 1) * self.num_gesture + j - 1
                x[i-1, (j-1)*self.num_row_perpage:(j*self.num_row_perpage), :] = dftemp
                y[i-1, (j-1)*self.num_row_perpage:(j*self.num_row_perpage)] = int(x_index)%12  # Change the way y is filled
        return x, y
    def sliding_window(self,data, window_length, step):
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
    
    def apply_sliding_window(self,all_emgs, all_labels, window_length, step):
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
        emgs_windows = self.sliding_window(all_emgs, window_length, step)
        labels_windows = self.sliding_window(all_labels[:, :, np.newaxis], window_length, step)  # Add an extra dimension for consistency

        # We need to decide on the label for each window. A common practice is to take the mode of the labels in the window.
        # Here we will just take the first label of the window for simplicity.
        labels_windows = labels_windows[:, :, 0, 0]

        return emgs_windows, labels_windows
    
    def split_data(self, emgs_windows, labels_windows, num_actions=12, split_ratio=0.8):
        """
        Split the data into training and test sets.

        Args:
            emgs_windows (np.array): The EMG windows data.
            labels_windows (np.array): The labels windows data.
            num_actions (int, optional): The number of different actions in the data. Defaults to 12.
            split_ratio (float, optional): The proportion of the dataset to include in the training split. Defaults to 0.8.

        Returns:
            tuple: Tuple containing training and test data for emgs and labels.
        """
        # Calculate the number of samples per action
        samples_per_action = len(emgs_windows) // num_actions

        emgs_train, emgs_test = [], []
        labels_train, labels_test = [], []

        # Split the data for each action
        for i in range(num_actions):
            start_idx = i * samples_per_action
            end_idx = (i + 1) * samples_per_action
            split_idx = start_idx + int(samples_per_action * split_ratio)

            emgs_train.append(emgs_windows[start_idx:split_idx])
            emgs_test.append(emgs_windows[split_idx:end_idx])

            labels_train.append(labels_windows[start_idx:split_idx])
            labels_test.append(labels_windows[split_idx:end_idx])

        # Concatenate the data from each action
        emgs_train, emgs_test = np.concatenate(emgs_train), np.concatenate(emgs_test)
        labels_train, labels_test = np.concatenate(labels_train), np.concatenate(labels_test)

        return emgs_train, emgs_test, labels_train, labels_test
    
    def flatten_windows(self,emgs_windows, labels_windows,window_length=60):
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
        
        return emgs_flattened, labels_flattened
    
    def flattern(self,x_train, y_train,x_test, y_test):
        x_train_flattern, y_train_flattern = self.flatten_windows(x_train, y_train)
        x_test_flattern, y_test_flattern = self.flatten_windows(x_test, y_test)
        return x_train_flattern, x_test_flattern,y_train_flattern,y_test_flattern

    
    def load_ang_split(self):
        x, y = self.load()
        x_windows, y_windows = self.apply_sliding_window(x, y, window_length=60, step=30)
        x_train, x_test, y_train, y_test = self.split_data(x_windows, y_windows,split_ratio=0.8)
        return x_train, x_test, y_train, y_test