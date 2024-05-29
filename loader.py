import pickle

import numpy as np
import pandas as pd
import torch
from fastai.learner import load_learner
from tsai.learner import load_all
from tsai.utils import toarray

from option import Option

option = Option()


class Loader:
    def __init__(self, Option: option):
        self.num_person = option.num_person
        self.num_gesture = option.num_gesture
        self.num_channel = option.num_channel
        self.num_row_perpage = option.num_row_perpage
        self.folder_path = option.folder_path

    def load_3d(self):
        num_samples = self.num_person * self.num_gesture
        num_features = self.num_channel
        num_steps = self.num_row_perpage
        x = np.zeros((num_samples, num_features, num_steps))
        y = np.zeros((num_samples))
        for i in range(1, self.num_person + 1):
            for j in range(1, self.num_gesture + 1):
                dftemp = pd.read_excel(self.folder_path + '/{}{}.xls'.format(i, j))
                x_index = (i - 1) * self.num_gesture + j - 1
                x[x_index, :, :] = dftemp.T
                y[x_index] = int(x_index + 1)
        return x, y
    
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

    def load_model(self, model_name):
        path='models'
        dls_fname=model_name + '_dls'
        model_fname=model_name + '_model'
        learner_fname=model_name + '_learner'
        learner=load_all(path=path,dls_fname=dls_fname,model_fname=model_fname,learner_fname=learner_fname)
        dls = learner.dls
        x_train, y_train,_ = learner.get_preds(dl=dls.train, with_decoded=True)

        # x_valid, y_valid,_ = learner.get_preds(dl=dls.valid, with_decoded=True)  #这里是原来的的读取
        # return x_train, x_valid, y_train, y_valid

        def load_test_dl(path, fname):      # 这里是读取模型train函数中的test_dls
            with open(f"{path}/{fname}", 'rb') as f:
                test_dl = pickle.load(f)
            return test_dl
        # 使用这个函数加载test_dl：
        test_dl = load_test_dl(path, model_name+'_test_dl.pkl')
        x_test, y_test,_ = learner.get_preds(dl=test_dl, with_decoded=True)

        return x_train, x_test, y_train, y_test
        # 如果想改回原来的读取只需要注释掉新的，原来的取消注释即可

    def load_for_dbn(self):
        xceptiontime_x_train, xceptiontime_x_valid, xceptiontime_y_train, xceptiontime_y_valid = self.load_model(
            'xceptiontime')
        ominiscale_x_train, ominiscale_x_valid, onimiscale_y_train, onimiscale_y_valid = self.load_model(
            'ominiscalecnn')
        xceptiontime_x_train_array = toarray(xceptiontime_x_train)
        ominiscale_x_train_array = toarray(ominiscale_x_train)
        dbn_x_train = np.zeros(
            (xceptiontime_x_train_array.shape[0],
             xceptiontime_x_train_array.shape[1] + ominiscale_x_train_array.shape[1]))
        dbn_x_train[:, 0:xceptiontime_x_train_array.shape[1]] = xceptiontime_x_train_array
        dbn_x_train[:,
        xceptiontime_x_train_array.shape[1]:xceptiontime_x_train_array.shape[1] + ominiscale_x_train_array.shape[
            1]] = ominiscale_x_train_array
        dbn_x_train = torch.from_numpy(dbn_x_train).float()

        xceptiontime_x_valid_array = toarray(xceptiontime_x_valid)
        ominiscale_x_valid_array = toarray(ominiscale_x_valid)
        dbn_x_valid = np.zeros(
            (xceptiontime_x_valid_array.shape[0],
             xceptiontime_x_valid_array.shape[1] + ominiscale_x_valid_array.shape[1]))
        dbn_x_valid[:, 0:xceptiontime_x_valid_array.shape[1]] = xceptiontime_x_valid_array
        dbn_x_valid[:,
        xceptiontime_x_valid_array.shape[1]:xceptiontime_x_valid_array.shape[1] + ominiscale_x_valid_array.shape[
            1]] = ominiscale_x_valid_array
        dbn_x_valid = torch.from_numpy(dbn_x_valid).float()

        x_train = dbn_x_train
        x_valid = dbn_x_valid
        y_train=xceptiontime_y_train
        y_valid=xceptiontime_y_valid

        return x_train,x_valid,y_train,y_valid

