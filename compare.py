from tsai.all import *
import pandas as pd
import time
import os
from fastai.learner import Learner
from fastai.callback.wandb import WandbCallback
import wandb

option=Option()

class Compare():
    def __init__(self,option:Option):
        self.compare_valid_size = option.compare_valid_size
        self.compare_test_size = option.compare_test_size
        self.compare_stratify = option.compare_stratify
        self.compare_random_state = option.compare_random_state
        self.compare_shuffle = option.compare_shuffle
        self.compare_show_plots = option.compare_show_plots
        self.compare_bs=option.compare_bs
        self.compare_inplace=option.compare_inplace
        self.wandb=option.wandb
        self.wandb_username=option.wandb_username



    def model_compare(self, x, y):
        if self.wandb:
            wandb.init(project='tsai-model-comparison', entity=self.wandb_username, config={
                "valid_size": self.compare_valid_size,
                "test_size": self.compare_test_size,
                "stratify": self.compare_stratify,
                "random_state": self.compare_random_state,
                "shuffle": self.compare_shuffle,
                "batch_size": self.compare_bs,
            })

        splits = get_splits(y, valid_size=self.compare_valid_size, test_size=self.compare_test_size,
                            stratify=self.compare_stratify, random_state=self.compare_random_state,
                            shuffle=self.compare_shuffle, show_plot=self.compare_show_plots)
        tfms = [None, [Categorize()]]
        batch_tfms = [TSStandardize(), TSNormalize()]
        dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.compare_inplace)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[self.compare_bs, self.compare_bs * 2])

        tsai_models = L(
            InceptionTime, XceptionTime, ResNet, ResCNN, FCN, MLP, TST, LSTM, GRU, TCN,
            RNN_FCN, LSTM_FCN, GRU_FCN, MiniRocket, TSiT, OmniScaleCNN,
            mWDN, TSTPlus, TSiTPlus
        )

        archs = [(m, {}) for m in tsai_models]
        results = pd.DataFrame(
            columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])

        for i, (arch, k) in enumerate(archs):
            model = create_model(arch, dls=dls, **k)
            callbacks = [WandbCallback(log_model=False, log_preds=False)] if self.wandb else []
            learn = Learner(dls, model, metrics=accuracy, cbs=callbacks)
            start = time.time()
            learn.fit_one_cycle(10, 1e-3)
            elapsed = time.time() - start
            vals = learn.recorder.values[-1]
            results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
            results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
            os.system('cls' if os.name == 'nt' else 'clear')
            display(results)

        if self.wandb:
            wandb.finish()

        return results
    # def model_compare(self,x,y):
    #     splits = get_splits(y, valid_size=self.compare_valid_size, test_size=self.compare_test_size,
    #                         stratify=self.compare_stratify, random_state=self.compare_random_state,
    #                         shuffle=self.compare_shuffle,show_plot=self.compare_show_plots)
    #     tfms = [None, [Categorize()]]
    #     batch_tfms = [TSStandardize(), TSNormalize()]
    #     dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.compare_inplace)
    #     dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[self.compare_bs, self.compare_bs * 2])

    #     archs = [(FCN, {}), (ResNet, {}), (ResCNN, {}),
    #              (LSTM, {'n_layers': 1, 'bidirectional': False}), (LSTM, {'n_layers': 2, 'bidirectional': False}),
    #              (LSTM, {'n_layers': 3, 'bidirectional': False}),
    #              (LSTM, {'n_layers': 1, 'bidirectional': True}), (LSTM, {'n_layers': 2, 'bidirectional': True}),
    #              (LSTM, {'n_layers': 3, 'bidirectional': True}),
    #              (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (XceptionTime, {}),
    #              (OmniScaleCNN, {}), (mWDN, {'levels': 4})]

    #     results = pd.DataFrame(
    #         columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
    #     for i, (arch, k) in enumerate(archs):
    #         model = create_model(arch, dls=dls, **k)
    #         print(model.__class__.__name__)
    #         learn = Learner(dls, model, metrics=accuracy)
    #         start = time.time()
    #         learn.fit_one_cycle(100, 1e-3)
    #         elapsed = time.time() - start
    #         vals = learn.recorder.values[-1]
    #         results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
    #         results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
    #         os.system('cls' if os.name == 'nt' else 'clear')
    #         display(results)