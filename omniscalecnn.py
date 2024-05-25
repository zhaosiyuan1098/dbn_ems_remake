from tsai.all import *
from fastai.learner import Learner
from fastai.callback.wandb import WandbCallback
import wandb

from option import Option

option=Option()

class Omniscalecnn:
    def __init__(self, option: Option):
        self.valid_size = option.omniscalecnn_valid_size
        self.test_size = option.omniscalecnn_test_size
        self.stratify = option.omniscalecnn_stratify
        self.random_state = option.omniscalecnn_random_state
        self.shuffle = option.omniscalecnn_shuffle
        self.show_plots = option.omniscalecnn_show_plots
        self.bs = option.omniscalecnn_bs
        self.inplace = option.omniscalecnn_inplace
        self.wandb = option.wandb
        self.wandb_username = option.wandb_username

    def train(self, x, y):
        if self.wandb:
            wandb.init(project='zhao_omni', entity=self.wandb_username, config={
                "valid_size": self.valid_size,
                "test_size": self.test_size,
                "batch_size": self.bs,
            })

        splits = get_splits(y, valid_size=self.valid_size, test_size=self.test_size,
                            stratify=self.stratify, random_state=self.random_state,
                            shuffle=self.shuffle, show_plot=self.show_plots)
        tfms = [None, [Categorize()]]
        dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.inplace)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[self.bs, self.bs * 2])

        model = build_ts_model(OmniScaleCNN, dls=dls)
        callbacks = [WandbCallback()] if self.wandb else []
        learn = Learner(dls, model, metrics=[accuracy, RocAuc()], cbs=callbacks)
        learn.fit_one_cycle(100, 1e-3)
        learn.save_all(path='models', model_fname='omniscalecnn_model',
                       learner_fname='omniscalecnn_learner')

        if self.wandb:
            wandb.finish()

        return model
