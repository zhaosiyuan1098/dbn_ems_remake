from tsai.all import *

from option import Option

option = Option()


class Ominiscalecnn:
    def __init__(self, Option: option):
        self.valid_size = option.ominiscalecnn_valid_size
        self.test_size = option.ominiscalecnn_test_size
        self.stratify = option.ominiscalecnn_stratify
        self.random_state = option.ominiscalecnn_random_state
        self.shuffle = option.ominiscalecnn_shuffle
        self.show_plots = option.ominiscalecnn_show_plots
        self.bs = option.ominiscalecnn_bs
        self.inplace = option.ominiscalecnn_inplace

    def train(self, x, y):
        splits = get_splits(y, valid_size=self.valid_size, test_size=self.test_size,
                            stratify=self.stratify, random_state=self.random_state,
                            shuffle=self.shuffle, show_plot=self.show_plots)

        tfms = [None, [Categorize()]]
        x_dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.inplace)
        x_dls = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid,
                                         bs=[self.bs, self.bs * 2])
        ominiscalecnn_model = build_ts_model(OmniScaleCNN, dls=x_dls)

        learn = Learner(x_dls, ominiscalecnn_model, metrics=[accuracy, RocAuc()])
        learn.fit_one_cycle(100, 1e-3)
        learn.save_all(path='models', dls_fname='ominiscalecnn_dls', model_fname='ominiscalecnn_model',
                       learner_fname='ominiscalecnn_learner')

        return ominiscalecnn_model
