from tsai.all import *
from option import Option

option=Option()

class Xceptiontime:
    def __init__(self,Option:option):
        self.valid_size = option.xceptiontime_valid_size
        self.test_size = option.xceptiontime_test_size
        self.stratify = option.xceptiontime_stratify
        self.random_state = option.xceptiontime_random_state
        self.shuffle = option.xceptiontime_shuffle
        self.show_plots = option.xceptiontime_show_plots
        self.bs = option.xceptiontime_bs
        self.inplace = option.xceptiontime_inplace

    def train(self,x,y):
        splits = get_splits(y, valid_size=self.valid_size, test_size=self.test_size,
                            stratify=self.stratify, random_state=self.random_state,
                            shuffle=self.shuffle, show_plot=self.show_plots)

        tfms = [None, [Categorize()]]
        x_dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.inplace)
        x_dls = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[self.bs, self.bs * 2])
        xceptiontime_model = build_ts_model(XceptionTime, dls=x_dls)

        learn = Learner(x_dls, xceptiontime_model, metrics=[accuracy, RocAuc()])
        learn.fit_one_cycle(100, 1e-3)
        learn.save_all(path='models', dls_fname='xceptiontime_dls', model_fname='xceptiontime_model', learner_fname='xceptiontime_learner')

        return xceptiontime_model
