from tsai.all import *
from option import Option

option=Option()

class Xceptiontime:
    def __int__(self):
        self.xceptiontime_valid_size = option.xceptiontime_valid_size
        self.xceptiontime_test_size = option.xceptiontime_test_size
        self.xceptiontime_stratify = option.xceptiontime_stratify
        self.xceptiontime_random_state = option.xceptiontime_random_state
        self.xceptiontime_shuffle = option.xceptiontime_shuffle
        self.xceptiontime_show_plots = option.xceptiontime_show_plots
        self.xceptiontime_bs = option.xceptiontime_bs
        self.xceptiontime_inplace = option.xceptiontime_inplace

    def train(self,x,y):
        splits = get_splits(y, valid_size=self.xceptiontime_valid_size, test_size=self.xceptiontime_test_size,
                            stratify=self.xceptiontime_stratify, random_state=self.xceptiontime_random_state,
                            shuffle=self.xceptiontime_shuffle, show_plot=self.xceptiontime_show_plots)

        tfms = [None, [Categorize()]]
        x_dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.xceptiontime_inplace)
        bs = 64
        x_dls = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[self.xceptiontime_bs, self.xceptiontime_bs * 2])
        xceptiontime_model = build_ts_model(XceptionTime, dls=x_dls)

        learn = Learner(x_dls, xceptiontime_model, metrics=[accuracy, RocAuc()])
        learn.fit_one_cycle(100, 1e-3)
        learn.save_all(path='models', dls_fname='x_dls', model_fname='x_model', learner_fname='x_learner')

        return xceptiontime_model



        