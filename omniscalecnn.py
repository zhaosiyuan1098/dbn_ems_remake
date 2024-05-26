from tsai.all import *
from fastai.learner import Learner
from fastai.callback.wandb import WandbCallback
import wandb
import pickle
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


    def train(self, x_train, y_train, x_test, y_test):
        if self.wandb:
            import wandb
            from fastai.callback.wandb import WandbCallback
            wandb.init(project='nina_xcep', entity=self.wandb_username, config={
                "valid_size": self.valid_size,
                "batch_size": self.bs,
            })
        # 这里会出现一张图，显示数据集的分布，但这里的test并不是test，而是valid
        splits = get_splits(y_train, valid_size=self.valid_size, stratify=self.stratify,
                            random_state=self.random_state, shuffle=self.shuffle, show_plot=self.show_plots)

        tfms = [None, [Categorize()]]
        train_dsets = TSDatasets(x_train, y_train, tfms=tfms, splits=splits, inplace=self.inplace)
        test_dset = TSDatasets(x_test, y_test, tfms=tfms, splits=None)

        train_dls = TSDataLoaders.from_dsets(train_dsets.train, train_dsets.valid, bs=[self.bs, self.bs * 2])
        test_dl = TSDataLoader(test_dset, bs=self.bs)

        xceptiontime_model = build_ts_model(XceptionTime, dls=train_dls)

        callbacks = [WandbCallback()] if self.wandb else []
        learn = Learner(train_dls, xceptiontime_model, metrics=[accuracy, RocAuc()], cbs=callbacks)
        learn.fit_one_cycle(100, 1e-3)

        # Validate the model on the test set
        #这里的test_dl是在preprocess里手工分出来的，所以这里的test是真正的test
        metrics = learn.validate(dl=test_dl)
        test_loss, test_accuracy, test_roc_auc = metrics[0], metrics[1], metrics[2]
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")

        if self.wandb:
            wandb.finish()

        learn.save_all(path='models', dls_fname='ominiscalecnn_dls', model_fname='ominiscalecnn_model',
                       learner_fname='ominiscalecnn_learner')

        # 添加保存test_dl
        def save_test_dl(test_dl, path, fname):
            with open(f"{path}/{fname}", 'wb') as f:
                pickle.dump(test_dl, f)
        save_test_dl(test_dl, 'models', 'ominiscalecnn_test_dl.pkl')

        return xceptiontime_model