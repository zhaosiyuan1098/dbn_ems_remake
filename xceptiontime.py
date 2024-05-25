from tsai.all import *
from option import Option

# Assuming Option class is defined elsewhere and imported correctly
option = Option()

class Xceptiontime:
    def __init__(self, option: Option):
        self.valid_size = option.xceptiontime_valid_size
        self.test_size = option.xceptiontime_test_size
        self.stratify = option.xceptiontime_stratify
        self.random_state = option.xceptiontime_random_state
        self.shuffle = option.xceptiontime_shuffle
        self.show_plots = option.xceptiontime_show_plots
        self.bs = option.xceptiontime_bs
        self.inplace = option.xceptiontime_inplace
        self.wandb = option.wandb  # WandB integration flag
        self.wandb_username = option.wandb_username  # WandB username

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

        return xceptiontime_model

    def evaluate(self, model, x_test, y_test):
        # Create a TSDataset for the test set
        test_dset = TSDatasets(x_test, y_test, tfms=[None, [Categorize()]], splits=None)

        # Create a DataLoader for the test set
        test_dl = TSDataLoader(test_dset, bs=self.bs)

        # Create a Learner for the test set
        learn = Learner(test_dl, model, metrics=[accuracy, RocAuc()])

        # Validate the model on the test set
        test_loss, test_accuracy = learn.validate(dl=test_dl)
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")

        return test_loss, test_accuracy