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

    def train(self, x, y):
        if self.wandb:
            import wandb
            from fastai.callback.wandb import WandbCallback
            wandb.init(project='nina_xcep', entity=self.wandb_username, config={
                "valid_size": self.valid_size,
                "test_size": self.test_size,
                "batch_size": self.bs,
            })

        splits = get_splits(y, valid_size=self.valid_size, test_size=self.test_size,
                            stratify=self.stratify, random_state=self.random_state,
                            shuffle=self.shuffle, show_plot=self.show_plots)

        tfms = [None, [Categorize()]]
        x_dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=self.inplace)
        x_dls = TSDataLoaders.from_dsets(x_dsets.train, x_dsets.valid, bs=[self.bs, self.bs * 2])
        xceptiontime_model = build_ts_model(XceptionTime, dls=x_dls)

        callbacks = [WandbCallback()] if self.wandb else []
        learn = Learner(x_dls, xceptiontime_model, metrics=[accuracy, RocAuc()], cbs=callbacks)
        learn.fit_one_cycle(100, 1e-3)
        # learn.save_all(path='models', model_fname='xceptiontime_model', learner_fname='xceptiontime_learner')

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
