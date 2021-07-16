"""Some utils function"""

import os
import glob
import random

import numpy as np
import torch


def fix_seed(seed):
    """Fix seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class ModeChoiceError(Exception):
    """Raise exception when not choose proper action"""

    def __str__(self):
        return "You should include --train or --test in your command!"


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="best_model.pt", trace_func=print):
        """
        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss improved, by default 7
        verbose : bool, optional
            If True, prints a message for each validation loss improvement, by default False
        delta : int, optional
            Minimum change in the monitored quantity to qualify as an improvement, by default 0
        path : str, optional
            Root path for the checkpoint to saved to, by default "best_model.pt"
        trace_func : [type], optional
            Trace print function, by default print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, epoch, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased \
                ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        new_path = os.path.join(
            self.path, "best_model_epoch_{}_loss_{:.3f}.pt".format(
                epoch, val_loss)
        )
        for filename in glob.glob(os.path.join(self.path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.val_loss_min = val_loss
