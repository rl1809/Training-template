"""Trainer for model"""
import os
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_file, get_data_loader
from utils import EarlyStopping

logger = logging.getLogger('trainer')

class Trainer:
    """Trainer class to train, validate and test"""
    def __init__(self, config, model):
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.model = model.to(self.device)
        
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.train_bs = config['train_bs']
        self.test_bs = config['test_bs']

        (train_data, train_label), (val_data, val_label) = load_file('data/train.csv', mode='train')

        test_data, test_label = load_file('data/test.csv')

        self.train_loader = get_data_loader(train_data, train_label, batch_size=self.train_bs)
        self.val_loader = get_data_loader(val_data, val_label, batch_size=self.train_bs)
        self.test_loader = get_data_loader(test_data, test_label, batch_size=self.test_bs)

        self.ckpt_path = config['ckpt_path']
        if os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        self.early_stopping = EarlyStopping(verbose=True, path=os.path.join(self.ckpt_path))

    def train(self):
        pass

    def _train_epoch(self):
        pass

    def _valid(self):
        pass
    
    def test(self, best_model):
        pass
    
