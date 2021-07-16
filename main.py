"""Main module of program"""
import os
from glob import glob

import torch

from config import load_config
from models.bert_model import ClassificationBert
from trainer import Trainer
from utils import fix_seed, ModeChoiceError


def main(args):
    """Main function"""
    fix_seed(args.seed)
    model = ClassificationBert()
    trainer = Trainer(config=vars(args), model=model)

    if args.train:
        model.train()

    elif args.test:
        test_model = ClassificationBert()
        checkpoint_path = glob(os.path.join(args.ckpt_path), "*.pt"[0])
        state_dict = torch.load(checkpoint_path)
        test_model.load_state_dict(state_dict)
        trainer.test(test_model)
    else:
        raise ModeChoiceError


if __name__ == '__main__':
    args = load_config()
    main(args)
