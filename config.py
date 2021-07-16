"""Module to pass parameters from keyboard"""
from argparse import ArgumentParser


def load_config():
    """Create config for arguments"""
    parser = ArgumentParser(description="Parser")

    parser.add_argument('--train', action='store_true',
                        help='do train')
    parser.add_argument('--test', action='store_true',
                        help='do test')
                        
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate')
    parser.add_argument('--seed', default=42, type=int,
                        help='Fixed seed')
    parser.add_argument('--train_bs', default=16, type=int,
                        help='Batch size in train step')
    parser.add_argument('--test_bs', default=32, type=int,
                        help='Batch size in test step')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        help='Select optimizer for problem')
    parser.add_argument('--ckpt', default='./checkpoint/', type=str,
                        help='Path to save checkpoint')

    args = parser.parse_args()

    return args
