"""Create dataset and dataloader from file"""
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from preprocessing import EncodeData

PRETRAINED_TOKENIZER = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)

def load_file_csv(file_path, mode=None):
    """Load file from file path"""
    dataframe = pd.read_csv(file_path)
    data = dataframe['data']
    label = dataframe['label']
    if mode == 'train':
        train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.2)
        return (train_data, train_label), (val_data, val_label)

    return (data, label)


class MakeDataset(Dataset):
    """Create Torch tensor dataset"""

    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        if transform is not None:
            self.data, self.label = transform(data, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index], self.label[index]
        return sample


def get_data_loader(data, label, transform, batch_size=16):
    """Create dataloader from dataset"""
    dataset = MakeDataset(data, label, transform)

    return DataLoader(dataset, batch_size)
