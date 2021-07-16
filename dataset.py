"""Create dataset and dataloader from file"""
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from preprocessing import EncodeData


def load_file_csv(file_path, mode=None):
    """Load file from file path"""
    dataframe = pd.read_csv(file_path)
    data = dataframe['content_preprocess'].values
    label = dataframe['label'].values
    if mode == 'train':
        train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=0.2)
        return (train_data, train_label), (val_data, val_label)

    return (data, label)


class MakeDataset(Dataset):
    """Create Torch tensor dataset"""

    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (self.data[index], self.label[index])
        if self.transform is not None:
            encoded_sample = self.transform(X, y)
        return X, y


def get_data_loader(data, label, transform, batch_size=16):
    """Create dataloader from dataset"""
    dataset = MakeDataset(data, label, transform)

    return DataLoader(dataset, batch_size)
