import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


def create_subset(X, y, train_frac, train_set):
    """
    Splits the dataset into train and test sets but takes into account the labels so that
    1. their relative distribution remains the same across the splits
    2. no unknown labels across the splits
    :param X: features
    :param y: labels
    :param train_frac: fraction of the dataset to be used for training
    :param train_set: whether to return the train set or the test set
    :return:
    """
    if train_frac != 1.0:
        X_sub, y_sub = [], []
        for l in tqdm(np.unique(y), ncols=80, desc='Creating Subset', colour='red'):
            idx = (y == l)
            split_idx = int(y[idx].shape[0] * train_frac)
            if train_set:
                X_sub.append(X[idx][:split_idx])
                y_sub.append(y[idx][:split_idx])
            else:
                X_sub.append(X[idx][split_idx:])
                y_sub.append(y[idx][split_idx:])
        X = np.concatenate(X_sub)
        y = np.concatenate(y_sub)
    return X, y


def get_shuffled(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


class ActivityDataset(data.Dataset):
    def __init__(self,
                 datamat,
                 window_size,
                 n_channel,
                 scaling=False,
                 lencoder=None,
                 shuffle=False,
                 train_frac=1.0,
                 train_set=True):
        """
        The main dataset class for the activity recognition task.

        :param datamat: expects the windowed data already preprocessed it should be of shape (n_samples, n_channels*window_size+1)
        :param window_size: window length used in the sliding window preprocessing script
        :param n_channel: number of IMU channels, e.g. 3 when only using accelerometer
        :param scaling: whether to scale the data or not
        :param lencoder: label encoder, if None, one will be created, use the same one across train and test dataset
        :param shuffle: whether to shuffle the dataset or not
        :param train_frac: fraction of the dataset to be used for training
        :param train_set: whether its the train set or not
        """
        self.n_channel = n_channel
        self.window_size = window_size
        self.scaler = StandardScaler()

        self.X = datamat[:, :-1].astype(np.float)
        if scaling:
            self.X = self.scaler.fit_transform(self.X)
        self.X = self.X.reshape(self.X.shape[0], -1, 1, self.n_channel)
        # output shape after last line: (samples, window, 1, channel)
        self.X = self.X.transpose(0, 3, 1, 2)  # output (samples, channel, window, 1)

        if lencoder is not None:
            self.lencoder = lencoder
            self.y = self.lencoder.transform(datamat[:, -1].astype(str))
        else:
            self.lencoder = LabelEncoder()
            self.y = self.lencoder.fit_transform(datamat[:, -1].astype(str))

        self.X, self.y = create_subset(self.X, self.y, train_frac, train_set)
        if shuffle: self.X, self.y = get_shuffled(self.X, self.y)

    def __getitem__(self, index):
        sample, target = self.X[index], self.y[index]
        # if needed reshape the sample
        return sample, target

    def __len__(self):
        return self.X.shape[0]

    def get_lencoder(self):
        return self.lencoder
