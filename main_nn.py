import os
import glob

import scipy.io
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics as tm
from tqdm import tqdm
from torch_audiomentations import Compose, PitchShift, Shift, Gain


def load_data_mat(root):
    files = glob.glob(os.path.join(root, "**", "*EEG.mat"), recursive=True)
    eeg_data = []

    for eeg_file in tqdm(files, desc="Loading files"):
        eeg_mat = scipy.io.loadmat(eeg_file)
        eeg_data.append(eeg_mat["EEG"])

    eeg_data = np.concatenate(eeg_data, axis=0, dtype=np.float32)
    return eeg_data

def normalize(tensor):
    ax_maxs = torch.max(tensor, dim=1, keepdim=True)[0]
    ax_mins = torch.min(tensor, dim=1, keepdim=True)[0]
    return (tensor - ax_mins) / (ax_maxs - ax_mins)

class EEGSpeechDataset(Dataset):
    def __init__(self, eeg_data, num_classes=11, split="train"):
        self.eeg_data = eeg_data
        self.num_classes = num_classes
        self.split = split

        self.num_eeg_channels = 6
        self.eeg_sample_point = 4096
        self.transforms = nn.Sequential(
            torchaudio.transforms.Resample(1024, 128),
        )
        self.X, self.y = self.get_eeg_data()

    def get_eeg_data(self):
        X, y, imaginated = self.eeg_data[:, :-3], self.eeg_data[:, -2], self.eeg_data[:, -3] # Checkd

        indices = np.logical_and(y >= 6, imaginated==1)
        X = X[indices]
        y = y[indices]

        y = y - np.min(y)
        X = np.reshape(X, (-1, self.num_eeg_channels, self.eeg_sample_point))
        
        if self.split == "train":
            return X[:int(0.9 * X.shape[0])], y[:int(0.9 * X.shape[0])]
        else:
            return X[int(0.9 * X.shape[0]):], y[int(0.9 * X.shape[0]):]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        X, y = torch.from_numpy(X), torch.from_numpy(np.array(y)).long()
        X = normalize(X)
        X = self.transforms(X)
        return X, y


class EEGModel(pl.LightningModule):
    def __init__(self, num_classes, dropout_ratio=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy(num_classes=num_classes)
        self.val_acc = tm.Accuracy(num_classes=num_classes)
        self.augmentations = Compose(
            transforms=
            [
                Gain(
                    min_gain_in_db=-15.0,
                    max_gain_in_db=5.0,
                    p=0.5,
                ),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            ]
        )

        self.seq1 = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=13),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_ratio),
        )
        self.seq2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=11),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_ratio),
        )

        self.seq3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=9),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_ratio),
        )

        self.seq4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_ratio),
        )

        self.seq5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_ratio),
        )

        self.seq6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        if self.training:
            x = self.augmentations(x, sample_rate=128)
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.seq5(x)
        x = self.seq6(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(y_hat, y.long())
        return loss

    def training_epoch_end(self, outputs):
        self.log("train_acc", self.train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_acc(y_hat, y.long())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, nesterov=True)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, min_lr=1e-4, factor=0.5, patience=10),
                "monitor": "val_loss",
            },
        }

def main():

    pl.seed_everything(seed=10, workers=True)
    num_classes = 6

    d = load_data_mat("dataset")
    ds_train = EEGSpeechDataset(d, num_classes, split="train")
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    ds_val = EEGSpeechDataset(d, num_classes, split="val")
    dl_val= DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = EEGModel(num_classes)

    logger = TensorBoardLogger("logs")
    callbacks = [LearningRateMonitor("step")]

    tr = pl.Trainer(gpus=1, max_epochs=100, logger=logger, callbacks=callbacks)
    tr.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == '__main__':
    main()