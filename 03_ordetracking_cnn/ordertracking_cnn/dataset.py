import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, X, y, augment: bool = False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = self.X[idx].clone()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x_tensor = x_tensor * torch.empty(1).uniform_(0.9, 1.1)
            if torch.rand(1).item() > 0.5:
                mask_size = torch.randint(2, 6, (1,)).item()
                start = torch.randint(0, x_tensor.shape[0] - mask_size, (1,)).item()
                x_tensor[start : start + mask_size, :] = 0
        return x_tensor, self.y[idx]
