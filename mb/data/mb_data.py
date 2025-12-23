import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MullerBrownDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        positions = data["position_x"]
        forces = data["force_x_unbiased"]
        self.x = positions.reshape(-1, 1).astype(np.float32)
        self.fx = forces.reshape(-1, 1).astype(np.float32)
        self.x = torch.from_numpy(self.x)
        self.fx = torch.from_numpy(self.fx)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.fx[idx]
    
def get_mb_dataloader(filepath,
                      batch_size=32, num_samples=20000):
    dataset = MullerBrownDataset(filepath)
    print(f"Dataset size: {len(dataset)}")
    sampler = torch.utils.data.SubsetRandomSampler(np.random.choice(len(dataset), num_samples, replace=False))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)
    return dataloader

if __name__ == "__main__":
    dataloader = get_mb_dataloader()
    for x_batch, fx_batch, weight_batch in dataloader:
        print(f"x_batch: {x_batch.shape}, fx_batch: {fx_batch.shape}, weight_batch: {weight_batch}")
        break   
