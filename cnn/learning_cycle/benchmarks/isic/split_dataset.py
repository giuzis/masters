import torch
from pathlib import Path
import os
import pandas as pd
from PIL import Image

def _split_dataset(dataset, batch_size, train_ratio, val_ratio, test_ratio):
    calc_size = lambda x: int(x * len(dataset)) - int(x * len(dataset)) % batch_size
    train_size =calc_size(train_ratio)
    val_size = calc_size(val_ratio) 
    test_size = calc_size(test_ratio)
    train_size += len(dataset) - train_size - val_size - test_size
    return torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
)


class ISICDataset(Dataset):
    def __init__(self, path_images: str, csv_file: str, trasform = None):
        all_files = self._get_all_image_files_from(path_images)
        self.labels = self._get_labels(csv_file) 
        self.data = [path for path in all_files if path.stem in self.labels.keys()]
        self.transform = transform
        
        if len(self.data) != len(self.labels):
            raise ValueError('Number of images and labels are not equal') 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _id = self.data[idx].stem
        img = Image.open(self.data[idx])
        label = torch.tensor(self.labels[_id], dtype=torch.long)
        return self.transform(img), label

    def _get_labels(self, csv_file):
        df = pd.read_csv(csv_file)
        df.loc[:, 'label'] = 0
        df.loc[df.melanoma == 1.0, 'label'] = 1
        df.loc[df.seborrheic_keratosis == 1.0, 'label'] = 2
        return df.set_index('image_id')['label'].to_dict()

    def _get_all_image_files_from(self, path):
        return sorted([
            Path(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(path)
            for filename in filenames
            if filename.endswith('.jpg')
        ])
