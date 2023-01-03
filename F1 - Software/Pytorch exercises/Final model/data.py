import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class mnist(Dataset):
    def __init__(self, *filepaths):
        """ Function: Load files from file paths and concatenate into a single dataset
        """

        # List of content from each training npz-file
        images = [np.load(file)['images'] for file in filepaths]
        labels = [np.load(file)['labels'] for file in filepaths]

        # Concatenate the content into a single dataset
        self.imgs = np.concatenate(images)
        self.labels = np.concatenate(labels)

        # Transform
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        imgs, labels = (self.imgs[idx], self.labels[idx])

        if self.transform:
            imgs = self.transform(imgs)

        return imgs.float(), labels
