from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, df, preprocessing):
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocessing(Image.open(self.images[idx]))  # preprocess from clip.load
        caption = self.caption[idx]
        return images, caption


