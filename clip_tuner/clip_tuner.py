"""Main module."""
from torch import nn
from torch import optim
import clip
import torch
from clip_tuner.dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class CLIPTuner:

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)  # Must set jit=False for training

        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=5e-5,
                                    betas=(0.9, 0.98),
                                    eps=1e-6,
                                    weight_decay=0.2)

    def tuner(self, dataframe, batch_size=4, epochs=5):
        dataset = ImageCaptioningDataset(dataframe, self.preprocess)
        train_dataloader = DataLoader(dataset, batch_size=4)  # Define your own dataloader

        for epoch in range(epochs):
            for batch in train_dataloader:
                self.optimizer.zero_grad()

                list_image, list_txt = batch

                images = list_image
                images = images.to(self.device)
                texts = clip.tokenize(list_txt).to(self.device)

                logits_per_image, logits_per_text = self.model(images, texts)

                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,
                                                                                            ground_truth)) / 2
                total_loss.backward()
                if self.device == "cpu":
                    self.optimizer.step()
                else:
                    convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)
