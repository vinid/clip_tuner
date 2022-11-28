"""Main module."""
from comet_ml import Experiment
from torch import nn
from torch import optim
import clip
import tqdm
import torch
from clip_tuner.dataset import HFDataset
from torch.utils.data import DataLoader
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel






class CLIPTuner:

    def __init__(self, lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2, comet_tracking=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if comet_tracking:
            self.experiment = Experiment(comet_tracking)
        else:
            self.experiment = Experiment()

        self.model = self.model.to(self.device)

        hyper_params = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        self.experiment.log_parameters(hyper_params)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                    lr=hyper_params["lr"],
                                    betas=hyper_params["betas"],
                                    eps=hyper_params["eps"],
                                    weight_decay=hyper_params["weight_decay"])

    def tuner(self, train_dataframe, validation_dataframe, batch_size=4, epochs=5, evaluation_steps=500):
        train_dataset = HFDataset(train_dataframe)
        validation_dataset = HFDataset(validation_dataframe)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

        step = 0
        with self.experiment.train():

            for epoch in range(epochs):
                pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
                pbar.set_description(f"{epoch}/{epochs}")

                for batch in train_dataloader:

                    self.optimizer.zero_grad()

                    list_image, list_txt = batch

                    images = images.to(self.device)
                    inputs = self.processor(text=list_txt, images=list_image, return_tensors="pt",
                                       padding=True)
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_image.T

                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                    total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,
                                                                                                ground_truth)) / 2
                    self.experiment.log_metric("loss", total_loss.item(), step=step)
                    step = step + 1

                    total_loss.backward()
                    self.optimizer.step()


                if step % evaluation_steps == 0:

                    for batch in validation_dataloader:
                        pbar.set_description("Currently Validating")

                        with torch.no_grad():

                            list_image, list_txt = batch

                            images = list_image

                            inputs = self.processor(text=list_txt, images=list_image, return_tensors="pt",
                                                    padding=True)
                            outputs = self.model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            logits_per_text = outputs.logits_per_image.T

                            ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

                            total_loss = (self.loss_img(logits_per_image, ground_truth) +
                                          self.loss_txt(logits_per_text, ground_truth)) / 2

                            self.experiment.log_metric("validation_loss", total_loss.item(), step=step)

                pbar.close()
