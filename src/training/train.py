#!/usr/bin/python3
import torch
import copy
import dataloader

def train_model(model, optimizer, scheduler, num_epochs=60):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    dataloaders = dataloader.get_data_loaders(batch_size=64)
