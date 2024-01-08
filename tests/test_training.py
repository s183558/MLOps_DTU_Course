import torch
from MLOps_DTU_project.models.model import MyAwesomeModel


def test_training():
    batch_size = 256
    lr = 1e-3
    epochs = 2
    # Create model
    model = MyAwesomeModel()

    # Load data and begin training
    train_set = torch.load("data/processed/traindata.pt")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    model.training_loop(lr, train_dataloader, epochs)
