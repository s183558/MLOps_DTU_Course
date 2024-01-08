from MLOps_DTU_project import MyAwesomeModel
import torch

def test_model():
    batch_size = 256
    # Create model
    model = MyAwesomeModel()

    # Load data and begin training
    train_set = torch.load('data/processed/traindata.pt')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    x, _ = train_dataloader.__iter__().__next__()

    # One training loop of one image
    y_pred = model(x)
    assert y_pred.shape == (batch_size, 10), "Wrong shape of y_pred"
