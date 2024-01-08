import pytest
from MLOps_DTU_project import MyAwesomeModel
import torch

@pytest.mark.parametrize("batch_size", [64, 128, 256])
def test_model(batch_size):
    # Create model
    model = MyAwesomeModel()

    # Load data and begin training
    train_set = torch.load('data/processed/traindata.pt')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    x, _ = train_dataloader.__iter__().__next__()

    # One training loop of one image
    y_pred = model(x)
    assert y_pred.shape == (batch_size, 10), "Wrong shape of y_pred"

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))