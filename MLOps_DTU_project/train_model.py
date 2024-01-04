import torch
from models.model import MyAwesomeModel

if __name__ == '__main__':
    # Create model
    model = MyAwesomeModel()

    # Load data and begin training
    print(f'Beginning to train the model, please wait')
    train_set = torch.load('data/processed/traindata.pt')
    model.training_loop(1e-3, train_set)
    
    # Save model
    model.save_model("trained_model_Session1.pth")

