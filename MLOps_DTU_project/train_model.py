from torch import nn, optim
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        # Input to a hidden layer
        self.cnn1 = nn.Conv2d(1, 32, 3)  # [B, 1, 28, 28] -> [B, 32, 26, 26]
        self.cnn2 = nn.Conv2d(32, 64, 3) # [B, 32, 26, 26] -> [B, 64, 24, 24]
        self.maxpool2d = nn.MaxPool2d(2)      # [B, 64, 24, 24] -> [B, 64, 12, 12]
        self.flatten = nn.Flatten(0, -1)        # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
        self.fc = nn.Linear(64 * 12 * 12, 10)

        self.LRelu = nn.LeakyReLU()
        #self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.LRelu(self.cnn1(x))
        x = self.LRelu(self.cnn2(x))
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
    
    def save_model(self, fname):
        torch.save(self.state_dict(), f'MLOps_DTU_project/models/{fname}')

    def training_loop(self, lr, train_set, epochs = 5):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_loss = []
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                
                optimizer.zero_grad()
                
                log_ps = self(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            else:
                t_loss = running_loss/len(train_set)
                train_loss.append(t_loss)
                print(f"Training loss: {t_loss}")
        
        
        plt.plot(train_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.savefig('reports/figures/train_loss.png')

if __name__ == '__main__':
    # Create model
    model = MyAwesomeModel()

    # Load data and begin training
    train_set = torch.load('data/processed/traindata.pt')
    model.training_loop(1e-3, train_set)
    
    # Save model
    model.save_model("trained_model_Session1.pth")
    pass
