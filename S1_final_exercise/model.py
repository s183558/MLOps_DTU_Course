from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import time 


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        # Input to a hidden layer
        self.cnn1 = nn.Conv2d(1, 32, 3)  # [B, 1, 28, 28] -> [B, 32, 26, 26]
        self.cnn2 = nn.Conv2d(32, 64, 3) # [B, 32, 26, 26] -> [B, 64, 24, 24]
        self.maxpool2d = nn.MaxPool2d(2)      # [B, 64, 24, 24] -> [B, 64, 12, 12]
        self.flatten = nn.Flatten()        # [B, 64, 12, 12] -> [B, 64 * 12 * 12]
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
        torch.save(self.state_dict(), f'MLOps_DTU_project/models/model_checkpoints/{fname}_{time.strftime("%d%m%Y-%H%M%S")}')

    def training_loop(self, lr, train_set, num_epochs = 5):
        loss_fn = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_loss = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time_ns()
            for x, y in train_set:
                
                optimizer.zero_grad()
                
                y_pred = self(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                
            # after a whole epoch
            train_loss.append(loss.detach())
            print(f"Epoch {epoch + 1} took {(time.time_ns() - epoch_start_time) * 10e-10:.2f}sec.\tLoss {loss}\n")

        plt.plot(train_loss)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.savefig('reports/figures/train_loss.png')

    def inference(self, test_dataloader):
        test_preds, test_labels = [], []
        # Dont want dropout in here
        self.eval()

        # Begin validation
        with torch.no_grad():
            for x, y in test_dataloader:
                # x = x.to(device)
                y_pred = self(x)                                # Find the probabilities of the validation set ran through the model
                test_preds.append(y_pred.argmax(dim=1).cpu())   # Find the top probability of each image
                test_labels.append(y)

        test_preds = torch.cat(test_preds, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        print(f'Accuracy: {(test_preds == test_labels).float().mean()}')    # Find the percentage of correct "guesses"