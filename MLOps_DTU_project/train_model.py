from torch import nn, optim
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        # Input to a hidden layer
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
    
    def save_model(self, location):
        torch.save(self.state_dict(), f'MLOps_DTU_project/models/session1/{location}')

    def training_loop(self, lr, train_set, epochs = 40):
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
        plt.savefig('reports/figures/final_exercise_S1/train_loss.png')

            
