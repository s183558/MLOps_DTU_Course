import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from MLOps_DTU_project.train_model import MyAwesomeModel

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('reports/figures/visualise_CNN2_features.png')

def visualization(model: torch.nn.Module):
    filter = model.cnn2.weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)



if __name__ == "__main__":
    model = MyAwesomeModel()
    state_dict = torch.load(f'MLOps_DTU_project/models/trained_model_Session1.pth')
    model.load_state_dict(state_dict)
    visualization(model)

    

