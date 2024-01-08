import torch
from MLOps_DTU_project.models.model import MyAwesomeModel
import glob
import os

def predict():
    # Find the latest file in the folder
    list_of_files = glob.glob('MLOps_DTU_project/models/model_checkpoints/*')
    latest_file = os.path.basename(max(list_of_files, key=os.path.getctime))
    print(f'\nWill run inference on model: \"{latest_file}\"\n')

    # Create model
    state_dict = torch.load(f'MLOps_DTU_project/models/model_checkpoints/{latest_file}')
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    # Load data and begin training
    print(f'Running inference on the model, please wait :)')
    test_set = torch.load('data/processed/testdata.pt')
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle = False)

    model.inference(test_dataloader)

if __name__ == '__main__':
    predict()
