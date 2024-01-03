import torch
import torchvision

def loadCorrupteMnist():
    """Return train and test dataloaders for MNIST."""
    # Load the 5 parts of the training data
    train_data, train_labels = [], []
    for i in range(5):
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/train_target_{i}.pt"))
    
    # Make them into 1 tensor
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Load the test data
    test_data = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")

    # normalize the images
    train_data_norm = normalize_tensor(train_data)
    test_data_norm = normalize_tensor(test_data)

    # make the images 3D, with only 1 channel
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # return the images and labels in a tuple
    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )

def normalize_tensor(data):
    # Calculate mean and std along dimensions (1, 2)
    mean = data.mean(dim=(1, 2), keepdim=True)
    std = data.std(dim=(1, 2), keepdim=True)

    # Normalize the data
    normalized_data = (data - mean) / std

    # Check mean and std of normalized_data
    mean_check = normalized_data.mean(dim=(1, 2))
    std_check = normalized_data.std(dim=(1, 2))

    # Define a tolerance level
    tolerance = 1e-6

    # Assert that mean is close to 0 and std is close to 1
    assert torch.all(torch.abs(mean_check) < tolerance), "Mean check failed!"
    assert torch.all(torch.abs(std_check - 1) < tolerance), "Std check failed!"

    print("Normalization check passed!")
    #print(f"Normalized data mean={normalized_data.mean(dim=(1, 2), keepdim=True)}, std={normalized_data.std(dim=(1, 2), keepdim=True)}")
    return normalized_data


if __name__ == '__main__':
    # Get the data and process it
    train, test = loadCorrupteMnist()

    torch.save(train, "data/processed/traindata.pt")
    torch.save(test, "data/processed/testdata.pt")

    pass