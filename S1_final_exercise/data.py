import torch
from torchvision import transforms

def mnist():
    """Return train and test dataloaders for MNIST."""
    trainset = []
    testset  = []
    # Make a tuple where the first elements are the images, and the second are the corresponding targets

    for im, label in zip(torch.load("data/train_images_0.pt"), torch.load("data/train_target_0.pt")):
        trainset.append((im[None, :, :], label.item()))
    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    for im, label in zip(torch.load("data/test_images.pt"), torch.load("data/test_target.pt")):
        testset.append((im[None, :, :], label.item()))
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    #train_im = torch.load("data/corruptmnist/train_images_0.pt")
    #print(f'\nTraining image 0 shape: {train_im[0].shape}\n')
    #print(f'\nTraining image 1 shape: {train_im[1].shape}\n')
    #print(f'\nTraining image -1 shape: {train_im[-1].shape}\n')
    #trainset = zip(torch.load("data/corruptmnist/train_images_0.pt"), torch.load("data/corruptmnist/train_target_0.pt"))
    #print(f'\nTraining set 0: {trainset[0]}\n')
    #print(f'\nTraining set -1: {trainset[-1]}\n')
    #train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    #testset = (torch.load("data/corruptmnist/test_images.pt"), torch.load("data/corruptmnist/test_target.pt"))
    #test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return train, test
