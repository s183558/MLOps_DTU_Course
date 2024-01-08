import click
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from model import MyAwesomeModel
import time
from data import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print(f"Training day and night, with lr = {lr}")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    model.training_loop(lr, train_dataloader, num_epochs)

    # Save model
    model.save_model("trained_model_Session1.pth")


@click.command()
@click.option("--model_checkpoint", default="trained_model_Session1_04012024-172630.pth", help="The filename of the checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it, with filename \"{model_checkpoint}\"")

    # TODO: Implement evaluation logic here
    state_dict = torch.load(f'MLOps_DTU_project/models/model_checkpoints/{model_checkpoint}')
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    _, test_set = mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle = False)

    model.inference(test_dataloader)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
