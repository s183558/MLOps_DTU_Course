import click
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night, with lr = {lr}")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    print(train_set[0].shape)

    train_set2 = torch.load('data/processed/traindata.pt')
    print(train_set2[0].shape)

    model.training_loop(lr, train_set)
    
    # Save model
    model.save_model("trained_model_Session1.pth")



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it, with filename \"{model_checkpoint}\"")

    # TODO: Implement evaluation logic here
    state_dict = torch.load(f's1_development_environment/exercise_files/final_exercise/{model_checkpoint}')
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)

    _, test_set = mnist()

    vali_labels = []
    vali_topclass = []
    # turn of gradients
    with torch.no_grad():
        # Dont wan dropout in here

        model.eval()

        # Begin validation
        for vali_image, vali_label in test_set:
            ps = torch.exp(model(vali_image))                      # Find the probabilities of the validation set ran through the model
            top_p, top_class = ps.topk(1, dim=1)                    # Find the top probability of each image

            vali_labels.append(vali_label)
            vali_topclass.append(top_class)
              
    # Each iteration in the loop is of a batch of 64, as long as 64 images can be retrieved from the data set.
    # Therefor we stack all but the last entry in the list, as they have similar size.
    # We then concat that tensor with the last entry in the list.
    vali_labels_start = torch.stack(vali_labels[:-1]).flatten()
    vali_labels_end   = vali_labels[-1].flatten()
    validation_labels = torch.cat((vali_labels_start, vali_labels_end)) 
    vali_topclass_start = torch.stack(vali_topclass[:-1]).flatten()
    vali_topclass_end   = vali_topclass[-1].flatten()
    validation_results  = torch.cat((vali_topclass_start, vali_topclass_end)) 

    equals = validation_results == validation_labels            # Which labels overlap
    accuracy = torch.mean(equals.type(torch.FloatTensor))       # Find the percentage of correct "guesses"
    print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
