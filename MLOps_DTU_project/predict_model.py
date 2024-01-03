import torch

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    state_dict = torch.load(f'MLOps_DTU_project/models/session1/trained_model.pt')
    model.load_state_dict(state_dict)

    test_set = torch.load("data/processed/testdata.pt")

    vali_labels, vali_topclass = [], []
    # turn of gradients
    with torch.no_grad():
        # Dont want dropout in here

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







    return torch.cat([model(batch) for batch in dataloader], 0)