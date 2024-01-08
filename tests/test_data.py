from MLOps_DTU_project import loadCorrupteMnist

def test_data():
    N_train = 25000
    N_test = 5000
    train_data, test_data = loadCorrupteMnist()
    # Check that the right amount of data is loaded
    assert len(train_data) == N_train
    assert len(test_data) == N_test

    # Look at the images in the data, and asser their shape
    assert train_data.tensors[0].shape == (N_train, 1, 28, 28)
    assert test_data.tensors[0].shape == (N_test, 1, 28, 28)

    # Make sure that all labels are present in the data
    assert train_data.tensors[1].unique().tolist() == list(range(10))
    assert test_data.tensors[1].unique().tolist() == list(range(10))