from MLOps_DTU_project import loadCorrupteMnist
import os.path
import pytest
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    N_train = 25000
    N_test = 5000
    train_data, test_data = loadCorrupteMnist()
    # Check that the right amount of data is loaded
    assert len(train_data) == N_train, "Dataset did not have the correct number of samples"
    assert len(test_data) == N_test, "Dataset did not have the correct number of samples"

    # Look at the images in the data, and asser their shape
    assert train_data.tensors[0].shape == (
        N_train,
        1,
        28,
        28,
    ), f"Images in the training dataset did not have the correct shape (N, 1, 28, 28) != {train_data.tensors[0].shape}"
    assert test_data.tensors[0].shape == (
        N_test,
        1,
        28,
        28,
    ), f"Images in the test dataset did not have the correct shape (N, 1, 28, 28) != {test_data.tensors[0].shape}"

    # Make sure that all labels are present in the data
    assert (
        train_data.tensors[1].unique().tolist() == list(range(10))
    ), f"Training dataset did not have all labels represented. {list(range(10))} != {train_data.tensors[1].unique().tolist()}"
    assert (
        test_data.tensors[1].unique().tolist() == list(range(10))
    ), f"Test dataset did not have all labels represented. {list(range(10))} != {test_data.tensors[1].unique().tolist()}"
