import pickle
import os
import numpy as np

from functools import partial

CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32
CIFAR_CHANNEL = 3


default_dataset_path = os.path.join(
    os.path.dirname(__file__), "dataset"
)
default_cifar10_path = os.path.join(
    default_dataset_path, "cifar-10-batches-py"
)


def get_cifar_batch(
    file_name: str
):
    with open(file_name, 'rb') as f:
        batch_data = pickle.load(
            f, encoding='bytes'
        )
        batch_data[b"data"] = batch_data[b"data"]# / 255
        return batch_data[b"data"], batch_data[b"labels"]

def get_cifar10_data(
    dataset_path: str = default_cifar10_path,
    num_samples_train: int = 50000,
    shuffle: bool = False,
    return_image: bool = False,
    feature_process: any = None,
    subset_train: int = None,
    subset_test: int = None
):
    x_train = []
    y_train = []
    for i in range(1, 6):
        x_batch, y_batch = get_cifar_batch(
            os.path.join(dataset_path, "data_batch_{}".format(i))
        )
        x_train.append(x_batch)
        y_train.append(y_batch)


    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    assert num_samples_train == 5 * 10000

    x_test, y_test = get_cifar_batch(
        os.path.join(dataset_path, "test_batch")
    )
    y_test = np.array(y_test)

    if subset_train is None:
        subset_train = num_samples_train
    if subset_test is None:
        subset_test = 10000
    dataset = {
        "x_train": x_train[:subset_train],
        "y_train": y_train[:subset_train],
        "x_test": x_test[:subset_test],
        "y_test": y_test[:subset_test]
    }
    if return_image:
        dataset["x_train"] = dataset["x_train"].reshape(
            (-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT)
        )
        dataset["x_test"] = dataset["x_test"].reshape(
            (-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT)
        )
    return dataset