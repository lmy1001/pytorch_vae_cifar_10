from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import numpy as np
import random
import torch
from run import parse_args


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = pickle.load(f,encoding='latin1')
        print(datadict.keys())
        train_data = datadict['data']

        train_label = datadict['labels']
        train_label = np.array(train_label)
        return train_data, train_label

def data_normalize(data, use_norm_shift = False, use_norm_scale = True):
    pixel_depth = 255
    if use_norm_shift:
        data = data - (pixel_depth / 2.0)
    if use_norm_scale:
        data = data / float(pixel_depth)
    return data

# Prepare Cifar-10 data
def prepare_cifar_10_data(use_norm_shift=False, use_norm_scale=True):
    validation_data_size = 5000  # Size of the validation set.
    train_data = []
    train_label = []
    args = parse_args()
    for id in range(1, 6):
        train_filename = os.path.join(args.data_dir, "data_batch_%d"%id)
        train_data_batch, train_label_batch = load_CIFAR_batch(train_filename)
        train_data.append(train_data_batch)
        train_label.append(train_label_batch)

    train_data = np.concatenate(train_data)         #50000 * 3072(32 * 32 * 3)
    train_label = np.concatenate(train_label).reshape(-1, 1)       #50000 * 1

    test_filename = os.path.join(args.data_dir, "test_batch")
    test_data, test_label = load_CIFAR_batch(test_filename)
    test_label = test_label.reshape(-1, 1)

    # Generate a validation set.
    validation_data = train_data[:validation_data_size, :]
    validation_labels = train_label[:validation_data_size, :]
    train_data = train_data[validation_data_size:, :]
    train_label = train_label[validation_data_size:, :]

    #data normalization
    train_data = data_normalize(train_data, use_norm_shift, use_norm_scale)
    test_data = data_normalize(test_data, use_norm_shift, use_norm_scale)
    validation_data = data_normalize(validation_data, use_norm_shift, use_norm_scale)

    return train_data, train_label, validation_data, validation_labels, test_data, test_label

def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    train_data, train_label,validation_data, validation_labels, test_data, test_label = prepare_cifar_10_data()