import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical

FASHION_MNIST_LABELS=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def log_data_samples(X_raw, y_raw, dataset_name):
    sample_table=wandb.Table(columns=['class label', 'Image'])
    for class_id in range(10):
        label_idx=np.where(y_raw==class_id)[0]
        sample_idx=label_idx[:5]
        for idx in sample_idx:
            img=X_raw[idx]
            label_name=(FASHION_MNIST_LABELS[class_id] if dataset_name=='fashion_mnist' else str(class_id))
            sample_table.add_data(label_name, wandb.Image(img))
    wandb.log({"Data Exploration/Sample Images": sample_table})

def load_dataset(dataset_name):
    print(f"Loading {dataset_name}...")

    if dataset_name=='mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name=='fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_train=X_train.astype('float32')/255.0
    X_test=X_test.astype('float32')/255.0
    X_train=X_train.reshape(len(X_train), -1).T
    X_test=X_test.reshape(len(X_test),  -1).T    
    return X_train, y_train, X_test, y_test


def one_hot(y, num_classes=10):
    return to_categorical(y, num_classes=num_classes).T