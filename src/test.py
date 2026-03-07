import numpy as np
import argparse

from ann.neural_network import NeuralNetwork

best_config= argparse.Namespace(
            dataset="mnist",
            epochs=10,
            batch_size=128,
            loss="cross_entropy",
            optimizer="rmsprop",
            weight_decay=6.939204254478591e-06,
            learning_rate=0.0006512770150461274,
            hidden_size=[128, 128, 128],
            activation="relu",
            weight_init="xavier"
        )

model = NeuralNetwork(best_config)

weights = np.load("best_model.npy", allow_pickle=True).item()

model.set_weights(weights)