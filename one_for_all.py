import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
import flwr as fl
from multiprocessing import Process
import warnings
import os
import json
from simple_lstm_model import create_simple_lstm_model
from simple_lstm_model import create_simple_lstm_model
from basic_cnn_model import create_basic_cnn_model
from simple_hybrid_model import create_simple_hybrid_model
from simple_gru_model import create_simple_gru_model

# Then use any of them in your federated learning client:
model = create_simple_lstm_model()  # or any other model
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LightweightVehicleEmissionClient(fl.client.NumPyClient):
    def __init__(self, client_id, client_data):
        self.client_id = client_id
        self.model = create_simple_gru_model()
        self.x_train, self.y_train = client_data

        # Enhanced callbacks for better training
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'client_{client_id}_model.keras',
                monitor='loss',
                save_best_only=True
            )
        ]

    def get_parameters(self, config):
        return [np.array(layer) for layer in self.model.get_weights()]

    def fit(self, parameters, config):
        if parameters:
            self.model.set_weights(parameters)

        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=5,
            batch_size=32,
            callbacks=self.callbacks,
            validation_split=0.2,
            verbose=1
        )

        # Get updated parameters
        parameters = self.get_parameters(config)

        # Compress parameters for efficient transmission
        compressed_parameters = [param.astype(
            np.float16) for param in parameters]

        return compressed_parameters, len(self.x_train), {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'mae': history.history['mae'][-1]
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"mae": mae}


def load_and_split_data():
    """Load preprocessed data and split for federated learning"""
    X = np.load('preprocessed_X.npy')
    y = np.load('preprocessed_y.npy')

    # Load scaler parameters for reference
    with open('scaler_params.json', 'r') as f:
        scaler_params = json.load(f)

    total_samples = len(X)
    samples_per_client = total_samples // 3

    # Shuffle data
    indices = np.random.permutation(total_samples)
    X = X[indices]
    y = y[indices]

    # Split data for three clients
    client_data = [
        (X[i * samples_per_client:(i + 1) * samples_per_client],
         y[i * samples_per_client:(i + 1) * samples_per_client])
        for i in range(3)
    ]

    return client_data


def start_client(client_id, client_data):
    client = LightweightVehicleEmissionClient(client_id, client_data)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client
    )


def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )


def run_federated_learning():
    # Load and split the data
    client_data = load_and_split_data()

    # Start server process
    server_process = Process(target=start_server)
    server_process.start()
    time.sleep(3)  # Give the server time to start

    # Start client processes
    client_processes = []
    for i in range(3):
        p = Process(target=start_client, args=(i, client_data[i]))
        client_processes.append(p)
        p.start()
        time.sleep(0.5)  # Stagger client starts

    # Wait for completion
    for p in client_processes:
        p.join()

    server_process.terminate()
    server_process.join()


def plot_metrics(history):
    """Plot training metrics"""
    metrics = ['loss', 'mae']
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))

    for i, metric in enumerate(metrics):
        axes[i].plot(history.history[metric])
        axes[i].plot(history.history[f'val_{metric}'])
        axes[i].set_title(f'Model {metric}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_federated_learning()
