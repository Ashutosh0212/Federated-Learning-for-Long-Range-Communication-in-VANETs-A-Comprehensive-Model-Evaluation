# basic_cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model


class BasicCNNModel(Model):
    def __init__(self, seq_length=20, n_features=7):
        super().__init__()

        # First convolutional layer
        self.conv1 = layers.Conv1D(
            filters=16,
            kernel_size=3,
            padding='same',
            activation='relu',
            input_shape=(seq_length, n_features)
        )

        # Second convolutional layer
        self.conv2 = layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu'
        )

        # Global pooling layer
        self.global_pool = layers.GlobalAveragePooling1D()

        # Dense layers
        self.dense = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.dense(x)
        return self.output_layer(x)


def create_basic_cnn_model():
    """Create and compile the Basic CNN model"""
    model = BasicCNNModel()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Simple mean squared error loss
        metrics=['mae']  # Mean absolute error for evaluation
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_basic_cnn_model()
    print("\nBasic CNN Model Summary:")
    print("- Two Conv1D layers (16 and 32 filters)")
    print("- Global average pooling")
    print("- Single dense layer before output")
    print("- Basic MSE loss function")

    # Print model summary
    x = tf.random.normal((1, 20, 7))  # Example input
    y = model(x)
    model.summary()
