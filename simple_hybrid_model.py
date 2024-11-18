# simple_hybrid_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model


class SimpleHybridModel(Model):
    def __init__(self, seq_length=20, n_features=7):
        super().__init__()

        # CNN part - single convolutional layer
        self.conv = layers.Conv1D(
            filters=16,
            kernel_size=3,
            padding='same',
            activation='relu',
            input_shape=(seq_length, n_features)
        )

        # Simple RNN layer
        self.rnn = layers.SimpleRNN(
            16,
            return_sequences=False
        )

        # Dense layers
        self.dense1 = layers.Dense(8, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        # Apply CNN
        x = self.conv(inputs)

        # Apply RNN
        x = self.rnn(x)

        # Dense layers
        x = self.dense1(x)
        return self.output_layer(x)


def create_simple_hybrid_model():
    """Create and compile the Simple Hybrid model"""
    model = SimpleHybridModel()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Simple mean squared error loss
        metrics=['mae']  # Mean absolute error for evaluation
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_simple_hybrid_model()
    print("\nSimple Hybrid Model Summary:")
    print("- One Conv1D layer (16 filters)")
    print("- One SimpleRNN layer (16 units)")
    print("- Single dense layer before output")
    print("- Basic MSE loss function")

    # Print model summary
    x = tf.random.normal((1, 20, 7))  # Example input
    y = model(x)
    model.summary()
