# simple_gru_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model


class SimpleGRUModel(Model):
    def __init__(self, seq_length=20, n_features=7):
        super().__init__()

        # First GRU layer
        self.gru1 = layers.GRU(32,
                               return_sequences=True,
                               input_shape=(seq_length, n_features))
        self.dropout1 = layers.Dropout(0.2)

        # Second GRU layer
        self.gru2 = layers.GRU(16,
                               return_sequences=False)
        self.dropout2 = layers.Dropout(0.2)

        # Dense layers for prediction
        self.dense = layers.Dense(8, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        x = self.gru1(inputs)
        x = self.dropout1(x)
        x = self.gru2(x)
        x = self.dropout2(x)
        x = self.dense(x)
        return self.output_layer(x)


def create_simple_gru_model():
    """Create and compile the Simple GRU model"""
    model = SimpleGRUModel()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Simple mean squared error loss
        metrics=['mae']  # Mean absolute error for evaluation
    )
    return model


if __name__ == "__main__":
    # Example usage
    model = create_simple_gru_model()
    print("\nSimple GRU Model Summary:")
    print("- Two GRU layers (32 and 16 units)")
    print("- Dropout for regularization")
    print("- Single dense layer before output")
    print("- Basic MSE loss function")

    # Print model summary
    x = tf.random.normal((1, 20, 7))  # Example input
    y = model(x)
    model.summary()
