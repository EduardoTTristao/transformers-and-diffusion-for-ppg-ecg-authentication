import numpy as np
from keras import layers
import tensorflow as tf
from keras.models import Model

latent_dim = 64


class CNNEnconderLayer(layers.Layer):
    def __init__(self, filters, kernel_size, max_pool_size):
        super(CNNEnconderLayer, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"),
            layers.MaxPooling2D(pool_size=(1, max_pool_size))
        ])

    def call(self, x):
        x = self.encoder(x)
        return x


class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, filters, kernel_size, max_pool_size):
        super().__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            CNNEnconderLayer(filters, kernel_size, max_pool_size)
            for _ in range(num_layers)]

        # self.enc_layers.insert(0, layers.Input(shape=(2, 120, 1)))

    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, vocab_size=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderTransformerLayer(layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = BaseAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class EnconderTransformer(layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding(d_model=d_model)

        self.enc_layers = [
            EncoderTransformerLayer(d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):

        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


class FCLayer(layers.Layer):
    def __init__(self, *, num_layers, neurons):
        super().__init__()

        self.num_layers = num_layers

        self.dense_layers = []

        for _ in range(num_layers):
            self.dense_layers.append(layers.Dense(neurons, activation='relu'))
            self.dense_layers.append(layers.BatchNormalization())
        self.dense_layers.append(layers.Dense(2, activation='sigmoid'))

    def call(self, x):
        for i in range(self.num_layers+1):
            x = self.dense_layers[i](x)

        return x


'''class CNNTransformer(tf.keras.Model):
    def __init__(self, *, num_layers_transformers, num_layers_cnn, num_filters, d_model, num_heads, dff,
                 dropout_rate=0.1, kernel_size, max_pool_size, final_neurons, num_layers_dense):
        super().__init__()

        self.enconder_cnn = CNNEncoder(num_layers=num_layers_cnn, filters=num_filters, kernel_size=kernel_size,
                                       max_pool_size=max_pool_size)

        self.flatten = layers.Flatten()

        self.encoder_trans = EnconderTransformer(num_layers=num_layers_transformers, d_model=d_model,
                                                 num_heads=num_heads, dff=dff,
                                                 dropout_rate=dropout_rate)

        self.final_layer = FCLayer(num_layers=num_layers_dense, neurons=final_neurons)

    def call(self, x):

        x = self.enconder_cnn(x)

        x = self.flatten(x)

        x = self.encoder_trans(x)

        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits'''