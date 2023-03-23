import tensorflow as tf
from keras import layers, losses, optimizers
from arquitetura_NN import CNNEncoder, EnconderTransformer, FCLayer  # , CNNTransformer
import numpy as np
from dataset import autenticacoes_permissoes, rotulo_e_batimentos

# hyperparametros
num_layers_transformers = 2
num_layers_cnn = 4
num_filters = 32
d_model = 64
dff = 64
num_heads = 4
dropout_rate = 0.1
kernel_size = 3
max_pool_size = 2
final_neurons = 128
num_layers_dense = 1
num_adversarios = 100


def main():
    dfs = rotulo_e_batimentos('butterworth', 'ecg')
    df = autenticacoes_permissoes(dfs['CapnoBase'], num_adversarios)
    model = tf.keras.Sequential([layers.Input(shape=(2, 120, 1)),
                                 CNNEncoder(num_layers=num_layers_cnn, filters=num_filters,
                                            kernel_size=kernel_size, max_pool_size=max_pool_size),
                                 layers.Flatten(),
                                 EnconderTransformer(num_layers=num_layers_transformers, d_model=d_model,
                                                     num_heads=num_heads, dff=dff,
                                                     dropout_rate=dropout_rate),
                                 layers.Flatten(),
                                 FCLayer(num_layers=num_layers_dense, neurons=final_neurons)])
    model.summary()
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    df['x'] = np.expand_dims(np.array(df['x']), axis=-1)
    df['y'] = np.expand_dims(np.array(df['y']), axis=-1)
    model.fit(df['x'], df['y'], epochs=10, batch_size=16)


if __name__ == '__main__':
    main()
