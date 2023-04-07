import tensorflow as tf
from keras import layers, losses, optimizers, metrics
from arquitetura_NN import CNNEncoder, EnconderTransformer, FCLayer  # , CNNTransformer
import numpy as np
from dataset import autenticacoes_permissoes, rotulo_e_batimentos
import pandas as pd
import datetime
import tensorflow_addons as tfa

# hyperparametros
num_layers_transformers = 2
num_layers_cnn = 4
num_filters = 64
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.1
kernel_size = 3
max_pool_size = 2
final_neurons = 256
num_layers_dense = 3
neurons_input_transformers = 256
num_adversarios = 100
GPU = 1


def main():
    with tf.device('/device:GPU:'+str(GPU)):
        for sinal_avaliado in ['ecg', 'ppg']:
            dfs = rotulo_e_batimentos('butterworth', sinal_avaliado)
            df = autenticacoes_permissoes(pd.concat([dfs['CapnoBase'], dfs['BIDMC'], dfs['TROIKA']]), num_adversarios)
            dfs = 0
            model = tf.keras.Sequential([layers.Input(shape=(2, 120, 1)),
                                         layers.BatchNormalization(),
                                         CNNEncoder(num_layers=num_layers_cnn, filters=num_filters,
                                                    kernel_size=kernel_size, max_pool_size=max_pool_size),
                                         layers.Flatten(),
                                         EnconderTransformer(num_layers=num_layers_transformers, d_model=d_model,
                                                             num_heads=num_heads, dff=dff,
                                                             dropout_rate=dropout_rate),
                                         layers.Flatten(),
                                         FCLayer(num_layers=num_layers_dense, neurons=final_neurons)])
            model.summary()
            model.compile(optimizer=optimizers.Adam(0.001),
                          loss=losses.SparseCategoricalCrossentropy(), metrics=[metrics.SparseCategoricalAccuracy()])

            checkpoint_path = "training_1/"+sinal_avaliado+"/"+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+"/cp.ckpt"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

            log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            df['x'] = np.expand_dims(np.array(df['x']), axis=-1)
            df['y'] = np.expand_dims(np.array(df['y']), axis=-1)
            model.fit(df['x'], df['y'], epochs=20, batch_size=32, callbacks=[tensorboard_callback, cp_callback])


if __name__ == '__main__':
    main()
