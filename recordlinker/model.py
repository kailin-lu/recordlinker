from __future__ import print_function
from __future__ import absolute_import

import keras
from keras import Model
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard, EarlyStopping

TEST_SIZE = 0.2
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

def binarize():
    pass

class VariationalAutoencoder():
    '''Variational autoencoder using dense layers'''
    def __init__(self,
                 latent_units):
        self.latent_units = latent_units

    def _sample_z(self, mu, log_sigma, batch_size):
        '''Sample epsilon from N(0,I) for reparameterization trick'''
        epsilon = keras.random_normal(shape=(batch_size, self.latent_units))
        return mu + keras.exp(log_sigma / 2.) * epsilon

    def _encode(self, input_x, encode_units):
        # encode_layers = []
        # for units in encode_units:
        #     encode_layers.append(Dense(units))
        #mu = Dense(self.latent_units)
        #log_sigma = Dense(self.latent_units)
        #return self._sample_z(mu, log_sigma)
        pass

    def _decode(self, sample_z, decode_units):
        # Save median_mu as part of the computational graph
        pass

    def _build_model(self, input_x, true_x, lr):
        inp = Input()(input_x)
        z = self._encode(inp)
        out = self._decode(z)
        # z_binary = binarize(mu)

        # median_mu = K.variable

        loss = categorical_crossentropy(true_x, out)
        adam = Adam(lr=lr)
        model = Model(inputs=inp, outputs=z_binary)
        model.compile(optimizer=adam, metrics='cross_entropy', loss=loss)
        return model

    def train(self, namesA, namesB, save_path, test_size=TEST_SIZE,
              epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
        # Inputs - check tensor shapes

        model = self._build_model()

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=.0001, patience=3),
                    TensorBoard()]

        model.fit(x=namesA, y=namesB,
                  batch_size=batch_size, epochs=epochs,
                  callbacks=callbacks, validation_split=test_size)

        # Save model
        pass


class LSTMVariationalAutoencoder(VariationalAutoencoder):
    def __init__(self):
        super().__init__()

    def _encode(self):
        pass

    def _decode(self):
        pass


# tf.set_random_seed(5555)
#
# class VariationalAutoencoder():
#     """Variational autoencoder using dense layers"""
#     def __init__(self,
#                  encode_units=[128],
#                  latent_units=16,
#                  batch_size=128,
#                  seed=5555,
#                  learning_rate=1e-3,
#                  model_path=None,
#                  median_mu=None):
#         self.encode_units = encode_units
#         self.decode_units = encode_units[::-1]
#         self.latent_units = latent_units
#         self.batch_size = batch_size
#         self.seed = seed
#         self.learning_rate = learning_rate
#         self.features = None
#         self.encode_layers = []
#         self.decode_layers = []
#         self.model_path = model_path
#         self.median_mu = median_mu
#
#     def _sample_z(self, mu, log_sigma, z_size):
#         # Introduce epsilon for reparameterization trick
#         epsilon = tf.random_normal([z_size, self.latent_units])
#         return mu + tf.exp(log_sigma / 2) * epsilon
#
#     def inputs(self):
#         with tf.name_scope('inputs'):
#             input_x = tf.placeholder(shape=(None, self.features),
#                                      dtype=tf.float32, name='input_x')
#             z_size = tf.placeholder(shape=(), dtype=tf.int32, name='z_size')
#         return input_x, z_size
#
#     def _encode(self, input_x):
#         """Encode with number of hidden layers specified by hidden units"""
#         print('Input:', input_x.shape)
#         self.encode_layers.append(input_x)
#         with tf.name_scope('encode'):
#             for layer in range(len(self.encode_units)):
#                 fc_layer = tf.layers.dense(self.encode_layers[-1],
#                                            units=self.encode_units[layer],
#                                            activation=tf.nn.relu)
#                 print('FC', fc_layer.shape)
#                 self.encode_layers.append(fc_layer)
#
#             mu = tf.layers.dense(self.encode_layers[-1],
#                                  units=self.latent_units, name='mu')
#             log_sigma = tf.layers.dense(self.encode_layers[-1],
#                                         units=self.latent_units,
#                                         name='log_sigma')
#             print('Mu', mu.shape, 'Log Sigma', log_sigma.shape)
#         return mu, log_sigma
#
#     def _decode(self, z):
#         self.decode_layers.append(z)
#         print('Z', z.shape)
#         with tf.name_scope('decode'):
#             for layer in range(len(self.decode_units)):
#                 if layer == len(self.decode_units):
#                     decode_layer = tf.layers.dense(self.decode_layers[-1],
#                                                    units=self.decode_units[layer],
#                                                    activation=tf.nn.sigmoid)
#                 else:
#                     decode_layer = tf.layers.dense(self.decode_layers[-1],
#                                                    units=self.decode_units[layer],
#                                                    activation=tf.nn.relu)
#                 print('Decode FC', decode_layer.shape)
#                 self.decode_layers.append(decode_layer)
#             decoded = tf.layers.dense(self.decode_layers[-1],
#                                       units=self.features)
#             print('Decoded:', decoded.shape)
#         return decoded
#
#     def _loss(self, decoded, x, mu, log_sigma):
#         loss = tf.cast(tf.reduce_sum(tf.square(decoded - x)), tf.float32)
#         kl = 0.5 * tf.reduce_mean(
#             tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, axis=1)
#         return tf.reduce_mean(loss + kl)
#
#     def _train_step(self, loss):
#         return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
#
#     def get_binary_hash(self, names, model_path=None):
#         if model_path is None:
#             model_path = self.model_path
#
#         tf.reset_default_graph()
#         with tf.Session() as sess:
#             saver = tf.train.import_meta_graph(model_path + '.meta')
#             saver.restore(sess, model_path)
#             input_x = sess.graph.get_tensor_by_name('inputs/input_x:0')
#             z_size = sess.graph.get_tensor_by_name('inputs/z_size:0')
#             encode_mu = sess.graph.get_tensor_by_name('encode/mu/BiasAdd:0')
#
#             # Normalize by training max
#             names_normalized = names / self.max_value
#             # Encode
#             new_mu = sess.run(encode_mu, feed_dict={input_x: names_normalized,
#                                                     z_size: len(names_normalized)})
#             # Compare to median
#             binary = []
#             for row in range(new_mu.shape[0]):
#                 binary.append(np.array([1 if e >= self.median_mu[i] else 0 for i,e in enumerate(new_mu[row])]))
#         return np.vstack(binary)
#
#     def batch_data(self, names, test):
#         # Normalize data to 0-1 range
#         self.max_value = np.max(names)
#         names = names / self.max_value
#         self.features = names.shape[1]
#
#         # Split into train/validation
#         if test is not None:
#             x, val_x = train_test_split(names, test_size=test,
#                                         random_state=self.seed)
#             val_size = len(val_x)
#
#         num_batches = len(x) // self.batch_size
#         batched_X = [x[i:i + self.batch_size] for i in
#                      range(0, len(x), self.batch_size)]
#         batched_names = [names[i:i + self.batch_size] for i in
#                          range(0, len(names), self.batch_size)]
#
#         print('Created {} batches of size {}'.format(num_batches,
#                                                      self.batch_size))
#         return x, val_x, val_size, batched_X, batched_names
#
#
# class LSTMVariationalAutoencoder(VariationalAutoencoder):
#     def __init__(self):
#         super().__init__()
