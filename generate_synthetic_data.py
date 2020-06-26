import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


from library_training import *

# Preparamos la data
ads = pd.read_pickle("dataset/preprocessed_ads.pkl")
ads = ads[ads['Cantidad']<=2000].reset_index(drop=True)



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential



# Definir el dataset
X = ads
y = X[["Cantidad"]]
del X["Cantidad"]

columns_dataset = X.columns.to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=20)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train = pd.concat([y_train, X_train], axis=1).reset_index(drop=True).to_numpy()
X_test = pd.concat([y_test, X_test], axis=1).reset_index(drop=True).to_numpy()




original_dim = 28
intermediate_dim = 14
latent_dim = 5
batch_size = 100
epochs = 50
epsilon_std = 1.0


def nll(y_true, y_pred):
    """ Log likelihood (Bernoulli) negativo. """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ 
    Agregar divergencia KL a el loss del vae
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')])

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='adam', loss=nll)




vae.fit(X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test))

encoder = Model(x, z_mu)

# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(X_test, batch_size=batch_size)


plt.figure(figsize=(6, 6))
plt.scatter(z_test[:, 0], z_test[:, 1])
plt.colorbar()
plt.show()



