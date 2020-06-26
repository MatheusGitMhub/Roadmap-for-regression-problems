import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.decomposition import PCA

from keras.layers import Conv1D, Conv2D, MaxPooling2D
import time

from library_training import *


filename = "PCA_CNN_models"

# Preparamos la data
ads = pd.read_pickle("dataset/preprocessed_ads.pkl")

# features = ['suma toneladas', "t_final", 'distancia_gps', 'd_gps_sc',
#               "num_cargado", "diff hrs", "t_cargado", "numero_cargas",
#               "carga_normal", "Cantidad"]


# features = ['Cantidad', 'suma toneladas', "t_final", 'distancia_gps', 'd_gps_sc',
#               "num_cargado", "diff hrs", "t_cargado", "numero_cargas",
#               "carga_normal"]

# features = ['Cantidad', 'suma toneladas', "t_final", 'distancia_gps', 'd_gps_sc',
#               "num_cargado", "diff hrs", "t_cargado", "numero_cargas",
#               "carga_normal", "vel_mean", "distancia_pendiente"]

features = ads.columns.to_list()
X = ads[features]


y = X[["Cantidad"]]
del X["Cantidad"]

columns_dataset = X.columns.to_list()



# =======================================================================
# Data preparation
# =======================================================================

# Crear objecto scaler
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler() 

# Para cargar el scaler en cualquier otro file
# scaler_filename = "scaler"
# scaler = joblib.load(scaler_filename)

# Normalizar
X = scaler.fit_transform(X)

# # Guardar objeto scaler
scaler_filename = f"scaler_dataset/{filename}_scaler.save"
joblib.dump(scaler, scaler_filename)



# =======================================================================
# PCA - Preprocessing
# =======================================================================

print(X.shape)
pca = PCA(n_components=18)
principalComponents = pca.fit_transform(X)
principalComponents = pd.DataFrame(principalComponents)
print(pca.explained_variance_ratio_.sum())




X = principalComponents
X = X.to_numpy()



print(X.shape)

# transformar el feature vetor to matrix
X = np.reshape(X, (-1, 9, 2, 1))

print(X.shape)
# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=69)


n_timesteps, n_features, n_outputs =\
    X_train.shape[1], X_train.shape[2], y_train.shape[1]

# Guardar resultados del modelo
resultados_modelos = []



# =======================================================================
# Arquitectura 1
# =======================================================================

# Hiperparámetros de la red
batch_size = 60
epochs = 280

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 1: {filename}"

model = Sequential()
model.add(Conv2D(32, input_shape = X_train.shape[1:],
                 kernel_size = (2, 1), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, (2, 1), padding="same", activation="relu"))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.summary()


y_pred1, model1 = neural_net_processing(model, X_train, X_test,
                                      y_train, y_test,
                                      batch_size, epochs,
                                      patience, min_delta,
                                      model_name, filename)
mae1 = mean_absolute_error(y_test, y_pred1)
mae_acum1 = abs(y_pred1-y_test)
std1 = mae_acum1.std()[0]

resultados_modelos.append([model_name, mae1, std1])

# =======================================================================
# Arquitectura 2 
# =======================================================================
# Hiperparámetros de la red
batch_size = 60
epochs = 200

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 2: {filename}"


model = Sequential()
model.add(Conv2D(64, input_shape = X_train.shape[1:],
                 kernel_size = (2, 1), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(128, (2, 2), padding="same", activation="relu"))
model.add(Conv2D(256, (1, 2), padding="same", activation="relu"))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.summary()


y_pred2, model2 = neural_net_processing(model, X_train, X_test,
                                      y_train, y_test,
                                      batch_size, epochs,
                                      patience, min_delta,
                                      model_name, filename)

mae2 = mean_absolute_error(y_test, y_pred2)
mae_acum2 = abs(y_pred2-y_test)
std2 = mae_acum2.std()[0]

resultados_modelos.append([model_name, mae2, std2])



# =======================================================================
# Arquitectura 3
# =======================================================================
# Hiperparámetros de la red
batch_size = 60
epochs = 200

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 3: {filename}"


model = Sequential()
model.add(Conv2D(128, input_shape = X_train.shape[1:],
                 kernel_size = (2, 1), padding="same", activation="relu"))
model.add(Conv2D(256, (2, 2), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 1)))


model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.summary()



y_pred3, model3 = neural_net_processing(model, X_train, X_test,
                                      y_train, y_test,
                                      batch_size, epochs,
                                      patience, min_delta,
                                      model_name, filename)
mae3 = mean_absolute_error(y_test, y_pred3)
mae_acum3 = abs(y_pred3-y_test)
std3 = mae_acum3.std()[0]
resultados_modelos.append([model_name, mae3, std3])


# =======================================================================
# Arquitectura 4
# =======================================================================
# Hiperparámetros de la red
batch_size = 60
epochs = 200

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 4: {filename}"

model = Sequential()
model.add(Conv2D(32, input_shape = X_train.shape[1:],
                 kernel_size = (2, 1), padding="same", activation="relu"))
model.add(Conv2D(64, (1, 2), padding="same", activation="relu"))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.summary()




y_pred4, model4 = neural_net_processing(model, X_train, X_test,
                                      y_train, y_test,
                                      batch_size, epochs,
                                      patience, min_delta,
                                      model_name, filename)

mae4 = mean_absolute_error(y_test, y_pred4)
mae_acum4 = abs(y_pred4-y_test)
std4 = mae_acum4.std()[0]
resultados_modelos.append([model_name, mae4, std4])


# =======================================================================
# Arquitectura 5
# =======================================================================
# Hiperparámetros de la red
batch_size = 60
epochs = 200

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 5: {filename}"

model = Sequential()
model.add(Conv2D(32, input_shape = X_train.shape[1:],
                 kernel_size = (2, 1), padding="same", activation="relu"))
model.add(Conv2D(64, (2, 2), padding="same", activation="relu"))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='relu'))
model.summary()


y_pred5, model5 = neural_net_processing(model, X_train, X_test,
                                      y_train, y_test,
                                      batch_size, epochs,
                                      patience, min_delta,
                                      model_name, filename)
mae5 = mean_absolute_error(y_test, y_pred5)
mae_acum5 = abs(y_pred5-y_test)
std5 = mae_acum5.std()[0]
resultados_modelos.append([model_name, mae5, std5])

# =======================================================================
# Guardar resultados
# =======================================================================

resultados_modelos = pd.DataFrame(resultados_modelos)
resultados_modelos.columns = ['Arquitectura', 'Error absoluto medio (MAE)',
                              'Desviación estandar error']

resultados_modelos.to_csv(f"training_results/{filename}_results.csv")


mae_acum1 = pd.DataFrame(mae_acum1).reset_index(drop=True)
mae_acum2 = pd.DataFrame(mae_acum2).reset_index(drop=True)
mae_acum3 = pd.DataFrame(mae_acum3).reset_index(drop=True)
mae_acum4 = pd.DataFrame(mae_acum4).reset_index(drop=True)
mae_acum5 = pd.DataFrame(mae_acum5).reset_index(drop=True)



mae_results = pd.concat([mae_acum1, mae_acum2, mae_acum3, mae_acum4,
                         mae_acum5], axis=1)

mae_results.columns = ['Arquitectura 1', 'Arquitectura 2',
                       'Arquitectura 3', 'Arquitectura 4',
                       'Arquitectura 5']

mae_results.to_csv(f"training_results/{filename}_mae_test.csv")

columnas = mae_results.columns.to_list()
errors = []
for i in range(5):
    
    columna = mae_results.iloc[:,i]   
    column = columnas[i]
    
    # print(columna.shape)
    
    errors.append([column, columna.mean(), columna.max(),
                      columna.std(), columna.quantile(0.05),
                      columna.quantile(0.95), columna.quantile(0.1),
                      columna.quantile(0.9), columna.median()])

errors = pd.DataFrame(errors)
errors.columns = ["Arquitectura", "MAE", "MAX error", "STD mae",
                     "Quantile 5%", "Quantile 95%", "Quantile 10%", 
                     "Quantile 90%", "Mediana"]

errors.to_csv(f"training_results/summary_{filename}.csv")




# errors = pd.read_csv('training_results/summary_PCA_CNN_models.csv')
# errors.drop(columns='Unnamed: 0', inplace=True)
# errors[['MAE', 'MAX error', 'STD mae', 'Quantile 5%',
#         'Quantile 95%', 'Quantile 10%',
#         'Quantile 90%', 'Mediana']] = errors[['MAE', 'MAX error', 'STD mae', 
#                                             'Quantile 5%', 'Quantile 95%',
#                                             'Quantile 10%', 'Quantile 90%',
#                                             'Mediana']].apply(lambda x: round(x,2))
                                              
# errors.to_csv('training_results/summary_PCA_CNN_models.csv')