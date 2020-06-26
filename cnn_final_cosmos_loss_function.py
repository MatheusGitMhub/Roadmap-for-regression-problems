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
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

from keras.layers import Conv1D, Conv2D, MaxPooling2D
import time

from library_training import *


filename = "CNN_final_model"

# Preparamos la data
ads = pd.read_pickle("dataset/oversampling_ads.pkl")

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

# Normalizar
X = scaler.fit_transform(X)

# # Guardar objeto scaler
scaler_filename = f"scaler_dataset/{filename}_scaler.save"
joblib.dump(scaler, scaler_filename)

print(X.shape)
# transformar el feature vetor to matrix
X = np.reshape(X, (-1, 9, 3, 1))
print(X.shape)
# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=20)






# Si penalization es True, ocupa la función de costos que penaliza errores <0
penalization = False
# Elección de hiperparámetros según la penalización que se utiliza
if penalization==True:
# Hiperparámetros de la red
    batch_size = 1024
    epochs = 1200
    # Hiperparámetros de los callbacks
    patience = 25
    min_delta = 500
    
elif penalization==False: 
    # Hiperparámetros de la red
    batch_size = 1024
    epochs = 1200
    # Hiperparámetros de los callbacks
    patience = 25
    min_delta = 500

    
    
model_name = f"CNN: {filename}"

model = Sequential()
model.add(Conv2D(32, input_shape = X_train.shape[1:],
                 kernel_size = (2, 1), padding="same", activation="relu"))
model.add(Conv2D(64, (2, 1), padding="same", activation="relu"))
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



# Agregar las función de costo a keras
keras.losses.handler_loss_function = handler_loss_function

# Compile optimizer
model.compile(loss=handler_loss_function(batch_size, penalization),
              optimizer='nadam')

keras.callbacks.Callback()

stop_condition = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode ='min',
                                          patience=patience,
                                          verbose=1,
                                          min_delta=min_delta,
                                          restore_best_weights=True)

learning_rate_schedule = ReduceLROnPlateau(monitor="val_loss",
                                     factor=0.5,
                                     patience=25,
                                     verbose=1,
                                     mode="auto",
                                     cooldown=0,
                                     min_lr=5E-4)

callbacks = [stop_condition, learning_rate_schedule]

history = model.fit(X_train, y_train,validation_split=0.2,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=False,
                    verbose=1,
                    callbacks=callbacks)

size_training = len(history.history['val_loss'])
fig = training_history(history, size_training, model_name,
                       filename + "_ultimas:"+ str(size_training)+"epocas")
fig =training_history(history, int(1.5 * size_training / 2), model_name,
                      filename + "_ultimas:"+\
                          str(1.5 * size_training / 2) + "epocas")
fig =training_history(history, int(size_training / 2), model_name,
                      filename + "_ultimas:"+ str(size_training / 2) +\
                          "epocas")
fig =training_history(history, int(size_training / 3), model_name,
                      filename + "_ultimas:"+ str(size_training / 3) +\
                          "epocas")
fig =training_history(history, int(size_training / 4), model_name,
                      filename + "_ultimas:"+ str(size_training / 4) +\
                          "epocas")
    
    

# Score del modelo entrenado
scores = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Mean squared error, Test:', scores)
# Predictions on y_test
y_pred = model.predict(X_test)
# Metricas de evaluación
mae_acum = abs(y_pred-y_test)
mae = int(mae_acum.mean())
std = int(mae_acum.std())
q95 = int(mae_acum.quantile(0.95))

print("=================================")
print("MAE -----> " + str(mae))
print("DEVEST --> " + str(std))
print("=================================")
# Save the model as .h5
# model.save(f"models_checkpoint/{filename}_{model_name}.h5")
diff = y_pred - y_test
diff = np.reshape(diff, -1)
negative_values = np.count_nonzero(diff<0)
print("Porcentaje de errores por debajo:",
      negative_values/y_pred.shape[0]*100)
    
    

fig,ax = plt.subplots(1,figsize=(22, 12))
plt.scatter(y_test, y_pred, color = 'blue')
plt.scatter(y_test, y_pred, color = 'blue')
plt.scatter(y_test, y_test, color = 'red')
titulo = f'CNN oversampling + originales' +\
    f'| Data original: {y_test.shape[0]} filas' +'\n'+\
    f'MAE: {str(mae)} [lts] --- STD: {str(std)} [lts] --- Q95: {str(q95)} [lts]'
plt.title(titulo, fontsize=30)
plt.xlabel('Cantidades reales de combustible', fontsize=30)
plt.ylabel('Predicción CNN de combustible', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(["Predicciones", "Cantidades reales"], fontsize=30,
           loc = "lower right")
plt.ylim(0, 4600)
plt.xlim(0, 4600)
plt.show()
    
model.save(f"models_checkpoint/{filename}_{model_name}.h5")

# data_original = pd.read_pickle("dataset/preprocessed_ads.pkl").reset_index(drop=True)

# # features
# features = data_original.columns.to_list()
# X_original = data_original[features]

# y_original = X_original[["Cantidad"]]
# del X_original["Cantidad"]


# X_original = scaler.transform(X_original)


# # Transformar a dataframe
# X_original = pd.DataFrame(X_original)
# X_original.columns = columns_dataset

# # transformar a dataframe el test

# X_test = np.reshape(X_test, (-1, 27))
# print(X_test.shape)
# X_test = pd.DataFrame(X_test)
# X_test.columns = columns_dataset

# y_test = pd.DataFrame(y_test)
# y_test.columns = ['Cantidad']




# X_original = X_original.reset_index(drop=True)
# y_original = y_original.reset_index(drop=True)
# # obtengo la data de testo y la data original
# data_original = pd.concat([y_original, X_original], axis=1)
# data_test = pd.concat([y_test, X_test], axis=1)

# # de la data de testeo saco solamente la que esta en el conjunto de datos
# # original
# alpha = data_test.merge(data_original, how = 'inner' ,indicator=False)

# y_original = alpha[["Cantidad"]]
# del alpha["Cantidad"]


# # X_original = X_original.to_numpy()
# # print(X_original.shape)
# # X_original = np.reshape(X_original, (-1, 9, 3, 1))
# # print(X_original.shape)

# # saco la matriz de datos original
# X_original = alpha.copy()

# y_pred_original = model.predict(X_original)
# y_pred_original = pd.DataFrame(y_pred_original, columns=['Cantidad'])
# mae_original = abs(y_pred_original-y_original)
# mae = int(mae_original.mean())
# std = int(mae_original.std())
# q95 = int(mae_original.quantile(0.95))

# max_original = max(y_original.max())
# fig,ax = plt.subplots(1,figsize=(22, 12))
# plt.scatter(y_test, y_pred, color = 'blue')
# plt.scatter(y_original, y_pred_original, color = 'green')
# plt.scatter(y_original, y_original, color = 'red')
# titulo = f'NN oversampling + originales' +\
#     f'| Data original: {alpha.shape[0]} filas' +'\n'+\
#     f'MAE: {str(mae)} [lts] --- STD: {str(std)} [lts] --- Q95: {str(q95)} [lts]'
# plt.title(titulo, fontsize=30)
# plt.xlabel('Cantidades reales de combustible', fontsize=30)
# plt.ylabel('Predicción NN de combustible', fontsize=30)
# ax.tick_params(axis='both', which='major', labelsize=20)
# plt.legend(["Predicciones oversampling",
#             "Predicciones que estan en la data de testeo y data original",
#             "Cantidades reales", ""], fontsize=20,
#            loc = "lower right")
# plt.ylim(0, 4600)
# plt.xlim(0, 4600)
# plt.show()

# print("=================================")
# print("MAE -----> " + str(mae))
# print("DEVEST --> " + str(std))
# print("QUANTILE 0.95-->" + str(q95))
# print("=================================")
