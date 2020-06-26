import warnings
warnings.filterwarnings("ignore")

# Preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

# Machine learning
from sklearn.model_selection import train_test_split

# Deep learning
import keras
import tensorflow as tf
# tf.enable_eager_execution()

import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Libreria deep learning tesseracto
from library_training import *
filename = "Final_model"


# =======================================================================
# Data normalization
# =======================================================================
# Preparamos la data
# ads = pd.read_pickle("dataset/preprocessed_ads.pkl").reset_index(drop=True)
data_oversampling = pd.read_pickle("dataset/oversampling_ads.pkl").reset_index(drop=True)
data_original = pd.read_pickle("dataset/preprocessed_ads.pkl").reset_index(drop=True)

# ads = data_original.merge(data_oversampling, how ='outer', indicator=True).\
#     loc[lambda x : x['_merge']=='right_only']
# ads.drop(columns=['_merge'], inplace=True)
# ads = ads[ads['Cantidad'] > 0]
# ads = ads.reset_index(drop=True)


ads = pd.read_pickle("dataset/oversampling_ads.pkl").reset_index(drop=True)
dataset = ads.copy()

# Eliminar outliers que salieron
ads = ads[ads['Cantidad'] < 4500]

# features
features = ads.columns.to_list()
X = ads[features]

y = X[["Cantidad"]]
del X["Cantidad"]
columns_dataset = X.columns.to_list()


# =======================================================================
# Data normalization
# =======================================================================

# Crear objecto scaler
# scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler() 

# Normalizar
X = scaler.fit_transform(X)

# # Guardar objeto scaler
scaler_filename = f"scaler_dataset/{filename}_scaler.save"
joblib.dump(scaler, scaler_filename)

# Dividir los conjuntos de datos
# Hacer la división por indices para ir a buscar los datos anomalos

X = pd.DataFrame(X, columns=columns_dataset).reset_index(drop=True)
y = y.reset_index(drop=True)

X = X.to_numpy(dtype="float64")
y = y.to_numpy(dtype="float64")

# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=20)

# Si penalization es True, ocupa la función de costos que penaliza errores <0
penalization = True
# Elección de hiperparámetros según la penalización que se utiliza
if penalization==True:
    batch_size = 2048
    epochs = 1200
    patience = 25*8
    min_delta = 500
elif penalization==False: 
    # Ocupo elif por si en el futuro agregamos más funciones de costo
    batch_size = 120
    epochs = 220
    patience = 25
    min_delta = 500
    
model_name = f"Arquitectura custom loss function: {filename}"

model = Sequential()
model.add(Dense(480, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(480, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(480, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))
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
mae = mae_acum.mean()
std = mae_acum.std()
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
print("Porcentaje de errores por arriba:",
      100 - negative_values/y_pred.shape[0]*100)

max_ = max(y_pred.max(), y_test.max())
fig,ax = plt.subplots(1,figsize=(20, 10))
plt.scatter(y_test, y_pred, color = 'blue')
plt.scatter(y_test, y_test, color = 'red')
plt.title(f'Modelo: Red neuronal (Penalización={penalization})', fontsize=30)
plt.xlabel('Cantidades reales de combustible', fontsize=30)
plt.ylabel('Predicción NN de combustible', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(["Predicciones", "Cantidades reales"], fontsize=30,
           loc = "lower right")
plt.ylim(0, 4600) 
plt.xlim(0, 4600)
plt.show()
model.save(f"models_checkpoint/{model_name}.h5")


# Análisis con los datos post testeo
# Eliminar outliers que salieron
data_original = data_original[data_original['Cantidad'] < 4500].\
    reset_index(drop=True)

# features
features = data_original.columns.to_list()
X_original = data_original[features]

y_original = X_original[["Cantidad"]]
del X_original["Cantidad"]

X_original = scaler.transform(X_original)

# Transformar a dataframe
X_original = pd.DataFrame(X_original)
X_original.columns = columns_dataset

# transformar a dataframe el test
X_test = pd.DataFrame(X_test)
X_test.columns = columns_dataset

y_test = pd.DataFrame(y_test)
y_test.columns = ['Cantidad']

# obtengo la data de testo y la data original
data_original = pd.concat([y_original, X_original], axis=1)
data_test = pd.concat([y_test, X_test], axis=1)

# de la data de testeo saco solamente la que esta en el conjunto de datos
# original
alpha = data_test.merge(data_original, how = 'inner' ,indicator=False)

y_original = alpha[["Cantidad"]]
del alpha["Cantidad"]

# saco la matriz de datos original
X_original = alpha.copy()

y_pred_original = model.predict(X_original)
y_pred_original = pd.DataFrame(y_pred_original, columns=['Cantidad'])
mae_original = abs(y_pred_original-y_original)
mae = int(mae_original.mean())
std = int(mae_original.std())
q95 = int(mae_original.quantile(0.95))

max_original = max(y_original.max())
fig,ax = plt.subplots(1,figsize=(22, 12))
plt.scatter(y_test, y_pred, color = 'blue')
plt.scatter(y_original, y_pred_original, color = 'green')
plt.scatter(y_original, y_original, color = 'red')
titulo = f'NN oversampling + originales' +\
    f'| Data original: {alpha.shape[0]} filas' +'\n'+\
    f'MAE: {str(mae)} [lts] --- STD: {str(std)} [lts] --- Q95: {str(q95)} [lts]'
plt.title(titulo, fontsize=30)
plt.xlabel('Cantidades reales de combustible', fontsize=30)
plt.ylabel('Predicción NN de combustible', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(["Predicciones oversampling",
            "Predicciones que estan en la data de testeo y data original",
            "Cantidades reales", ""], fontsize=20,
           loc = "lower right")
plt.ylim(0, 4600)
plt.xlim(0, 4600)
plt.show()

print("=================================")
print("MAE -----> " + str(mae))
print("DEVEST --> " + str(std))
print("QUANTILE 0.95-->" + str(q95))
print("=================================")

