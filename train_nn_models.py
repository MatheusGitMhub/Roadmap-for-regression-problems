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
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Libreria deep learning tesseracto
from library_training import *

filename = "NN_models"


# Preparamos la data
ads = pd.read_pickle("dataset/preprocessed_ads.pkl")


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


# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=20)

# Guardar resultados del modelo
resultados_modelos = []


# =======================================================================
# Arquitectura 1
# =======================================================================

# Hiperparámetros de la red
batch_size = 1024
epochs = 1500

# Hiperparámetros de los callbacks
patience = 100
min_delta = 500

model_name = f"Arquitectura 1: {filename}"

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
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
batch_size = 1024
epochs = 1500

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 2: {filename}"

model = Sequential()
model.add(Dense(224, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(224, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(224, activation='relu'))
model.add(Dropout(0.3))
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
# Hiperparámetros 
batch_size = 1024
epochs = 1500

patience = 25
min_delta = 500

model_name = f"Arquitectura 3: {filename}"

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
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
batch_size = 1024
epochs = 1500

# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 4: {filename}"


model = Sequential()
model.add(Dense(192, input_dim=X_train.shape[1], activation='linear'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(192, activation='linear'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(192, activation='linear'))
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
batch_size = 1024
epochs = 1500
# Hiperparámetros de los callbacks
patience = 25
min_delta = 500

model_name = f"Arquitectura 5: {filename}"

model = Sequential()
model.add(Dense(160, input_dim=X_train.shape[1], activation='linear'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(160, activation='linear'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(160, activation='linear'))
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

errors[['MAE', 'MAX error', 'STD mae', 'Quantile 5%',
        'Quantile 95%', 'Quantile 10%',
        'Quantile 90%', 'Mediana']] = errors[['MAE', 'MAX error', 'STD mae', 
                                            'Quantile 5%', 'Quantile 95%',
                                            'Quantile 10%', 'Quantile 90%',
                                            'Mediana']].apply(lambda x: round(x,2))


errors.columns = ["Arquitectura", "MAE", "MAX error", "STD mae",
                     "Quantile 5%", "Quantile 95%", "Quantile 10%", 
                     "Quantile 90%", "Mediana"]

errors.to_csv(f"training_results/summary_{filename}.csv")

