import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


from library_training import *

# Preparamos la data
ads = pd.read_pickle("dataset/preprocessed_ads.pkl")
X = ads


y = X[["Cantidad"]]
del X["Cantidad"]


columns_dataset = X.columns.to_list()

# =======================================================================
# Data preparation
# =======================================================================
# Para cargar el scaler en cualquier otro file
# scaler_filename = "scaler"
# scaler = joblib.load(scaler_filename)


# Crear objecto scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalizar
X = scaler.fit_transform(X)

# # Guardar objeto scaler
scaler_filename = "scaler_dataset/scaler_xgboost.save"
joblib.dump(scaler, scaler_filename)



# Dividir los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=20)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


# Suport vector machines regressor
regressor = SVR(kernel='rbf')
# regressor = SVR(kernel='linear')

# Entrenar el modelo
regressor.fit(X_train,y_train)

# Generar predicciones
y_pred = regressor.predict(X_test)


diff = y_pred - y_test

mae = abs(diff).mean()

print(mae)
