import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import os
from os import listdir
from library_training import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler



ads = read_pkl_files('dataset/data_2')
ads.drop_duplicates(subset=["Date", "Equipo"], inplace=True)
ads = ads.sort_values(by=['Date']).reset_index()


ads = ads[ads["diff hrs"]>3]
ads = ads[(ads["Cantidad"]>300) & (ads['Cantidad'] < 5300)]
ads = ads[ads['t_cargado'] >= 0]
ads = ads[ads['tiempo_ciclo_mean'] >= 0]
ads = ads[ads["suma toneladas"]>300]
ads = ads[ads["diff hrs"]<48]

# features
variables =  ads.columns.to_list()
X = ads[variables]

# # Reemplazamos carga normal por valores 0 y 1
X['carga_normal'] = X['carga_normal'].apply(str)
X["carga_normal"] = X["carga_normal"].replace(["True", "False"], [1,0])

# Buscamos obtener las estadisticas de la carga anterior
X.sort_values(by=["Equipo", "Date"], inplace=True)

for variable in variables:
    if variable in ["Cantidad", "carga_normal"]:
        new_parametro = variable + "_A"
        X[new_parametro] = X.groupby(by=["Equipo"]).shift()[variable]
# Eliminamos las cargas anomalas
X = X[(X["Cantidad"] / X["diff hrs"]) < 240]


X = X[['Cantidad','t_encendido', 'Cantidad_A', 'suma toneladas', 't_final',
       'distancia_gps',
       'd_gps_sc', 'num_cargado', 'diff hrs', 'num_registros',
       'num_cargado_subiendo', 'diff_cota_sub', 'diff_cota', 'tiempo',
       't_apagado', 'distancia_gps_c', 'd_gps_sv', 'd_gps_bc',
       'd_gps_bv', 'carga_normal', 't_cargado', 'numero_cargas',
       'vel_mean', 'vel025', 'vel075', 'tiempo_ciclo_mean',
       'distancia_pendiente', 'carga_normal_A']]
X = X.reset_index(drop=True)
# borrar esta columna después
# X = X.fillna(X.mean())

X = X.dropna()
df = X.copy()

fig,ax = plt.subplots(1,figsize=(16, 8))
sns.set_context("paper",
                rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})   
x1 = X['Cantidad']
sns.distplot(x1,color='r')
ax.set_title('Distribución Datos', fontname="Arial", fontsize=25)
plt.legend(labels=['Data'], ncol=1, loc='upper left', fontsize=20)
for tick in ax.get_xticklabels():
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    tick.set_fontsize(16)


X.to_pickle('dataset/preprocessed_ads.pkl')

# realizar un análsis estadístico de las columnas para ver el antes y el 
# después del sobremuestreo
analisis_antes = analisis_variables(X)

# Comenzar el proceso de generación de data sintetica
target = 'Cantidad'

# El número de clases debe ser menor al mínimo número de vecinos que tiene cada
# tiene cada punto en las subdivisiones por clase
oversampling_smote_selection(X, target, n_max_clases=35)


num_classes = 25
# Calculo del máximo y mínimo de las cantidades que se quiere crear
max_ = X[target].max()
min_ = X[target].min()

Y_classes = X[target].apply(lambda x: crear_n_clases(x,
                                                     num_classes,
                                                     max_,
                                                     min_))

smote = SMOTE(random_state=27)
random_os = SMOTE(random_state=27)

# Pass the entire data, including the target variable
new_data, _ = random_os.fit_resample(X, Y_classes)
analisis_despues = analisis_variables(new_data)

new_data.to_pickle('dataset/oversampling_ads.pkl')





