import warnings
warnings.filterwarnings("ignore")

# Preprocessing
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np
import io
import boto3
import pytz
import pickle
import time
import os
from os import listdir
import seaborn as sns

# Stats models
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Machine learning
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
# from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler

import joblib
# Deep learning
# import keras
import tensorflow as tf
from tensorflow import keras

# tf.enable_eager_execution()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import keras.backend as K




# Visualization
import matplotlib.pyplot as plt

def crear_dos_clases(cantidad, punto_corte):
    """
    Crea las variables con las que se podría predecir en los sectores

    Parameters
    ----------
    cantidad : Int
        Entero que hay que sobre muestrear.

    Returns
    -------
    clase : TYPE
        La clase a la que pertenece en la distribución.

    """
    
    if cantidad > punto_corte:
        clase = 0
    if cantidad <= punto_corte:
        clase = 1
        
    return clase

def crear_tres_clases(cantidad, punto_corte1, punto_corte2):
    """
    Crea las variables con las que se podría predecir en los sectores

    Parameters
    ----------
    cantidad : Int
        Entero que hay que sobre muestrear.

    Returns
    -------
    clase : TYPE
        La clase a la que pertenece en la distribución.

    """
    
    if cantidad <= punto_corte1:
        clase = 0
    elif (cantidad > punto_corte1) & (cantidad <= punto_corte2):
        clase = 1
    elif cantidad > punto_corte2:
        clase = 2
    return clase


def crear_n_clases(cantidad, n, max_, min_):
    """
    Función bacán para crear las clases considerando tratando de formar 
    una distribución lo más parecida a la uniforme    

    Parameters
    ----------
    cantidad : Int
        Entero que hay que sobre muestrear.
    n : int
        Número de subdivisiones en la data que se quiere crear
    max_ : TYPE
        máx del array
    min_ : TYPE
        min del array

    Returns
    -------
    None.

    """    
    delta = max_ - min_
    delta_n = int(delta / n) + 1     
    num = cantidad - min_     
    clase = int(num / delta_n)

    return clase



def read_pkl_s3(bucket, ruta):
    """
    La funcion lee un archivo pkl desde s3

    Parameters
    ----------
    bucket : Nombre del bucket
    ruta : Ruta del archivo

    Returns
    -------
    data : Dataframe con los datos

    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket,
                        Key=ruta)
    body = obj["Body"].read()
    data = pickle.loads(body)
    # data.reset_index(inplace=True, drop=True)
    return data


def get_data_operator(bucket, fecha_actual, dias, dataframe):
    """
    La funcion busca los archivos .pickle que cumplen un rango de
    fechas

    Parameters
    ----------
    s3: Conexion a servicio s3
    fecha_actual : Ultima fecha desde la que se analiza
    dias : Dias hacia atras que se quiere buscar
    dataframe: nombre del dataframe
    Returns
    -------
    data_final : dataframe con los datos

    """

    # Data final
    data_final = pd.DataFrame()

    # Lista de fechas en el rango
    rutas = []

    # Calculamos los dias anteriores
    for i in range(dias):
        # print(i)
        d = (fecha_actual-timedelta(i)).day
        m = (fecha_actual-timedelta(i)).month
        y = (fecha_actual-timedelta(i)).year
        ruta = "indicadores/" + str(y) + "/" + str(m) + "/" + dataframe + f"_{d}-{m}-{y}.pkl"
        # print(prefijo)
        rutas.append(ruta)

    # Para cada dia
    for ruta in tqdm(rutas):

        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket,
                                Key=ruta)
            body = obj["Body"].read()
            data = pickle.loads(body)

            data_final = pd.concat([data_final, data], axis=0)
        
        except:
            
            pass

    return data_final

def load_ads_s3(fecha_inicio, fecha_final):


    dias = fecha_final - fecha_inicio
    dias = int(dias.total_seconds() / (24*60*60)) + 1
    
    # Dataframes a analizar
    estadisticas = get_data_operator("cosmos-amsa-salida", fecha_final, dias,
                                     "ads_modelo")
    estadisticas = estadisticas.drop_duplicates().reset_index(drop=True)

    return estadisticas

def analisis_variables(X):
    """
    Realiza el análisis estadístico de las columnas con las que se están
    trabajando antes y después del sobremuestreo    

    Parameters
    ----------
    X : Dataframe
        ADS.

    Returns
    -------
        Análsis estadístico de la muestra

    """
    
    analisis = pd.DataFrame()
    for columna in X.columns.to_list():
        
        mean = X[columna].mean()
        std = X[columna].std()
        min_ = X[columna].min()
        max_ = X[columna].max()
        
        list_ = [columna, mean, std, min_, max_]
        df_ = pd.DataFrame(list_).transpose()
        
        analisis = pd.concat([analisis, df_], axis=0)
        
    analisis.columns = ['columna', 'media', 'desviacion', 'min', 'max']
    analisis = analisis.reset_index(drop=True)
    return analisis
        



def selection_by_correlation(dataset, threshold):
    """
    Selecciona solo una de las columnas altamente correlacionadas

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    dataset : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
    
        for j in range(i):
        
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
            
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    dataset = dataset.reset_index(drop=True)
    
    return dataset, dataset.columns.to_list()




def xgboost_processing(X_train, X_test, y_train,
                       y_test, early_stopping_rounds,
                       columns_dataset):
    """
    

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    early_stopping_rounds : TYPE
        DESCRIPTION.
    columns_dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    y_pred : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    regressor : TYPE
        DESCRIPTION.

    """
    
    columns_X = columns_dataset
    
    # Reordenar en la forma (algo,)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # Conjunto de testeo
    eval_set = [(X_test, y_test)]
    
    # XGBRegressor MODEL
    regressor = xgboost.XGBRegressor()
    
    # Entrenar el modelo
    regressor.fit(X_train, y_train, eval_set = [(X_test, y_test)],
                  early_stopping_rounds = 40)
    
    # Generar predicciones
    y_pred = regressor.predict(X_test)
    
    feature_importance =\
        regressor.get_booster().get_score(importance_type="weight")
    
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())

    data = pd.DataFrame(data=values,
                        index=keys,
                        columns=["score"]).sort_values(by = "score",
                                                       ascending=False)

    list_feature_importance = []  

    # Get the feature importance on a dataframe by column name
    for col,score in zip(columns_X,regressor.feature_importances_):
        list_feature_importance.append([col, score])
        
    data =pd.DataFrame(list_feature_importance,
                     columns=["feature", "importance_score"])
    data = data.sort_values(by="importance_score",
                            ascending=False).reset_index(drop=True)
    
    pickle.dump(regressor, open("xgboost.pickle.dat", "wb"))
    
    # # Plot importance
    # plt.figure(figsize=(40,20))
    # plot_importance(regressor)
    # pyplot.show()
    # plt.show()

    return  y_pred, data, regressor




def neural_net_processing(model, X_train, X_test, y_train, y_test,
                          batch_size, epochs, patience, min_delta,
                          model_name, filename):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.
    patience : TYPE
        DESCRIPTION.
    min_delta : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    penalization = False
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
                                         min_lr=5E-3)

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
    time.sleep(5)    
    diff = y_pred - y_test
    diff = np.reshape(diff, -1)
    negative_values = np.count_nonzero(diff<0)
    print("Porcentaje de errores por debajo:",
          negative_values/y_pred.shape[0]*100)
    # Save the model as .h5
    model.save(f"models_checkpoint/{filename}_{model_name}.h5")

    return y_pred, model



def training_history(history, epocas_hacia_atras, model_name, filename):
    
    
    # x_labels
    # Hist training 
    largo = len(history.history['loss'])
    x_labels = np.arange(largo-epocas_hacia_atras, largo)
    x_labels = list(x_labels)
    loss_training =  history.history['loss'][-epocas_hacia_atras:]
    loss_validation =  history.history['val_loss'][-epocas_hacia_atras:]

    fig,ax = plt.subplots(1,figsize=(16, 8))
    ax.plot(x_labels, loss_training,'b', linewidth=2)
    ax.plot(x_labels, loss_validation,'r', linewidth=2)
    ax.set_xlabel('Epochs', fontname="Arial", fontsize=14)
    ax.set_ylabel('Cosmos loss function', fontname="Arial", fontsize=14)
    ax.set_title(f"{model_name}", fontname="Arial", fontsize=20)
    ax.legend(['Training', 'Validation'], loc='upper left',prop={'size': 14})

    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    
    fig.savefig(f"training_results/{model_name}.png")





def handler_loss_function(batch_size, penalization):
    
    if penalization==True:
        
        # Retorna la función de costos de cosmos con penalización
        def cosmos_loss_function(y_true, y_pred):
            # Covertir a tensor de tensorflow con keras como backend    
            y_true=K.cast(y_true, dtype='float32')
            y_pred=K.cast(y_pred, dtype='float32')
            # Reshape como vector
            y_true = K.reshape(y_true, (-1, 1))    
            y_pred = K.reshape(y_pred, (-1, 1))
            # Vector de error mae
            diff_error = y_pred - y_true
            # Cuenta el número de veces que se equivo por abajo
            negative_values = K.cast(tf.math.count_nonzero(diff_error > 0),
                                     dtype='float32')
            
            size_train = K.shape(y_pred)[0]
            size_train = K.reshape(size_train, (-1, 1))    
            loss = K.square(y_pred - y_true)  
            loss = K.sum(loss, axis=1)       
            loss = K.mean(loss)
            loss = loss * (0.1 + negative_values / batch_size)
        
            return loss

    elif penalization==False:
        
        # Retorna el error cuadratico medio        
        def cosmos_loss_function(y_true, y_pred):
            
            y_true=K.cast(y_true, dtype='float32')
            y_pred=K.cast(y_pred, dtype='float32')
            y_true = K.reshape(y_true, (-1, 1))    
            y_pred = K.reshape(y_pred, (-1, 1))
            size_train = K.shape(y_pred)[0]
            size_train = K.reshape(size_train, (-1, 1))    
            loss = K.square(y_pred - y_true)  
            loss = K.sum(loss, axis=1)       
            loss = K.mean(loss)
        
            return loss        
            
    return cosmos_loss_function

def get_model_summary(model):
    """
    Retorna el sumary de los modelos

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    summary_string : TYPE
        DESCRIPTION.

    """

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def read_pkl_files(path):    
    files = os.listdir(path)
    data = pd.DataFrame()
    for file in files:
        root = path +"/" + file
        data_i = pd.read_pickle(root)
        data = pd.concat([data, data_i], axis=0)
    return data

def oversampling_smote_selection(X, target, n_max_clases):
    """
    Función que ayuda a determinar la evolución de la distribución de los datos
    en función del oversampling    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.
    n_max_clases : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """ 
    
    for i in range(2, n_max_clases +1):
        
        num_classes = int(i)
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
        
        porcentaje = new_data.shape[0] / X.shape[0] 
        porcentaje = round(porcentaje, 2)

        filas_nuevas = new_data.shape[0] 
                
        
        fig,ax = plt.subplots(1,figsize=(16, 8))
        sns.set_context("paper",
                        rc={"font.size":20, "axes.titlesize":20,
                            "axes.labelsize":20})  
        x0 = new_data['Cantidad'] 
        x1 = X['Cantidad']
        sns.distplot(x1,color='r')
        sns.distplot(x0,color='b')
        ax.set_title(f'Sobre muestreo: clases = {num_classes}, ratio = {porcentaje}, filas = {filas_nuevas}' ,
                     fontname="Arial", fontsize=25)
        plt.legend(labels=['Distribución original',
                           'Distribución sobre muestreada'],
                   ncol=1, loc='upper left',
                   fontsize=20)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(16)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(16)
            
        fig.savefig(f'synthetic_data/{num_classes}.png', dpi=fig.dpi)
