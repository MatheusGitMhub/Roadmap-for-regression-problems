
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import boto3
import pandas as pd
import numpy as np
import pytz
from library_training import *
import pickle


# ========================================================================
# Cargar los datos desde S3
# ========================================================================
# Conexion S3
s3 = boto3.client('s3')



# Fecha de incio y termino del procesamiento
fecha_inicio1 =\
    datetime.strptime("2019-11-01", "%Y-%m-%d")
fecha_final1 =\
    datetime.strptime("2019-11-30", "%Y-%m-%d")

stats1 = load_ads_s3(fecha_inicio1, fecha_final1)


fecha_inicio2 =\
    datetime.strptime("2019-12-01", "%Y-%m-%d")
fecha_final2 =\
    datetime.strptime("2019-12-31", "%Y-%m-%d")

stats2 = load_ads_s3(fecha_inicio2, fecha_final2)


fecha_inicio3 =\
    datetime.strptime("2020-01-01", "%Y-%m-%d")
fecha_final3 =\
    datetime.strptime("2020-01-31", "%Y-%m-%d")

stats3 = load_ads_s3(fecha_inicio3, fecha_final3)


fecha_inicio4 =\
    datetime.strptime("2020-02-01", "%Y-%m-%d")
fecha_final4 =\
    datetime.strptime("2020-02-28", "%Y-%m-%d")

stats4 = load_ads_s3(fecha_inicio4, fecha_final4)


fecha_inicio5=\
    datetime.strptime("2020-03-01", "%Y-%m-%d")
fecha_final5 =\
    datetime.strptime("2020-03-31", "%Y-%m-%d")

stats5 = load_ads_s3(fecha_inicio5, fecha_final5)


fecha_inicio6 =\
    datetime.strptime("2020-04-01", "%Y-%m-%d")
fecha_final6 =\
    datetime.strptime("2020-04-30", "%Y-%m-%d")

stats6 = load_ads_s3(fecha_inicio6, fecha_final6)


fecha_inicio7 =\
    datetime.strptime("2020-05-01", "%Y-%m-%d")
fecha_final7 =\
    datetime.strptime("2020-05-31", "%Y-%m-%d")

stats7 = load_ads_s3(fecha_inicio7, fecha_final7)




ads_training = pd.concat([stats1, stats2, stats3, stats4, 
                          stats5, stats6, stats7],
                         axis=0).reset_index(drop=True)

print(ads_training.shape)

ads_training = ads_training.drop_duplicates().reset_index(drop=True)

print(ads_training.shape)



ads_training.to_pickle('ads_training.pkl')


