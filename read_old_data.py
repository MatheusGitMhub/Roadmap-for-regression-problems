import pandas as pd

dataset = pd.read_pickle('dataset/FINAL.pkl')
columns_data = dataset.columns.to_list()





data_s_camniones = dataset[['Cantidad', 'Con_Max', 'Consumo_Encendido',
       'Date', 'Equipo', 'ID', 'Operador', 'Station Name', 'anomalia', 'crew',
       'd_gps_bc', 'd_gps_bv', 'd_gps_sc', 'd_gps_sv', 'des_v1', 'des_v2',
       'diff hrs', 'diff_cota', 'diff_cota_sub', 'distancia_gps',
       'distancia_gps_c', 'distancia_total1', 'distancia_total2', 'lt_an',
       'lt_extra', 'lt_sc', 'mean toneladas', 'num_cargado',
       'num_cargado_subiendo', 'num_registros', 'suma toneladas', 't_apagado',
       't_final', 'temperatura', 'tiempo', 'tiempo_c', 'v1_max', 'v1_mean',
       'v2_max', 'v2_mean']]

data_s_camniones = data_s_camniones[data_s_camniones["diff hrs"]>3]
data_s_camniones = data_s_camniones[data_s_camniones["Cantidad"]>1000]
data_s_camniones = data_s_camniones[data_s_camniones["suma toneladas"]>300]
data_s_camniones = data_s_camniones[data_s_camniones["diff hrs"]<48]



head = data_s_camniones.head(10000)


data_s_camniones = data_s_camniones.sort_values(by="Date")

ads = data_s_camniones[['suma toneladas', "t_final", 'distancia_gps',
                        'd_gps_sc', "num_cargado", "diff hrs", "Cantidad"]]

