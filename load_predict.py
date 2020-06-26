import boto3
import tensorflow as tf
import tempfile
import numpy as np
import decimal
import os
import json


def predict_estanque(event, context):
    """
    La funcion recibe los parametros actuales y predice el combustible
    gastado por los camiones
    """

    print(event)
    s3 = boto3.resource('s3')

    # Datos del camion
    camion = event[1][0]
    cantidad = event[1][1]
    carga_normal = event[1][2]
    real_time_param = event[1][3]
    parametros = np.array([event[0]])

    trainbucket = os.environ["buckettrain"]

    # Traemos el modelo de s3
    with tempfile.TemporaryFile() as f:

        # Cargamos el modelo
        s3.meta.client.download_fileobj(trainbucket,
                                        f"Modelos/all.h5", f)
        
        model = tf.keras.models.load_model(f, compile=False)
        # Prediccion
        if p >= 0:
            print(model.predict(parametros).flatten())
            p = model.predict(parametros).flatten()[0]
        else:
            p = 0      

    if real_time_param == "True":
        # Nuevos datos
        capacidad_inicial = set_capacidad(cantidad, carga_normal)
        estanque_actual = capacidad_inicial - p
        estanque_actual = max(estanque_actual, 0)

        # ADS actualmente cargado
        dynamodb_r = boto3.resource("dynamodb")
        adsdynamodb = os.environ["dynamodbads"]
        tabla = dynamodb_r.Table(adsdynamodb)

        # Datos a actualizar en el ADS
        prediccion = round_float_to_decimal(float(p))
        estanque_actual = round_float_to_decimal(estanque_actual)
        capacidad_inicial = round_float_to_decimal(capacidad_inicial)

        response = tabla.\
            update_item(Key={"Equipo": camion},
                        UpdateExpression="set prediccion = :r ",
                        ExpressionAttributeValues={':r': prediccion},
                        ReturnValues="UPDATED_NEW")

        response = tabla.\
            update_item(Key={"Equipo": camion},
                        UpdateExpression="set estanque_actual = :r ",
                        ExpressionAttributeValues={':r': estanque_actual},
                        ReturnValues="UPDATED_NEW")

        response = tabla.\
            update_item(Key={"Equipo": camion},
                        UpdateExpression="set capacidad = :r ",
                        ExpressionAttributeValues={':r': capacidad_inicial},
                        ReturnValues="UPDATED_NEW")

        print(f"El equipo {camion} ha consumido {p} litros comenzo el"
              f" ciclo con {capacidad_inicial} litros, por ende solo le "
              f"quedan {estanque_actual}")

    return json.dumps(str(p))


def set_capacidad(cantidad, carga_normal, min_estanque=300, max_estanque=4500):
    """
    """
    if ((carga_normal == "True") | (carga_normal == True)):
        return float(max_estanque)
    else:
        return float(min_estanque) + float(cantidad)


def round_float_to_decimal(float_value):
    """
    Convert a floating point value to a decimal that DynamoDB can store,
    and allow rounding.
    """

    # Perform the conversion using a copy of the decimal context that boto3
    # uses. Doing so causes this routine to preserve as much precision as
    # boto3 will allow.
    with decimal.localcontext(boto3.dynamodb.types.DYNAMODB_CONTEXT) as \
        decimalcontext:

        # Allow rounding.
        decimalcontext.traps[decimal.Inexact] = 0
        decimalcontext.traps[decimal.Rounded] = 0
        decimal_value = decimalcontext.create_decimal_from_float(float_value)
        return decimal_value
