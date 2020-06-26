from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import *



class CosmosHyperModel(HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):

        model = Sequential()
        model.add(Dense(
            units=hp.Int('units', 64, 512, 16, default=128),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'linear'],
                    default='relu'),
                input_shape=(27,)))  # ver como dejar parametrizado esto
        
        model.add(BatchNormalization())

        model.add(
            Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.1,
                    max_value=0.5,
                    default=0.3,
                    step=0.1)))

        model.add(
            Dense(
                units=hp.Int('units', 32, 256, 16, default=64),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'linear'],
                    default='relu')))

        model.add(BatchNormalization())

        model.add(
            Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.3,
                    max_value=0.3,
                    default=0.3,
                    step=0.1)))

        model.add(
            Dense(
                units=hp.Int('units', 16, 128, 16, default=16),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'linear'],
                    default='relu')))
        
        model.add(BatchNormalization())

        model.add(
            Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.3,
                    max_value=0.3,
                    default=0.3,
                    step=0.1)))
        
        model.add(
            Dense(1,
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'linear'],
                    default='relu')))
        
        model.compile(
            optimizer='nadam', loss='mse', metrics=['mse'])
        
        return model
