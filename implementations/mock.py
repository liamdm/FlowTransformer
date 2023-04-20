#  FlowTransformer 2023 by liamdm / liam@riftcs.com

from framework import BaseSequential, BaseClassificationHead

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras.layers import Dense, Flatten

class DummyTransformer(BaseSequential):
    def apply(self, X, prefix:str=None):
        x = X
        #x = Dense(3)(x)
        return x

class DummyClassificationHead(BaseClassificationHead):
    def apply(self, X, prefix:str=None):
        x = X
        x = Flatten()(x)
        x = Dense(1, activation="sigmoid")(x)
        return x