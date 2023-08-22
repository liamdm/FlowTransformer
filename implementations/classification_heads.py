#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import numpy as np
import tensorflow as tf

from framework.base_classification_head import BaseClassificationHead

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras.layers import Dense, Concatenate, Flatten, Lambda, GlobalAveragePooling1D

class FlattenClassificationHead(BaseClassificationHead):
    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""
        x = Flatten(name=f"{prefix}flatten")(X)
        return x

    @property
    def name(self) -> str:
        return "Flatten"

    @property
    def parameters(self) -> dict:
        return {}


class FeaturewiseEmbedding(BaseClassificationHead):
    def __init__(self, project:bool=False):
        super().__init__()
        self.project: bool = project

    @property
    def name(self):
        if self.project:
            return f"Featurewise Embed - Projection"
        else:
            return f"Featurewise Embed - Dense"

    @property
    def parameters(self):
        return {}


    def apply(self, X, prefix:str=None):
        if prefix is None:
            prefix = ""

        if self.model_input_specification is None:
            raise Exception("Please call build() before calling apply!")

        x = Dense(1,
                  activation="linear",
                  use_bias=(not self.project),
                  name=f"{prefix}featurewise_embed")(X)

        x = Flatten()(x)

        return x

class GlobalAveragePoolingClassificationHead(BaseClassificationHead):
    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""
        return GlobalAveragePooling1D(name=f"{prefix}global_avg_pooling_1d")(X)

    @property
    def name(self) -> str:
        return "Global Average Pooling"

    @property
    def parameters(self) -> dict:
        return {}


class LastTokenClassificationHead(BaseClassificationHead):
    def __init__(self):
        super().__init__()

    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""

        x = Lambda(lambda x: x[..., -1, :], name=f"{prefix}slice_last")(X)
        #x = Flatten(name=f"{prefix}flatten_last")(x)

        return x

    @property
    def name(self) -> str:
        return "Last Token"

    @property
    def parameters(self) -> dict:
        return {}


class CLSTokenClassificationHead(LastTokenClassificationHead):


    @property
    def name(self) -> str:
        return "CLS Token"

    @property
    def parameters(self) -> dict:
        return {}

    def apply_before_transformer(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""

        window_size = self.sequence_length

        x = X
        batch_size = tf.shape(x)[0]
        flow_size = tf.shape(x)[2]

        cls_token_horizontal_single = np.zeros((window_size + 1,))
        cls_token_horizontal_single[-1] = 1.
        cls_token_horizontal_single = tf.convert_to_tensor(cls_token_horizontal_single, dtype=tf.float32)

        cls_token_horizontal = tf.ones((batch_size, window_size + 1,), dtype=tf.float32)
        cls_token_horizontal = tf.multiply(cls_token_horizontal, cls_token_horizontal_single)
        cls_token_horizontal = tf.expand_dims(cls_token_horizontal, axis=-1)

        cls_token_vertical = tf.zeros((batch_size, 1, flow_size,), dtype=tf.float32)

        x = Concatenate(axis=-2, name=f'{prefix}cls_vertical')([x, cls_token_vertical])
        x = Concatenate(axis=-1, name=f'{prefix}cls_horizontal')([x, cls_token_horizontal])

        return x
