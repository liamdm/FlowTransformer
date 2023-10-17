#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import warnings
from enum import Enum
from typing import List

from framework.base_input_encoding import BaseInputEncoding
from framework.enumerations import CategoricalFormat

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras.layers import Embedding, Dense, Concatenate, Reshape, Lambda
import tensorflow as tf

class NoInputEncoder(BaseInputEncoding):
    def apply(self, X, prefix:str=None):

        numerical_feature_inputs = X[:self.model_input_specification.n_numeric_features]
        categorical_feature_inputs = X[self.model_input_specification.n_numeric_features:]

        if self.model_input_specification.categorical_format == CategoricalFormat.Integers:
            warnings.warn("It doesn't make sense to be using integer based inputs without encoding!")
            categorical_feature_inputs = [Lambda(lambda x: tf.cast(x, tf.float32))(c) for c in categorical_feature_inputs]

        concat = Concatenate()(numerical_feature_inputs + categorical_feature_inputs)

        return concat

    @property
    def name(self):
        return "No Input Encoding"

    @property
    def parameters(self):
        return {}

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.OneHot

class EmbedLayerType(Enum):
    Dense = 0,
    Lookup = 1,
    Projection = 2

class RecordLevelEmbed(BaseInputEncoding):
    def __init__(self, embed_dimension: int, project:bool = False):
        super().__init__()

        self.embed_dimension: int = embed_dimension
        self.project: bool = project

    @property
    def name(self):
        if self.project:
            return "Record Level Projection"
        return "Record Level Embedding"

    @property
    def parameters(self):
        return {
            "dimensions_per_feature": self.embed_dimension
        }

    def apply(self, X:List[keras.Input], prefix: str = None):
        if prefix is None:
            prefix = ""

        assert self.model_input_specification.categorical_format == CategoricalFormat.OneHot

        x = Concatenate(name=f"{prefix}feature_concat", axis=-1)(X)
        x = Dense(self.embed_dimension, activation="linear", use_bias=not self.project, name=f"{prefix}embed")(x)

        return x

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.OneHot

class CategoricalFeatureEmbed(BaseInputEncoding):
    def __init__(self, embed_layer_type: EmbedLayerType, dimensions_per_feature: int):
        super().__init__()

        self.dimensions_per_feature: int = dimensions_per_feature
        self.embed_layer_type: EmbedLayerType = embed_layer_type

    @property
    def name(self):
        if self.embed_layer_type == EmbedLayerType.Dense:
            return f"Categorical Feature Embed - Dense"
        elif self.embed_layer_type == EmbedLayerType.Lookup:
            return f"Categorical Feature Embed - Lookup"
        elif self.embed_layer_type == EmbedLayerType.Projection:
            return f"Categorical Feature Embed - Projection"
        raise RuntimeError()

    @property
    def parameters(self):
        return {
            "dimensions_per_feature": self.dimensions_per_feature
        }

    def apply(self, X:List[keras.Input], prefix:str=None):
        if prefix is None:
            prefix = ""

        if self.model_input_specification is None:
            raise Exception("Please call build() before calling apply!")

        numerical_feature_inputs = X[:self.model_input_specification.n_numeric_features]
        categorical_feature_inputs = X[self.model_input_specification.n_numeric_features:]

        #print(len(numerical_feature_inputs), len(categorical_feature_inputs))
        #print(len(self.model_input_specification.categorical_feature_names), self.model_input_specification.categorical_feature_names)

        collected_numeric = Concatenate(name=f"{prefix}concat_numeric")(numerical_feature_inputs)

        collected_categorical = []
        for categorical_field_i, categorical_field_name in enumerate(self.model_input_specification.categorical_feature_names):
            cat_field_x = categorical_feature_inputs[categorical_field_i]
            if self.embed_layer_type != EmbedLayerType.Lookup:
                assert self.model_input_specification.categorical_format == CategoricalFormat.OneHot

                x = Dense(self.dimensions_per_feature,
                          activation="linear",
                          use_bias=(self.embed_layer_type == EmbedLayerType.Dense),
                          name=f"{prefix}embed_{categorical_field_name.replace('/', '')}")(cat_field_x)
                collected_categorical.append(x)

            elif self.embed_layer_type == EmbedLayerType.Lookup:
                assert self.model_input_specification.categorical_format == CategoricalFormat.Integers

                # reshape the sequence to a flat array
                x = cat_field_x
                x = Embedding(input_dim=self.model_input_specification.levels_per_categorical_feature[categorical_field_i] + 1, output_dim=self.dimensions_per_feature, input_length=self.sequence_length)(x)
                x = Reshape((self.sequence_length, self.dimensions_per_feature), name=f"{prefix}expand_{categorical_field_name}")(x)

                collected_categorical.append(x)
        collected_categorical = Concatenate(name=f"{prefix}concat_categorical")(collected_categorical)

        collected = Concatenate()([collected_numeric, collected_categorical])

        return collected

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.Integers if self.embed_layer_type == EmbedLayerType.Lookup else CategoricalFormat.OneHot


