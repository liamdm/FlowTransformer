#  FlowTransformer 2023 by liamdm / liam@riftcs.com
import numpy as np

from framework.enumerations import CategoricalFormat
from framework.framework_component import Component


class BasePreProcessing(Component):
    def __init__(self):
        pass

    def fit_numerical(self, column_name:str, values:np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def transform_numerical(self, column_name:str, values: np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def fit_categorical(self, column_name:str, values:np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def transform_categorical(self, column_name:str, values:np.array, expected_categorical_format:CategoricalFormat):
        raise NotImplementedError("Please override this base class with a custom implementation")
