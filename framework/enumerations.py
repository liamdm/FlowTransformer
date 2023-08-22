#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from enum import Enum

class CategoricalFormat(Enum):
    """
    The format of variables expected by the model as input
    """
    Integers = 0,
    """
    If categorical values should be dictionary encoded as integers
    """
    OneHot = 1
    """
    If categorical values should be one-hot encoded
    """

class EvaluationDatasetSampling(Enum):
    """
    How to choose evaluation samples from the raw dataset
    """
    LastRows = 0
    """
    Take the last rows in the dataset to form the evaluation dataset
    """
    RandomRows  = 1
    """
    Randomly sample rows to make up the evaluation dataset
    """
    FilterColumn = 2
    """
    Define a column that contains a flag indicating if this row is part of the evaluation set
    """
