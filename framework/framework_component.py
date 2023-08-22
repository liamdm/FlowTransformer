#  FlowTransformer 2023 by liamdm / liam@riftcs.com
import warnings
from typing import Optional, Tuple

from framework.model_input_specification import ModelInputSpecification


class Component():
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def parameters(self) -> dict:
        warnings.warn("Parameters have not been implemented for this class!")
        return {}
class FunctionalComponent(Component):
    def __init__(self):
        self.sequence_length: Optional[int] = None
        self.model_input_specification: Optional[ModelInputSpecification] = None
        self.input_shape: Optional[Tuple[int]] = None

    def apply(self, X, prefix: str = None):
        raise NotImplementedError()

    def build(self, sequence_length:int, model_input_specification:ModelInputSpecification):
        self.sequence_length = sequence_length
        self.model_input_specification = model_input_specification
