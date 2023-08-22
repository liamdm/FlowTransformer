#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from typing import Tuple

from framework_component import FunctionalComponent
from model_input_specification import ModelInputSpecification


class SequentialInputEncoding(FunctionalComponent):
    """
    Represents an encoding for a particular input,  that consists of
    a series of sequential steps, which are applied in order
    """
    def __init__(self, *steps: FunctionalComponent):
        super().__init__()
        self.steps: Tuple[FunctionalComponent] = steps

    def apply(self, X, prefix: str = None):
        v = X
        for step in self.steps:
            v = step.apply(v)
        return v

    def build(self, sequence_length:int, model_input_specification:ModelInputSpecification):
        for step in self.steps:
            step.build(sequence_length, model_input_specification)

