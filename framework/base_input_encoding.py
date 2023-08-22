#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from typing import List

from framework.enumerations import CategoricalFormat
from framework.framework_component import FunctionalComponent


class BaseInputEncoding(FunctionalComponent):
    def apply(self, X:List["keras.Input"], prefix: str = None):
        raise NotImplementedError("Please override this with a custom implementation")

    @property
    def required_input_format(self) -> CategoricalFormat:
        raise NotImplementedError("Please override this with a custom implementation")
