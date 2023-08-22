#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from framework.framework_component import FunctionalComponent
class BaseClassificationHead(FunctionalComponent):
    def __init__(self):
        super().__init__()

    def apply_before_transformer(self, X, prefix:str=None):
        return X