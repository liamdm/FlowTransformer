#  FlowTransformer 2023 by liamdm / liam@riftcs.com
class BaseSequential:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def parameters(self) -> dict:
        raise NotImplementedError()

    @property
    def apply(self, X, prefix: str = None):
        raise NotImplementedError()