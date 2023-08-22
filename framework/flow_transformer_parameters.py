#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from typing import List
class FlowTransformerParameters:
    """
    Allows the configuration of overall parameters of the FlowTransformer
    :param window_size: The number of flows to use in each window
    :param mlp_layer_sizes: The number of nodes in each layer of the outer classification MLP of FlowTransformer
    :param mlp_dropout: The amount of dropout to be applied between the layers of the outer classification MLP
    """
    def __init__(self, window_size:int, mlp_layer_sizes:List[int], mlp_dropout:float=0.1):
        self.window_size:int = window_size
        self.mlp_layer_sizes = mlp_layer_sizes
        self.mlp_dropout = mlp_dropout

        # Is the order of flows important within any individual window
        self._train_ensure_flows_are_ordered_within_windows = True

        # Should windows be sampled sequentially during training
        self._train_draw_sequential_windows = False
