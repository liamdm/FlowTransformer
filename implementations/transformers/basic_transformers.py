#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from framework.base_sequential import BaseSequential
from implementations.transformers.basic.decoder_block import TransformerDecoderBlock
from implementations.transformers.basic.encoder_block import TransformerEncoderBlock

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

class BasicTransformer(BaseSequential):

    @property
    def name(self) -> str:
        if self.use_conv:
            return f"Basic Conv Transformer" + (" Decoder" if self.is_decoder else "")
        else:
            return f"Basic Dense Transformer" + (" Decoder" if self.is_decoder else "")

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "use_conv": self.use_conv,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.internal_size
        }

    def __init__(self, n_layers:int, internal_size:int, n_heads:int, use_conv:bool=False, dropout_rate:float=0.1, is_decoder=False):
        super().__init__()
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.use_conv = use_conv
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.is_decoder = is_decoder

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            if self.is_decoder:
                if self.use_conv:
                    raise NotImplementedError()
                m_x = TransformerDecoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)
            else:
                m_x = TransformerEncoderBlock(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, use_conv=self.use_conv, prefix=f"{prefix}block_{layer_i}_")(m_x)

        return m_x