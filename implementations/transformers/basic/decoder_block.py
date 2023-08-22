try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import tensorflow as tf
from keras.layers import Dense, Layer, MultiHeadAttention, Dropout, LayerNormalization

class TransformerDecoderBlock(Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=input_dimension)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])
        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):
        # inputs = (target_seq, enc_output)
        target_seq = inputs
        enc_output = inputs

        # self attention of target_seq
        attn_output = self.mha(target_seq, target_seq)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = target_seq + attn_output
        out1 = self.layernorm1(out1)

        # multi-head attention with encoder output as the key and value, and target_seq as the query
        attn_output = self.mha(out1, enc_output)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = out1 + attn_output
        out2 = self.layernorm2(out2)

        # feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.layernorm2(out3)

        return out3