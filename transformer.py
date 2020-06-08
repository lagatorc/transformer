import tensorflow as tf
import numpy as np

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len) 

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def normalized_dot_product_attention(Q, K, V, mask):
    """
    Calculates the dot product attention weights
    and returns a matrix Z, which is the same size as 
    Q, K, and V.
    
    Parameters:
    
    Q - The result of applying Wq to X. 
      - Shape (..., sequence_len, dim_Wq)
    K - The result of applying Wk to X. 
      - Shape (..., sequence_len, dim_Wk)
    V - The result of applying Wv to X. 
      - Shape (..., sequence_len, dim_Wv)
    
    Returns:
    
    Z - The matrix created by applying the scaled attention 
            weights to the V matrix.
      - Shape (..., sequence_len, model_dim)
      
    """
    #Normalizing Q
    Q = tf.divide(Q, tf.norm(Q, axis=-1, keepdims=True))
    
    #Normalizing K
    K = tf.divide(K, tf.norm(K, axis=-1, keepdims=True))
    
    #compute the dot product of all query and key vectors (b/c they are normalized 0 >= values <=1
    attention_logits = tf.matmul(Q, K, transpose_b=True)
    
    attention_logits *= 1e3
    
    if mask is not None:
        attention_logits += (mask * -1e9)
        
    #apply softmax to find the weights
    attention_weights = tf.nn.softmax(attention_logits, axis=-1)
    
    #multiply the weights by the Value matrix
    Z = tf.matmul(attention_weights, V)
    
    return Z, attention_weights

class MHAttention(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, embedding_dim):
        super(MHAttention, self).__init__()
        assert embedding_dim % num_heads == 0

        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.Wq = tf.keras.layers.Dense(self.embedding_dim)
        self.Wk = tf.keras.layers.Dense(self.embedding_dim)
        self.Wv = tf.keras.layers.Dense(self.embedding_dim)

        self.Wz = tf.keras.layers.Dense(self.embedding_dim)
        
    def create_heads(self, x, batch_size):
        
        return tf.reshape(tf.transpose(x), (batch_size, self.num_heads, -1, self.head_dim))
        
    def call(self, q, k, v, mask):
         
        batch_size = q.shape[0]
        
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        q = self.create_heads(q, batch_size)
        k = self.create_heads(k, batch_size)
        v = self.create_heads(v, batch_size)
        
        z, attention_weights = normalized_dot_product_attention(q, k, v, mask)
        
        concat_z = tf.transpose(z, perm=[0, 2, 1, 3])
        
        concat_z = tf.reshape(concat_z, (batch_size, -1, self.embedding_dim))
        
        z = self.Wz(concat_z)
        
        return z, attention_weights


def feed_forward(embedding_dim, ff_hidden_dim):
    
    hidden_layer = tf.keras.layers.Dense(ff_hidden_dim, activation='relu') #(batch size, seq len, hidden dim)
    output_layer = tf.keras.layers.Dense(embedding_dim) #(batch size, seq len, embedding dim)
    
    return tf.keras.Sequential([
        hidden_layer,
        output_layer
    ])

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()
        
        self.mha = MHAttention(num_heads, embedding_dim)
        
        self.ff = feed_forward(embedding_dim, ff_hidden_dim)
        
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, input_tensor, training, mask):
        
        #Sublayer 1
        mha_output, _ = self.mha(input_tensor, input_tensor, input_tensor, mask)
        mha_output = self.dropout_1(mha_output, training=training)
        sublayer_1_output = self.layernorm_1(mha_output + input_tensor)
        
        #Sublayer 2
        ff_output = self.ff(sublayer_1_output)
        ff_output = self.dropout_2(ff_output, training=training)
        return self.layernorm_2(ff_output + sublayer_1_output)
    
class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, num_encoders, embedding_dim, num_heads, ff_hidden_dim, dropout = 0.1):
        super(EncoderStack, self).__init__()
        
        self.num_encoders = num_encoders
        
        self.encoders = []
        for i in range(self.num_encoders):
            self.encoders.append(Encoder(embedding_dim, num_heads, ff_hidden_dim))
            
    def call(self, input_tensor, training, mask):
        
        output_tensor = input_tensor
        
        for i in range(self.num_encoders):
            output_tensor = self.encoders[i](output_tensor, training, mask)
            
        return output_tensor
    
class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout_rate=0.1):
        super(Decoder, self).__init__()
        
        self.mha_1 = MHAttention(num_heads, embedding_dim)
        self.mha_2 = MHAttention(num_heads, embedding_dim)
        
        self.ff = feed_forward(embedding_dim, ff_hidden_dim)
        
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, input_tensor, encoder_output, training, mask):
        
        #Sublayer 1
        mha_1_output, _ = self.mha_1(input_tensor, input_tensor, input_tensor, mask)
        mha_1_output = self.dropout_1(mha_1_output, training=training)
        sublayer_1_output = self.layernorm_1(mha_1_output + input_tensor)
        
        #Sublayer 2
        mha_2_output, _ = self.mha_2(input_tensor, sublayer_1_output, sublayer_1_output, mask)
        mha_2_output = self.dropout_2(mha_2_output, training=training)
        sublayer_2_output = self.layernorm_2(mha_2_output + sublayer_1_output)
        
        #Sublayer 2
        ff_output = self.ff(sublayer_2_output)
        ff_output = self.dropout_3(ff_output, training=training)
        return self.layernorm_3(ff_output + sublayer_2_output)
    
class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, num_decoders, embedding_dim, num_heads, ff_hidden_dim, dropout = 0.1):
        super(DecoderStack, self).__init__()
        
        self.num_decoders = num_decoders
        
        self.decoders = []
        for i in range(self.num_decoders):
            self.decoders.append(Decoder(embedding_dim, num_heads, ff_hidden_dim))
            
    def call(self, input_tensor, encoder_output, training, mask):
        
        output_tensor = input_tensor
        
        for i in range(self.num_decoders):
            output_tensor = self.decoders[i](output_tensor, encoder_output, training, mask)
            
        return output_tensor    