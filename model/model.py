from __future__ import absolute_import, division, print_function, unicode_literals 
import tensorflow as tf


class BiLSTMAttn(tf.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(dropout) 
        lstm_cells = [tf.keras.layers.LSTMCELL(hidden_dim // 2, dropout=dropout if num_layers > 1 else 0) for _ in range (num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)
        self.encoder = tf.keras.layers.Bidirectional(stacked_lstm) 


    def attnetwork(self, encoder_out, final_hidden):
        hidden = tf.squeeze(final_hidden, axis=0)
        attn_weights = tf.squeeze(tf.matmul(encoder_out. tf.expand_dims(hidden, 2)), axis=2)
        soft_attn_weights = tf.nn.softmax(attn_weights, 1)
        new_hidden = tf.squeeze(tf.matmul(tf.transpose(encoder_out, perm=[1, 2]), tf.expand_dims(soft_attn_weights, 2)), axis=2)

        return new_hidden

    def forward(self, features):
        features = self.dropout(features) 
        outputs, (hn, cn) = self.encoder(features)
        fbout = outputs[:, :, :self.hidden_dim // 2] + outputs[:, :, self.hidden_dim // 2:]
        fbhn = tf.expand_dims(hn[-2, :, :] + hn[-1, :, :], axis=0)
        attn_out = self.attnetwork(fbout, fbhn)

        return attn_out  


class BiLSTM(tf.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = tf.keras.layers.Dropout(dropout) 
        lstm_cells = [tf.keras.layers.LSTMCELL(hidden_dim // 2, dropout=dropout) for _ in range(num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)
        self.bilstm = tf.keras.layers.Bidirectional(stacked_lstm) 

    def forward(self, features):
        features = self.dropout(features)
        outputs, hidden_state = self.bilstm(features)
        return outputs, hidden_state  


class HistoricCurrent(tf.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, model):
        super().__init__()
        self.model = model
        if self.model == "tlstm":
            self.historic_model = TimeLSTM(embedding_dim, hidden_dim)
        elif self.model == "bilstm":
            self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        elif self.model == "bilstm-attention":
            self.historic_model = BiLSTMAttn(embedding_dim, hidden_dim, num_layers, dropout)

        self.fc_ct = tf.keras.layers.Dense(hidden_dim)
        self.fc_ct_attn = tf.keras.layers.Dense(hidden_dim//2)

        self.fc_concat = tf.keras.layers.Dense(hidden_dim)
        self.fc_concat_attn = tf.keras.layers.Dense(hidden_dim)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.final = tf.keras.layers.Dense(2)

    @staticmethod
    def combine_features(tweet_features, historic_features):
        return tf.concat([tweet_features, historic_features], 1)

    def forward(self, tweet_features, historic_features, timestamp):
        if self.model == "tlstm":
            outputs = self.historic_model(historic_features, timestamp)
            tweet_features = tf.nn.relu(self.fc_ct(tweet_features))
            outputs = tf.reduce_mean(outputs, axis=1)
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = tf.nn.relu(self.fc_concat(combined_features))
        elif self.model == "bilstm":
            outputs, (h_n, c_n) = self.historic_model(historic_features)
            outputs = tf.reduce_mean(outputs, axis=1)
            tweet_features = tf.nn.relu(self.fc_ct(tweet_features))
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = tf.nn.relu(self.fc_concat(combined_features))
        elif self.model == "bilstm-attention":
            outputs = self.historic_model(historic_features)
            tweet_features = tf.nn.relu(self.fc_ct_attn(tweet_features))
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = tf.nn.relu(self.fc_concat_attn(combined_features))

        x = self.dropout(x)

        return self.final(x)

    
class Historic(tf.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc1 = tf.keras.layer.Dense(32)
        self.final = tf.keras.layers.Dense(2)

    def __call__(self, tweet_features, historic_features, timestamp):
        outputs, (h_n, c_n) = self.historic_model(historic_features)
        hidden = tf.concat([h_n[-2, :, :], h_n[-1, :, :]], axis=1)
        x = tf.nn.relu(self.fc1(hidden))
        return self.final(x)


class Current(tf.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc2 = tf.keras.layers.Dense(32)
        self.final = tf.keras.layers.Dense(2)

    def __call__(self, tweet_features, historic_features, timestamp):
        x = tf.nn.relu(self.fc1(tweet_features))
        x = self.dropout(x)
        x = tf.nn.relu(self.fc2(x))
        return self.final(x)


class TimeLSTM(tf.Module): 
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_all = tf.keras.layers.Dense(hidden_size * 4)
        self.U_all = tf.keras.layers.Dense(hidden_size * 4)
        self.W_d = tf.keras.layers.Dense(hidden_size)
        self.bidirectional = bidirectional

    def __call__(self, inputs, timestamps, reverse=False):
        b, seq, embed = tf.shape(inputs)
        h = tf.zeros([b, self.hidden_size])
        c = tf.zeros([b, self.hidden_size])
        
        outputs = []
        for s in range(seq):
            c_s1 = tf.math.tanh(self.W_d(c))
            timestamps = tf.cast(timestamps, tf.float32)
            c_s2 = c_s1 * tf.broadcast_to(timestamps[:, s:s + 1], c_s1.shape)  # expands a tensor to the same size as c_s1.shape
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = tf.split(outs, 4, axis=1)
            f = tf.math.sigmoid(f)
            i = tf.math.sigmoid(i)
            o = tf.math.sigmoid(o)
            c_tmp = tf.math.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * tf.math.tanh(c)
            outputs.append(h)  
        if reverse:
            outputs.reverse()  
        outputs = tf.stack(outputs, axis=1)  # Concatenates a sequence of tensors along a new dimension.
        return outputs

