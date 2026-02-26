import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        xavier_scale = np.sqrt(2.0 / (input_size + hidden_size))

        self.Wf = np.random.randn(hidden_size, input_size) * xavier_scale
        self.Uf = np.random.randn(hidden_size, hidden_size) * xavier_scale
        self.bf = np.zeros((hidden_size, 1))

        self.Wi =np.random.randn(hidden_size, input_size) * xavier_scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * xavier_scale
        self.bi = np.zeros((hidden_size, 1))

        self.Wc = np.random.randn(hidden_size, input_size) * xavier_scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * xavier_scale
        self.bc = np.zeros((hidden_size, 1))

        self.Wo = np.random.randn(hidden_size, input_size) * xavier_scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * xavier_scale
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, x, h_prev, c_prev):
        self.f = self.sigmoid(self.Wf @ x + self.Uf @ h_prev + self.bf)    #Forget gate
        self.i = self.sigmoid(self.Wi @ x + self.Ui @ h_prev + self.bi)    #Input gate
        self.c_hat = self.tanh(self.Wc @ x + self.Uc @ h_prev + self.bc)   #Candidate value
        self.c = self.f * c_prev + self.i * self.c_hat                     #Cell state
        self.o = self.sigmoid(self.Wo @ x + self.Uo @ h_prev + self.bo)    #Output gate
        self.h = self.o * self.tanh(self.c)                                #hidden state

        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev

        return self.h, self.c



class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.cell = LSTMCell(input_size, hidden_size)

        xavier_scale = np.sqrt(2.0 / hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)*xavier_scale
        self.by = np.zeros((output_size, 1))

        self.reset_cache()

    def reset_cache(self):
        self.h_cache = []
        self.c_cache = []
        self.x_cache = []
        self.y_cache = []

    def forward_sequence(self, X, return_sequences=False):
        seq_len = X.shape[0]
        batch_size = X.shape[2] if len(X.shape) > 2 else 1

        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))

        self.reset_cache()
        outputs = []

        for t in range(seq_len):
            x_t = X[t].reshape(self.input_size, -1)
            h, c = self.cell.forward(x_t, h, c)

        y_t = self.Wy @ h + self.by

        self.h_cache.append(h.copy())
        self.c_cache.append(c.copy())
        self.x_cache.append(x_t.copy())
        self.y_cache.append(y_t.copy())

        outputs.append(y_t)

        if return_sequences:
            return np.array(outputs)
        else:
            return outputs[-1]


    def backward_sequence(self, dL_dy_seq, clip_value=5.0):

        seq_len = len(self.h_cache)

        dWf = np.zeros_like(self.cell.Wf)
        dUf = np.zeros_like(self.cell.Uf)
        dbf = np.zeros_like(self.cell.bf)

        dWi = np.zeros_like(self.cell.Wi)
        dUi = np.zeros_like(self.cell.Ui)
        dbi = np.zeros_like(self.cell.bi)

        dWc = np.zeros_like(self.cell.Wc)
        dUc = np.zeros_like(self.cell.Uc)
        dbc = np.zeros_like(self.cell.bc)

        dWo = np.zeros_like(self.cell.Wo)
        dUo = np.zeros_like(self.cell.Uo)
        dbo = np.zeros_like(self.cell.bo)

        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(self.h_cache[0])
        dc_next = np.zeros_like(self.c_cache[0])

        for t in reversed(range(seq_len)):
            h = self.h_cache[t]
            c = self.c_cache[t]
            x = self.x_cache[t]

            c_prev = self.c_cache[t-1] if t>0 else np.zeros_like(c)
            h_prev = self.h_cache[t-1] if t>0 else np.zeros_like(h)

            # Gradient from output layer
            dy = dL_dy_seq[t].reshape(self.output_size, -1)
            dWy += dy @ h.T
            dby += np.sum(dy, axis=1, keepdims=True)
            dh = self.Wy.T @ dy + dh_next

            f = self.cell.sigmoid(self.cell.Wf @ x + self.cell.Uf @ h_prev + self.cell.bf)
            i = self.cell.sigmoid(self.cell.Wi @ x + self.cell.Ui @ h_prev + self.cell.bi)
            c_hat = self.cell.tanh(self.cell.Wc @ x + self.cell.Uc @ h_prev + self.cell.bc)
            o = self.cell.sigmoid(self.cell.Wo @ x + self.cell.Uo @ h_prev + self.cell.bo)

            do = dh * self.cell.tanh(c)
            do_input = do * o * (1 - o)

            dWo += do_input @ x.T
            dUo += do_input @ h_prev.T
            dbo += np.sum(do_input, axis=1, keepdims=True)

            # Backprop through cell state
            dc = dh * o * (1 - self.cell.tanh(c)**2) + dc_next
            dc_prev = dc * f

            # Backprop through forget gate
            df = dc * c_prev
            df_input = df * f * (1-f)

            dWf += df_input @ x.T
            dUf += df_input @ h_prev.T
            dbf += np.sum(df_input, axis=1, keepdims=True)

            #BackPropagation through input Gate
            di =  dc * c_hat
            di_input = di * i * (1-i)

            dWi += di_input @ x.T
            dUi += di_input @ h_prev.T
            dbi += np.sum(di_input, axis=1, keepdims=True)

            # Through candidate
            dc_hat = dc * i
            dc_hat_input = dc_hat * (1 - c_hat**2)

            dWc += dc_hat_input @ x.T
            dUc += dc_hat_input @ h_prev.T
            dbc += np.sum(dc_hat_input, axis=1, keepdims=True)

            # Gradients for next iteration
            dh_next = (self.cell.Uf.T @ df_input +
                        self.cell.Ui.T @ di_input +
                        self.cell.Uc.T @ dc_hat_input +
                        self.cell.Uo.T @ do_input)
            dc_next = dc_prev

        gradients = [dWf, dUf, dbf, dWi, dUi, dbi, dWc, dUc, dbc, dWo, dUo, dbo, dWy, dby]
        for grad in gradients:
            np.clip(grad, -clip_value, clip_value, out=grad)

        self.cell.Wf -= self.learning_rate * dWf
        self.cell.Uf -= self.learning_rate * dUf
        self.cell.bf -= self.learning_rate * dbf

        self.cell.Wi -= self.learning_rate * dWi
        self.cell.Ui -= self.learning_rate * dUi
        self.cell.bi -= self.learning_rate * dbi

        self.cell.Wc -= self.learning_rate * dWc
        self.cell.Uc -= self.learning_rate * dUc
        self.cell.bc -= self.learning_rate * dbc

        self.cell.Wo -= self.learning_rate * dWo
        self.cell.Uo -= self.learning_rate * dUo
        self.cell.bo -= self.learning_rate * dbo

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    def train_step(self, X, y):
        predictions = self.forward_sequence(X, return_sequences=True)

        loss = np.mean((predictions - y)**2)

        dL_dy = 2 * (predictions - y) / y.size

        self.backward_sequence(dL_dy)

        return loss

    def predict(self, X, return_sequences=False):
        return self.forward_sequence(X, return_sequences)
