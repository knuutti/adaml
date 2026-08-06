
import numpy as np
import matplotlib.pyplot as plt


class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lr = lr

        self.Wxh = rng.randn(hidden_size, input_size)
        self.Whh = rng.randn(hidden_size, hidden_size)
        self.Why = rng.randn(output_size, hidden_size)

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, X):
        # In forward model we feed the data X through the network.

        # X.shape = (batch, seq, features)
        batch, seq_len, _ = X.shape

        h = np.zeros((batch, seq_len + 1, self.hidden_size))

        for t in range(seq_len):
            xt = X[:, t, :].reshape(batch, -1)  
            # compute the hidden
            pre = xt.dot(self.Wxh.T) + h[:, t, :].dot(self.Whh.T) + self.bh.T
            h[:, t+1, :] = np.tanh(pre)

        # Compute the output
        y_pred = h[:, -1, :].dot(self.Why.T) + self.by.T  

        return h, y_pred

    def loss(self, y_pred, y_true):
        # MSE loss
        diff = y_pred - y_true
        loss = np.mean(diff ** 2)

        return loss, diff

    def bptt_update(self, X, h, y_pred, y_true):
        # Using backpropagation through time (bptt) to update the weights
        # X: (batch, seq_len, input_size)

        batch, seq_len, _ = X.shape

        # Init the gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dy = (y_pred - y_true) * (2.0 / batch) 

        h_last = h[:, -1, :].reshape(batch, self.hidden_size)

        dWhy += dy.T.dot(h_last)
        dby += dy.T.sum(axis=1, keepdims=True)

        dh_next = dy.dot(self.Why)

        for t in reversed(range(seq_len)):
            ht = h[:, t+1, :]
            ht_prev = h[:, t, :]
            
            # derivative of tanh
            dt = dh_next * (1 - ht ** 2)
            dbh += dt.T.sum(axis=1, keepdims=True)

            xt = X[:, t, :].reshape(batch, -1)
            dWxh += dt.T.dot(xt)
            dWhh += dt.T.dot(ht_prev)

            # Propagate dh to the next time step
            dh_next = dt.dot(self.Whh)

        for grad in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(grad, -5, 5, out=grad)

        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

        return

    def train(self, X, y, epochs=50, batch_size=32, verbose=True):

        n = X.shape[0]

        for epoch in range(1, epochs+1):
            
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            loss_epoch = 0.0
            losses = []
            for i in range(0, n, batch_size):
                xb = X_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]

                # feed the data through the network
                h, y_pred = self.forward(xb)
                loss, _ = self.loss(y_pred, yb)
                loss_epoch += loss * xb.shape[0]

                self.bptt_update(xb, h, y_pred, yb)

            loss_epoch /= n
            losses.append(loss_epoch)

        return losses


def generate_sine_sequences(n_samples=2000, seq_len=20, input_size=1, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 50, n_samples * seq_len * input_size)
    data = np.sin(x) + 0.1 * rng.randn(n_samples * seq_len * input_size)
    X = data.reshape(n_samples, seq_len, input_size)
    rolled = np.roll(data, -1).reshape(n_samples, seq_len, input_size)
    y_last = rolled[:, -1, :]
    return X.astype(np.float32), y_last.astype(np.float32)


# Creating the data
X, y = generate_sine_sequences(n_samples=1500, seq_len=20, input_size=1)
print(f"X shape: {X.shape}, y shape: {y.shape}")
# train/test split
split = int(0.8 * X.shape[0])
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Training the network
rnn = RNN(input_size=1, hidden_size=80, output_size=1, lr=1e-3)
losses = rnn.train(X_train, y_train, epochs=600, batch_size=8, verbose=True)

# Test set
_, y_pred_test = rnn.forward(X_test)
test_loss, _ = rnn.loss(y_pred_test, y_test)
print(f"\nTest MSE: {test_loss:.6f}\n")


# Plotting the predictions.
plt.plot(y_test)
plt.plot(y_pred_test)
plt.savefig("rnn_pred.png")
plt.show()
