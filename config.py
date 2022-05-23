class Params(object):

    def __init__(self, X_train):
        
        self.n_channels = len(X_train[0][0])
        self.time_steps = len(X_train[0])
        self.ff_dim = 32
        self.n_head = 3
        self.dropout = 0.1
        self.n_classes = 3
        self.n_layers = 6
        self.batch_size = 64
        self.n_epochs = 1100
        self.learning_rate = 0.001
