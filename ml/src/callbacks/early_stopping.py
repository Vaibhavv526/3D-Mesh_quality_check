class EarlyStopping:

    def __init__(
        self,
        patience=5,
    ):

        self.patience = patience

        self.best_f1 = 0.0

        self.counter = 0

        self.should_stop = False

    def step(
        self,
        val_f1,
    ):

        if val_f1 > self.best_f1:

            self.best_f1 = val_f1

            self.counter = 0

        else:

            self.counter += 1

        if self.counter >= self.patience:

            self.should_stop = True