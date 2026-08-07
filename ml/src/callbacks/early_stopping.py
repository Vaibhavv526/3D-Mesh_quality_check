class EarlyStopping:

    def __init__(
        self,
        patience=5,
    ):

        self.patience = patience

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False

    def step(
        self,
        val_loss,
    ):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

        if self.counter >= self.patience:

            self.should_stop = True