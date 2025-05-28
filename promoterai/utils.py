import tensorflow.keras as tk


class CustomLearningRateScheduler(tk.callbacks.Callback):
    def __init__(self, learning_rate, weight_decay, epochs):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        scale = self._schedule(epoch)
        self.model.optimizer.learning_rate = self._learning_rate * scale
        self.model.optimizer.weight_decay = self._weight_decay * scale

    def on_epoch_end(self, epoch, logs=None):
        logs['lr'] = tk.backend.get_value(self.model.optimizer.learning_rate)
        logs['wd'] = tk.backend.get_value(self.model.optimizer.weight_decay)

    def _schedule(self, epoch):
        if epoch < 0.1 * self._epochs:
            return (epoch + 1) / (0.1 * self._epochs)
        elif epoch > 0.9 * self._epochs:
            return (self._epochs - epoch) / (0.1 * self._epochs)
        else:
            return 1


class CustomModelCheckpoint(tk.callbacks.Callback):
    def __init__(self, custom_model, model_folder):
        self._custom_model = custom_model
        self._model_folder = model_folder
        self._val_loss_best = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        if self._val_loss_best is None or val_loss < self._val_loss_best:
            self._val_loss_best = val_loss
            self._custom_model.save(self._model_folder)
