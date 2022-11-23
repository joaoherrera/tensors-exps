""" A basic model for machine learning.
"""

import numpy as np
import torch


class Supervision:
    EPOCHS = 100

    def __init__(self, device, model):
        self.device = device
        self.model = model

        self.model.to(self.device)  # Load model in the GPU

    def train(self, dataset, optimizer, loss_func):
        loss_training = []

        # Set module status to training. Implemented in torch.nn.Module
        self.model.train()

        with torch.set_grad_enabled(True):
            for batch in dataset:
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                # Loss computation and weights correction
                loss = loss_func(y_pred, y_true)
                loss.backward()    # backpropagation
                optimizer.step()

                loss_value = loss.item()
                loss_training.append(loss_value)
                
                print(f"Training loss: {loss_value}")
        return np.mean(loss_training)

    def evaluate(self, dataset, coef_func):
        coef_validation = []

        # Set module status to evalutation. Implemented in torch.nn.Module
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                x_pred, y_true = batch
                x_pred, y_true = x_pred.to(self.device), y_true.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                coef = coef_func(y_pred, y_true)
                coef_value = coef.item()
                coef_validation.append(coef_value)
                
                print(f"Validation loss {coef_value}")
        return np.mean(coef_validation)

    def fit(self, training_dataset, validation_dataset, optimizer, loss_func, coef_func):
        for epoch in range(self.EPOCHS):
            print(f"Epoch {epoch}")

            loss_training = self.train(training_dataset, optimizer, loss_func)
            coef_evalutation = self.evaluate(validation_dataset, coef_func)

            print(f"Loss training: {loss_training}")
            print(f"Loss validation: {coef_evalutation}")


class SemiSupervision(Supervision):
    """SemiSupervision works similar to Supervision. However, instead of considering all training
    samples as supervised samples, some of them are predicted first and the results are shuffled
    to the ground-truth samples.
    """

    def __init__(self, device, model):
        super(SemiSupervision, self).__init__(device, model)

    def generate_training_dataset(self, training_dataset, samples, labels):
        pass

    def predict(self, dataset_unlabeled, number_of_samples=50):
        samples = []
        labels = []

        self.model.eval()

        with torch.no_grad():
            for batch in dataset_unlabeled:
                x_pred = batch.to(self.device)

                # Predict
                y_pred = self.model(x_pred)

                samples.extend(x_pred.detach().cpu().numpy().transpose(0, 2, 3, 1))
                labels.extend(y_pred.detach().cpu().numpy().transpose(0, 2, 3, 1))

        return samples, labels

    def fit(self, training_dataset_labeled, training_dataset_unlabeled, validation_dataset,
            optimizer, loss_func, coef_func, unlabeled_batch_size=50):

        # Train labeled
        super(SemiSupervision, self).\
            fit(training_dataset_labeled, validation_dataset, optimizer, loss_func, coef_func)

        while True:
            # Predict unlabeled
            samples, labels = self.predict(training_dataset_unlabeled, unlabeled_batch_size)

            # Join to labeled
            self.generate_training_dataset(training_dataset_labeled, samples, labels)

            # Train labeled
            super(SemiSupervision, self).\
                fit(training_dataset, validation_dataset, optimizer, loss_func, coef_func)

            # Break looping when there's no samples left to be trained.
            if len(samples) < unlabeled_batch_size:
                break
