import math

import torch
import torch.nn as nn

from ai4plasma.piml.pinn import PINN
from ai4plasma.utils.math import df_dX


class SecondDerivativePINN(PINN):
    def __init__(self):
        self.points = torch.linspace(0.0, 1.0, 6).reshape(-1, 1).requires_grad_(True)
        network = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        super().__init__(network)

    @staticmethod
    def residual(network, x):
        value = network(x)
        first_derivative = df_dX(value, x)
        return df_dX(first_derivative, x)

    def _define_loss_terms(self):
        self.add_equation("domain", self.residual, data=self.points)


class AdaptiveWeightPINN(PINN):
    def __init__(self):
        self.points = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1).requires_grad_(True)
        super().__init__(nn.Linear(1, 1))

    @staticmethod
    def first_residual(network, x):
        return network(x)

    @staticmethod
    def second_residual(network, x):
        return 2.0 * network(x) + 1.0

    def _define_loss_terms(self):
        self.add_equation("first", self.first_residual, data=self.points)
        self.add_equation("second", self.second_residual, data=self.points)


def test_compute_residual_keeps_autograd_enabled_for_derivatives():
    model = SecondDerivativePINN()
    input_data = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)

    residual = model.compute_residual("domain", input_data)

    assert residual.shape == input_data.shape
    assert torch.isfinite(residual).all()
    assert not residual.requires_grad
    assert not input_data.requires_grad
    assert model.network.training


def test_train_restores_training_mode_after_prediction():
    model = SecondDerivativePINN()
    model.predict(model.points.detach())
    assert not model.network.training

    model.train(num_epochs=1, print_loss=False)

    assert model.network.training


def test_batched_adaptive_weights_accept_numeric_epoch_losses():
    model = AdaptiveWeightPINN()
    model.enable_adaptive_weights(enable=True, update_freq=1)

    model.train(num_epochs=1, batch_size=2, print_loss=False)

    for term in model.equation_terms.values():
        assert isinstance(term.weight, float)
        assert math.isfinite(term.weight)
        assert term.weight > 0
