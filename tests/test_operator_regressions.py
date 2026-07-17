import torch
import torch.nn as nn

from ai4plasma.core.network import FNN
from ai4plasma.operator.deepcsnet import DeepCSNetModel
from ai4plasma.operator.deeponet import DeepONet, DeepONetModel


def test_deeponet_split_by_trunk_collates_expected_shapes():
    network = DeepONet(FNN([2, 5, 3]), FNN([1, 5, 3]))
    model = DeepONetModel(network)
    branch_inputs = torch.randn(4, 2)
    trunk_inputs = torch.randn(6, 1)
    targets = torch.randn(4, 6)
    model.prepare_train_data(
        branch_inputs,
        trunk_inputs,
        targets,
        split_by_branch=False,
        batch_size=2,
    )
    model.set_loss_func(nn.MSELoss())

    branch_batch, trunk_batch, target_batch = next(iter(model.dataloader))

    assert branch_batch.shape == (4, 2)
    assert trunk_batch.shape == (2, 1)
    assert target_batch.shape == (4, 2)
    assert model.calc_loss((branch_batch, trunk_batch, target_batch)).ndim == 0


def test_deeponet_training_mode_and_prediction_grad_state():
    network = DeepONet(FNN([2, 4, 2]), FNN([1, 4, 2]))
    model = DeepONetModel(network)
    branch_inputs = torch.randn(3, 2)
    trunk_inputs = torch.randn(4, 1)
    targets = torch.randn(3, 4)
    model.prepare_train_data(branch_inputs, trunk_inputs, targets)

    prediction = model.predict(branch_inputs, trunk_inputs)
    assert not prediction.requires_grad
    assert not model.network.training

    model.train(num_epochs=1, print_loss=False)
    assert model.network.training


class TinyDeepCSNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, trunk_input, molecule_input=None, energy_input=None):
        return molecule_input[:, :1] * self.scale + trunk_input[:, 0].unsqueeze(0)


def test_deepcsnet_training_mode_and_prediction_grad_state():
    model = DeepCSNetModel(TinyDeepCSNetwork())
    molecule_inputs = torch.randn(3, 1)
    trunk_inputs = torch.randn(4, 1)
    targets = torch.randn(3, 4)
    model.dataloader = [(molecule_inputs, None, trunk_inputs, targets)]

    prediction = model.predict(trunk_inputs, molecule_input=molecule_inputs)
    assert not prediction.requires_grad
    assert not model.network.training

    model.train(num_epochs=1, print_loss=False, save_final_model=False)
    assert model.network.training
