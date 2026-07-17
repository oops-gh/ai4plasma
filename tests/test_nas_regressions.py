import torch
import torch.nn as nn

from ai4plasma.piml.nas_pinn import NasPINN


class TinyNASNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.architecture = nn.Parameter(torch.tensor([0.5]))

    def arch_parameters(self):
        return [self.architecture]

    def searched_neuron(self):
        return [1]


class TinyPINNModel:
    def __init__(self):
        self.network = TinyNASNetwork()
        self.writer = None

    def calc_loss(self):
        return self.network.weight.square().sum(), {}

    def calc_loss_archi(self):
        return self.network.architecture.square().sum(), {}

    def _execute_visualization_callbacks(self, *args, **kwargs):
        return None


def test_nas_search_without_tensorboard_writer():
    pinn_model = TinyPINNModel()
    nas = NasPINN(pinn_model)
    pinn_model.network.eval()

    nas.search(
        outer_epochs=1,
        inner_epochs=1,
        tensorboard_logdir=None,
        log_freq=1,
        print_freq=0,
        checkpoint_freq=0,
    )

    assert nas.writer is None
    assert pinn_model.network.training
