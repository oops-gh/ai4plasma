import torch

from ai4plasma.core.network import CNN


def test_cnn_lazy_fc_parameters_are_registered_before_first_forward():
    network = CNN(
        conv_layers=[1, 2],
        fc_layers=[1, 4],
        input_dim=2,
        use_pooling=False,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    conv_weight_before = network.conv_net.conv1.weight.detach().clone()

    output = network(torch.randn(3, 1, 4, 4))

    assert output.shape == (3, 4)
    assert all(id(parameter) in optimizer_parameter_ids for parameter in network.parameters())
    assert torch.equal(network.conv_net.conv1.weight, conv_weight_before)

    output.sum().backward()
    optimizer.step()
