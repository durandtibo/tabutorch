from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from torch.nn.functional import mse_loss

from tabutorch.preprocessing import StandardScaler

SIZES = [1, 2, 3]
SHAPES = [(2, 4), (2, 3, 4), (2, 3, 3, 4)]

# TODO (tibo): behavior when NaN values # noqa: TD003

####################################
#     Tests for StandardScaler     #
####################################


def test_standard_scaler_str() -> None:
    assert str(StandardScaler(num_features=4)).startswith("StandardScaler(")


@pytest.mark.parametrize("feature_size", SIZES)
def test_standard_scaler_init_feature_size(feature_size: int) -> None:
    module = StandardScaler(num_features=feature_size)
    assert module.mean.equal(torch.zeros(feature_size))
    assert module.scale.equal(torch.ones(feature_size))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_forward(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=3).to(device=device)
    module.load_state_dict(
        {"mean": torch.tensor([4.0, 2.0, 1.0]), "scale": torch.tensor([1.0, 2.0, 4.0])}
    )
    module.train(mode)
    out = module(torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=device))
    assert objects_are_allclose(
        out,
        torch.tensor([[-3.0, 0.0, 0.5], [-3.0, -0.5, 0.0], [-2.0, 0.0, 0.25]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_forward_shape(device: str, shape: tuple[int, ...], mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.randn(*shape, device=device)
    out = module(x)
    assert x is not out
    assert objects_are_equal(out, x)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_backward(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, 4, device=device, requires_grad=True)
    out = module(x)
    loss = mse_loss(out, torch.randn(batch_size, 4, device=device)).mean()
    loss.backward()
    assert objects_are_equal(out, x)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    module.fit(
        torch.tensor(
            [[4.0, 3.0, 4.0, 1.0], [0.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]], device=device
        )
    )
    assert objects_are_allclose(
        module.state_dict(),
        OrderedDict(
            {
                "mean": torch.tensor([2.0, 2.0, 3.0, 2.0], device=device),
                "scale": torch.tensor([2.0, 1.0, 1.0, 1.0], device=device),
            }
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_shape(device: str, shape: tuple[int, ...], mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.randn(*shape, device=device)
    module.fit(x)
    out = module.transform(x)
    assert out.shape == shape
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_constant(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    module.fit(torch.ones(10, 4, device=device))
    assert objects_are_allclose(
        module.state_dict(),
        OrderedDict(
            {
                "mean": torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
                "scale": torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
            }
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_transform(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    out = module.fit_transform(
        torch.tensor(
            [[4.0, 3.0, 4.0, 1.0], [0.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]], device=device
        )
    )
    assert objects_are_allclose(
        module.state_dict(),
        OrderedDict(
            {
                "mean": torch.tensor([2.0, 2.0, 3.0, 2.0], device=device),
                "scale": torch.tensor([2.0, 1.0, 1.0, 1.0], device=device),
            }
        ),
    )
    assert objects_are_allclose(
        out,
        torch.tensor(
            [[1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]], device=device
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_transform_shape(
    device: str, shape: tuple[int, ...], mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.randn(*shape, device=device)
    out = module.fit_transform(x)
    assert out.shape == shape
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_transform_constant(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    out = module.fit_transform(torch.ones(10, 4, device=device))
    assert objects_are_allclose(
        module.state_dict(),
        OrderedDict(
            {
                "mean": torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
                "scale": torch.tensor([1.0, 1.0, 1.0, 1.0], device=device),
            }
        ),
    )
    assert objects_are_allclose(out, torch.zeros(10, 4, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_transform(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=3).to(device=device)
    module.load_state_dict(
        {"mean": torch.tensor([4.0, 2.0, 1.0]), "scale": torch.tensor([1.0, 2.0, 4.0])}
    )
    module.train(mode)
    out = module.transform(
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=device)
    )
    assert objects_are_allclose(
        out,
        torch.tensor([[-3.0, 0.0, 0.5], [-3.0, -0.5, 0.0], [-2.0, 0.0, 0.25]], device=device),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_transform_shape(device: str, shape: tuple[int, ...], mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.randn(*shape, device=device)
    out = module.transform(x)
    assert x is not out
    assert objects_are_equal(out, x)


def test_standard_scaler_load_state_dict() -> None:
    module = StandardScaler(num_features=4)
    module.load_state_dict({"mean": -torch.ones(4), "scale": torch.ones(4).mul(0.5)})
    assert objects_are_equal(module(torch.ones(2, 4)), torch.ones(2, 4).mul(4.0))
    assert objects_are_equal(
        module.state_dict(), OrderedDict({"mean": -torch.ones(4), "scale": torch.ones(4).mul(0.5)})
    )


def test_standard_scaler_state_dict() -> None:
    module = StandardScaler(num_features=4)
    assert objects_are_equal(module(torch.ones(2, 4)), torch.ones(2, 4))
    assert objects_are_equal(
        module.state_dict(), OrderedDict({"mean": torch.zeros(4), "scale": torch.ones(4)})
    )
