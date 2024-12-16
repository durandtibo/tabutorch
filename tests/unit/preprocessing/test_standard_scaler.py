from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from torch.nn.functional import mse_loss

from tabutorch.preprocessing import StandardScaler

SIZES = [1, 2, 3]

# TODO (tibo): behavior for batch size == 1  # noqa: TD003
# TODO (tibo): behavior when std == 0 # noqa: TD003

####################################
#     Tests for StandardScaler     #
####################################


def test_standard_scaler_str() -> None:
    assert str(StandardScaler(num_features=4)).startswith("StandardScaler(")


@pytest.mark.parametrize("feature_size", SIZES)
def test_standard_scaler_init_feature_size(feature_size: int) -> None:
    module = StandardScaler(num_features=feature_size)
    assert module.mean.equal(torch.zeros(feature_size))
    assert module.std.equal(torch.ones(feature_size))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_forward_2d(
    device: str, batch_size: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, feature_size, device=device)
    out = module(x)
    assert objects_are_equal(out, x)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_forward_3d(
    device: str, batch_size: int, d1: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, d1, feature_size, device=device)
    out = module(x)
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
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_2d(
    device: str, batch_size: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, feature_size, device=device)
    module.fit(x)
    out = module.transform(x)
    assert out.shape == (batch_size, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_3d(
    device: str, batch_size: int, d1: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, d1, feature_size, device=device)
    module.fit(x)
    out = module.transform(x)
    assert out.shape == (batch_size, d1, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_custom(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.tensor(
        [[4.0, 3.0, 4.0, 1.0], [0.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]], device=device
    )
    module.fit(x)
    assert objects_are_allclose(
        module.state_dict(),
        OrderedDict(
            {
                "mean": torch.tensor([2.0, 2.0, 3.0, 2.0], device=device),
                "std": torch.tensor([2.0, 1.0, 1.0, 1.0], device=device),
            }
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_transform_2d(
    device: str, batch_size: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, feature_size, device=device)
    out = module.fit_transform(x)
    assert out.shape == (batch_size, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_transform_3d(
    device: str, batch_size: int, d1: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, d1, feature_size, device=device)
    out = module.fit_transform(x)
    assert out.shape == (batch_size, d1, feature_size)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_fit_transform_custom(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.train(mode)
    x = torch.tensor(
        [[4.0, 3.0, 4.0, 1.0], [0.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]], device=device
    )
    out = module.fit_transform(x)
    assert objects_are_allclose(
        module.state_dict(),
        OrderedDict(
            {
                "mean": torch.tensor([2.0, 2.0, 3.0, 2.0], device=device),
                "std": torch.tensor([2.0, 1.0, 1.0, 1.0], device=device),
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
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_transform_2d(
    device: str, batch_size: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, feature_size, device=device)
    out = module.transform(x)
    assert objects_are_equal(out, x)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("d1", SIZES)
@pytest.mark.parametrize("feature_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_transform_3d(
    device: str, batch_size: int, d1: int, feature_size: int, mode: bool
) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=feature_size).to(device=device)
    module.train(mode)
    x = torch.randn(batch_size, d1, feature_size, device=device)
    out = module.transform(x)
    assert objects_are_equal(out, x)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("mode", [True, False])
def test_standard_scaler_transform_custom(device: str, mode: bool) -> None:
    device = torch.device(device)
    module = StandardScaler(num_features=4).to(device=device)
    module.load_state_dict(
        {"mean": torch.tensor([1, 2, 3, 4]), "std": torch.tensor([0.1, 0.2, 0.3, 0.4])}
    )
    module.train(mode)
    out = module.transform(
        torch.tensor([[4.0, 3.0, 2.0, 1.0], [0.0, 1.0, 2.0, 3.0]], device=device)
    )
    assert objects_are_allclose(
        out,
        torch.tensor(
            [[30.0, 5.0, -10.0 / 3.0, -7.5], [-10.0, -5.0, -10.0 / 3.0, -2.5]], device=device
        ),
    )


def test_standard_scaler_load_state_dict() -> None:
    module = StandardScaler(num_features=4)
    module.load_state_dict({"mean": -torch.ones(4), "std": torch.ones(4).mul(0.5)})
    assert objects_are_equal(module(torch.ones(2, 4)), torch.ones(2, 4).mul(4.0))
    assert objects_are_equal(
        module.state_dict(), OrderedDict({"mean": -torch.ones(4), "std": torch.ones(4).mul(0.5)})
    )


def test_standard_scaler_state_dict() -> None:
    module = StandardScaler(num_features=4)
    assert objects_are_equal(module(torch.ones(2, 4)), torch.ones(2, 4))
    assert objects_are_equal(
        module.state_dict(), OrderedDict({"mean": torch.zeros(4), "std": torch.ones(4)})
    )
