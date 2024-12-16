from __future__ import annotations

import torch
from coola import objects_are_allclose
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from tabutorch.preprocessing import StandardScaler

####################################
#     Tests for StandardScaler     #
####################################


def test_standard_scaler_fit() -> None:
    x = torch.randn(100, 10)
    # sklearn uses the unbiased variant to estimate the standard deviation
    m1 = StandardScaler(num_features=10, std_correction=0)
    m1.fit(x)
    m2 = SklearnStandardScaler()
    m2.fit(x.numpy())
    # The mean and standard deviation are computed using the float64 precision
    assert objects_are_allclose(m1.mean.numpy().astype(float), m2.mean_, atol=1e-6)
    assert objects_are_allclose(m1.scale.numpy().astype(float), m2.scale_, atol=1e-6)


def test_standard_scaler_fit_transform() -> None:
    x = torch.randn(100, 10)
    # sklearn uses the unbiased variant to estimate the standard deviation
    m1 = StandardScaler(num_features=10, std_correction=0)
    out1 = m1.fit_transform(x)
    m2 = SklearnStandardScaler()
    out2 = m2.fit_transform(x.numpy())
    # The mean and standard deviation are computed using the float64 precision
    assert objects_are_allclose(m1.mean.numpy().astype(float), m2.mean_, atol=1e-6)
    assert objects_are_allclose(m1.scale.numpy().astype(float), m2.scale_, atol=1e-6)
    assert objects_are_allclose(out1.numpy(), out2, atol=1e-6)


def test_standard_scaler_transform() -> None:
    x = torch.randn(100, 10)
    # sklearn uses the unbiased variant to estimate the standard deviation
    m1 = StandardScaler(num_features=10, std_correction=0)
    m1.fit(x)
    out1 = m1.transform(x)
    m2 = SklearnStandardScaler()
    m2.fit(x.numpy())
    out2 = m2.transform(x.numpy())
    # The mean and standard deviation are computed using the float64 precision
    assert objects_are_allclose(m1.mean.numpy().astype(float), m2.mean_, atol=1e-6)
    assert objects_are_allclose(m1.scale.numpy().astype(float), m2.scale_, atol=1e-6)
    assert objects_are_allclose(out1.numpy(), out2, atol=1e-6)


def test_standard_scaler_fit_transform_1_sample() -> None:
    x = torch.randn(1, 10)
    # sklearn uses the unbiased variant to estimate the standard deviation
    m1 = StandardScaler(num_features=10, std_correction=0)
    out1 = m1.fit_transform(x)
    m2 = SklearnStandardScaler()
    out2 = m2.fit_transform(x.numpy())
    # The mean and standard deviation are computed using the float64 precision
    assert objects_are_allclose(m1.mean.numpy().astype(float), m2.mean_, atol=1e-6)
    assert objects_are_allclose(m1.scale.numpy().astype(float), m2.scale_, atol=1e-6)
    assert objects_are_allclose(out1.numpy(), out2, atol=1e-6)
