from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax_extra_activation import (
    Celu,
    HardShrink,
    HardSwish,
    LogSigmoid,
    Mish,
    Relu6,
    Silu,
    SoftShrink,
    SoftSign,
    TanhExp,
    TanhShrink,
)


def test_celu() -> None:
    layer = Celu()
    x = jnp.array([0.0, 2.0])
    y = layer(x)
    assert tl.to_list(y) == [0.0, 2.0]


def test_hardshrink() -> None:
    layer = HardShrink()
    x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = layer(x)
    assert tl.to_list(y) == [-1.0, 0.0, 0.0, 0.0, 1.0]


def test_hardswish() -> None:
    layer = HardSwish()
    x = jnp.array([-4.0, -3.0, 0.0, 3.0, 4.0])
    y = layer(x)
    assert tl.to_list(y) == [0.0, 0.0, 0.0, 3.0, 4.0]


def test_logsigmoid() -> None:
    layer = LogSigmoid()
    x = jnp.array([-4.0, 0.0, 4.0])
    y = layer(x)
    assert tl.to_list(y) == [
        jnp.log(1 / (1 + jnp.exp(4.0))),
        jnp.log(0.5),
        jnp.log(1 / (1 + jnp.exp(-4.0))),
    ]


def test_mish() -> None:
    layer = Mish()
    softplus = tl.Softplus()
    x = jnp.array([-4.0, 0.0, 4.0])
    y = layer(x)
    x_prime = jnp.array([-4.0, 4.0])
    y_prime = softplus(x_prime)
    y_prime_list = tl.to_list(y_prime)
    assert tl.to_list(y) == [
        -4.0 * jnp.tanh(y_prime_list[0]),
        0.0,
        4.0 * jnp.tanh(y_prime_list[1]),
    ]


def test_relu6() -> None:
    layer = Relu6()
    x = jnp.array([-2.0, 0.0, 2.0, 6.0, 8.0])
    y = layer(x)
    assert tl.to_list(y) == [0.0, 0.0, 2.0, 6.0, 6.0]


def test_silu() -> None:
    layer = Silu()
    sigmoid = tl.Sigmoid()
    x = jnp.array([-4.0, 0.0, 4.0])
    y = layer(x)
    x_prime = jnp.array([-4.0, 4.0])
    y_prime = sigmoid(x_prime)
    y_prime_list = tl.to_list(y_prime)
    assert tl.to_list(y) == [-4.0 * y_prime_list[0], 0.0, 4.0 * y_prime_list[1]]


def test_softshrink() -> None:
    layer = SoftShrink()
    x = jnp.array([-2, -0.5, 0.0, 0.5, 2])
    y = layer(x)
    assert tl.to_list(y) == [-1.5, 0.0, 0.0, 0.0, 1.5]


def test_softsign() -> None:
    layer = SoftSign()
    x = jnp.array([-1.0, 0.0, 1.0])
    y = layer(x)
    assert tl.to_list(y) == [-0.5, 0.0, 0.5]


def test_tanhexp() -> None:
    layer = TanhExp()
    x = jnp.array([-2.0, 0.0, 2.0])
    y = layer(x)
    assert tl.to_list(y) == [
        -2.0 * jnp.tanh(jnp.exp(-2.0)),
        0.0,
        2.0 * jnp.tanh(jnp.exp(2.0)),
    ]


def test_tanhshrink() -> None:
    layer = TanhShrink()
    x = jnp.array([-2.0, 0.0, 2.0])
    y = layer(x)
    assert tl.to_list(y) == [-2.0 - jnp.tanh(-2.0), 0.0, 2.0 - jnp.tanh(2.0)]
