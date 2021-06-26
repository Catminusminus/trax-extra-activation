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
    x = jnp.array([0.0])
    y = layer(x)
    assert tl.to_list(y) == [0.0]
