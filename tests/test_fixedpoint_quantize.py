"""
fixedpoint_quantize on a format small enough to derive every expected value by hand.

A fixed-point format holds the multiples of the step ``2**-fl`` that a ``wl``-bit
integer holds (see mptorch/csrc/common/cast_fixedpoint.h). The reference format
here is signed, ``wl=4, fl=2``: sixteen codes, -2 .. 1.75 in steps of 0.25. That
is few enough values that each expectation table below is written out from the
rounding mode's definition rather than from another implementation, which is
what makes it independent of the kernel. The interior ties on the parity of the
integer code, below the step the two candidates are zero and the step itself,
and past either end the result saturates.

Every table is checked a second way by a sweep against an exact reference, the
rounding done in Python's ``fractions`` over every float16 value and random
float32 and float64 ones, which plays the part gfloat plays for binaryK. Only
round-to-nearest-even is implemented so far, so only it is tested; the CPU is
the only backend.
"""

import math
import random
from fractions import Fraction

import pytest
import torch

from mptorch.number import RoundMode
from mptorch.quant import fixedpoint_quantize, fixedpoint_quantize_

# The CPU is the only backend with a kernel so far; `available_devices` from
# tests.markers once the CUDA and MPS kernels exist.
DEVICES = ["cpu"]

# The reference format: 1 sign bit and 3 magnitude bits, 2 of them fractional.
# Codes -8 .. 7 hold -2 .. 1.75 in steps of 0.25.
CFG = {"wl": 4, "fl": 2}
STEP = 0.25
MAX = 1.75
MIN = -2.0


def _quantize(x_val, device, dtype=torch.float32, cfg=CFG, **kw):
    """One value through ``fixedpoint_quantize`` under RNE, as a Python float."""
    x = torch.tensor([x_val], dtype=dtype, device=device)
    return fixedpoint_quantize(x, rounding_mode=RoundMode.RNE, **cfg, **kw).item()


def _same(got: float, want: float) -> bool:
    """Equal, and for a zero equally signed: `-0.0 == 0.0` is true, and a
    fixed-point format's only zero is +0.0."""
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


# ---------------------------------------------------------------------------
# The interior: two neighbouring codes, and a tie between them goes to the even
# code. 0.375 is 1.5 steps, between codes 1 and 2, so it goes to 2 (0.5); 0.625
# is 2.5 steps and also goes to 2; 0.875 is 3.5 steps and goes to 4 (1.0).
INTERIOR_CASES = [
    # (value, expected)
    (0.25, 0.25),  # a grid point is left alone
    (0.3, 0.25),
    (0.4, 0.5),
    (0.375, 0.5),  # tie -> even code (2)
    (0.625, 0.5),  # tie -> even code (2)
    (0.875, 1.0),  # tie -> even code (4)
    (1.1, 1.0),
    (-0.3, -0.25),
    (-0.375, -0.5),  # tie -> even code (-2)
    (-0.625, -0.5),
    (-0.875, -1.0),
    (-1.1, -1.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,expected", INTERIOR_CASES)
def test_interior_rounding(device, value, expected):
    assert _same(_quantize(value, device), expected)


# ---------------------------------------------------------------------------
# Below the step: the candidates are zero and the step. Half a step is a tie,
# and zero is the even code, so it takes it; anything above half goes up. The
# one zero is unsigned, so a negative value that rounds to it is +0.0.
BELOW_STEP_CASES = [
    (0.0, 0.0),
    (-0.0, 0.0),  # no -0.0 out
    (0.01, 0.0),
    (0.1, 0.0),
    (0.125, 0.0),  # half a step: tie -> zero, the even code
    (0.13, 0.25),
    (0.2, 0.25),
    (-0.1, 0.0),  # rounds to zero: +0.0, not -0.0
    (-0.125, 0.0),
    (-0.13, -0.25),
    (1e-30, 0.0),  # far below the step
    (-1e-30, 0.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,expected", BELOW_STEP_CASES)
def test_below_the_step(device, value, expected):
    assert _same(_quantize(value, device), expected)


# ---------------------------------------------------------------------------
# The ends: a result past the range saturates to its end, whether the input was
# already past it or rounding carried it there. 1.875 is the tie between 1.75
# (code 7) and 2.0 (code 8, past the top), and goes to the even code 8, which
# saturates back to 1.75. The range is two's complement, so the bottom, -2, is
# one step further from zero than the top.
ENDS_CASES = [
    (1.75, MAX),
    (1.8, MAX),
    (1.875, MAX),  # tie -> code 8, past the top -> saturates
    (1.9, MAX),
    (100.0, MAX),
    (1e30, MAX),
    (-2.0, MIN),
    (-2.1, MIN),
    (-2.125, MIN),  # tie -> code -8, the bottom itself
    (-2.2, MIN),
    (-100.0, MIN),
    (-1e30, MIN),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,expected", ENDS_CASES)
def test_ends_saturate(device, value, expected):
    assert _same(_quantize(value, device), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_symmetric_drops_the_bottom_code(device):
    """``symmetric`` drops code -8, so the bottom is -1.75 like the top."""
    for value in (-2.0, -1.9, -100.0, float("-inf")):
        assert _same(_quantize(value, device, symmetric=True), -MAX)
    assert _same(_quantize(-1.75, device, symmetric=True), -MAX)
    assert _same(_quantize(1.9, device, symmetric=True), MAX)


@pytest.mark.parametrize("device", DEVICES)
def test_unsigned(device):
    """Unsigned, the four bits are all magnitude: 0 .. 3.75, and every negative
    value is zero."""
    kw = dict(is_signed=False)
    assert _same(_quantize(3.3, device, **kw), 3.25)
    assert _same(_quantize(3.8, device, **kw), 3.75)
    assert _same(_quantize(100.0, device, **kw), 3.75)
    for value in (-0.3, -2.0, -0.0, float("-inf")):
        assert _same(_quantize(value, device, **kw), 0.0)
    assert _same(_quantize(float("inf"), device, **kw), 3.75)


@pytest.mark.parametrize("device", DEVICES)
def test_nonfinite_inputs(device):
    """A NaN passes through; an infinity saturates, since a fixed-point word has
    no code for one."""
    assert math.isnan(_quantize(float("nan"), device))
    assert _same(_quantize(float("inf"), device), MAX)
    assert _same(_quantize(float("-inf"), device), MIN)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "cfg,value,expected",
    [
        ({"wl": 4, "fl": -2}, 13.0, 12.0),  # a step of 4
        ({"wl": 4, "fl": -2}, 14.0, 16.0),  # tie -> even code (4)
        ({"wl": 4, "fl": -2}, 40.0, 28.0),  # saturates at 7 * 4
        ({"wl": 4, "fl": 6}, 0.03, 0.03125),  # pure fractions: steps of 1/64
        ({"wl": 4, "fl": 6}, 0.2, 0.109375),  # saturates at 7/64
        ({"wl": 8, "fl": 0}, 2.5, 2.0),  # integers: tie -> even
        ({"wl": 8, "fl": 0}, 3.5, 4.0),
        ({"wl": 8, "fl": 0}, -128.4, -128.0),
    ],
)
def test_other_steps(device, cfg, value, expected):
    """A step coarser than 1, a format of pure fractions, and plain integers."""
    assert _same(_quantize(value, device, cfg=cfg), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_docstring_example(device):
    x = torch.tensor([1.1, 3.3, -9.0, 0.03], device=device)
    assert fixedpoint_quantize(x, 8, 4).tolist() == [1.125, 3.3125, -8.0, 0.0]


# ---------------------------------------------------------------------------
# The exact reference: the same rounding in Python's fractions, where nothing is
# rounded but the rule itself. The kernel's result is stored in the tensor's
# dtype, so the reference's is too: through float32, where every value of these
# formats is exact, and then once into the dtype.


def _reference(x: float, wl: int, fl: int, is_signed: bool, symmetric: bool) -> float:
    """``x`` rounded to nearest, ties to the even code, onto the fixed-point grid."""
    if math.isnan(x):
        return x
    mag_bits = wl - (1 if is_signed else 0)
    step = Fraction(2) ** -fl
    top = (2**mag_bits - 1) * step
    bottom = Fraction(0) if not is_signed else (-top if symmetric else -(2**mag_bits) * step)
    if math.isinf(x):
        return float(top if x > 0 else bottom)
    if x < 0 and not is_signed:
        return 0.0
    code, rest = divmod(Fraction(x) / step, 1)
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and code % 2):
        code += 1
    return float(min(max(code * step, bottom), top)) + 0.0  # + 0.0: no -0.0


def _float16_values() -> list[float]:
    words = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    return words.view(torch.float16).double().tolist()


def _random_values(seed: int) -> list[float]:
    """Values across 60 binades, both signs, and the inputs every cast must handle."""
    rng = random.Random(seed)
    values = [rng.gauss(0, 1) * 2.0 ** rng.randint(-30, 30) for _ in range(5000)]
    return values + [0.0, -0.0, math.inf, -math.inf, math.nan, 1e-30, -1e-30, 1e30, -1e30]


FORMATS = [
    # (wl, fl, is_signed, symmetric)
    (4, 2, True, False),
    (4, 2, True, True),
    (4, 2, False, False),
    (8, 4, True, False),
    (8, -3, True, False),
    (8, 10, True, False),
    (16, 8, True, False),
    (24, 10, True, False),  # 23 magnitude bits
    (24, 0, False, False),  # 24 magnitude bits: all of binary32's precision
    (2, 0, True, False),  # one magnitude bit
    (1, 3, False, False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("wl,fl,is_signed,symmetric", FORMATS)
def test_against_the_exact_reference(device, dtype, wl, fl, is_signed, symmetric):
    """Every float16 value (in float16) or 5000 random ones (in the other dtypes)
    rounds as the reference rounds it, word for word, signed zeros included."""
    values = _float16_values() if dtype is torch.float16 else _random_values(wl * 100 + fl)
    x = torch.tensor(values, dtype=torch.float64).to(dtype).to(device)
    got = fixedpoint_quantize(x, wl, fl, is_signed=is_signed, symmetric=symmetric)
    want = torch.tensor(
        [_reference(v, wl, fl, is_signed, symmetric) for v in x.double().tolist()],
        dtype=torch.float64,
    )
    want = want.to(torch.float32).to(dtype) if dtype is not torch.float64 else want
    same = (got.cpu() == want) & (torch.signbit(got.cpu()) == torch.signbit(want))
    same |= torch.isnan(got.cpu()) & torch.isnan(want)
    assert bool(same.all()), f"first mismatch at x={x[~same][0].item()!r}"


# ---------------------------------------------------------------------------
# The call's other spellings: the binary64 carrier and the in-place op round
# exactly as the plain call does.


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_binary64_carrier_matches_widening_by_hand(device, dtype):
    x = torch.tensor(_random_values(7), dtype=torch.float64).to(dtype).to(device)
    got = fixedpoint_quantize(x, 16, 8, carrier=torch.float64)
    by_hand = fixedpoint_quantize(x.double(), 16, 8).to(torch.float32).to(dtype)
    assert torch.equal(got.nan_to_num(0.5), by_hand.nan_to_num(0.5))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_in_place_matches_out_of_place(device, dtype):
    x = torch.tensor(_random_values(11), dtype=torch.float64).to(dtype).to(device)
    expected = fixedpoint_quantize(x, 8, 4)
    y = x.clone()
    assert fixedpoint_quantize_(y, 8, 4) is y
    assert torch.equal(y.nan_to_num(0.5), expected.nan_to_num(0.5))


@pytest.mark.parametrize("device", DEVICES)
def test_a_strided_input_rounds_like_its_contiguous_copy(device):
    x = torch.tensor(_random_values(13)[:1000], device=device).reshape(40, 25).t()
    got = fixedpoint_quantize(x, 8, 4)
    assert torch.equal(
        got.nan_to_num(0.5), fixedpoint_quantize(x.contiguous(), 8, 4).nan_to_num(0.5)
    )
