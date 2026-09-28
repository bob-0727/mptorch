"""
fixedpoint_quantize on a format small enough to derive every expected value by hand.

A fixed-point format holds the multiples of the step ``2**-fl`` that a ``wl``-bit
integer holds (see mptorch/csrc/common/cast_fixedpoint.h). The reference format
here is signed, ``wl=4, fl=2``: sixteen codes, -2 .. 1.75 in steps of 0.25. That
is few enough values that each expectation table below is written out from the
rounding mode's definition rather than from another implementation, which is
what makes it independent of the kernel. The interior ties on the parity of the
integer code, below the step the two candidates are zero and the step itself,
and past either end the result saturates. The round-to-nearest-even tables
come first; the other deterministic modes follow, one row per mode.

Every table is checked a second way by a sweep against an exact reference, the
rounding done in Python's ``fractions`` over every float16 value and random
float32 and float64 ones, which plays the part gfloat plays for binaryK.
Stochastic rounding is not implemented yet, and the CPU is the only backend.
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

# The deterministic modes, all implemented.
ALL_MODES = [RoundMode.RNE, RoundMode.RNA, RoundMode.RU, RoundMode.RD, RoundMode.RZ, RoundMode.RO]


def _quantize(x_val, device, dtype=torch.float32, cfg=CFG, mode=RoundMode.RNE, **kw):
    """One value through ``fixedpoint_quantize``, as a Python float."""
    x = torch.tensor([x_val], dtype=dtype, device=device)
    return fixedpoint_quantize(x, rounding_mode=mode, **cfg, **kw).item()


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
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("value,expected", ENDS_CASES)
def test_ends_saturate(device, mode, value, expected):
    """Every mode saturates the same way: each rounds these values onto one end
    or past it, and past it is the end."""
    assert _same(_quantize(value, device, mode=mode), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_symmetric_drops_the_bottom_code(device, mode):
    """``symmetric`` drops code -8, so the bottom is -1.75 like the top."""
    for value in (-2.0, -1.9, -100.0, float("-inf")):
        assert _same(_quantize(value, device, mode=mode, symmetric=True), -MAX)
    assert _same(_quantize(-1.75, device, mode=mode, symmetric=True), -MAX)
    assert _same(_quantize(1.9, device, mode=mode, symmetric=True), MAX)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_unsigned(device, mode):
    """Unsigned, the four bits are all magnitude: 0 .. 3.75, and every negative
    value is zero in every mode."""
    kw = dict(mode=mode, is_signed=False)
    assert _same(_quantize(3.25, device, **kw), 3.25)
    assert _same(_quantize(3.8, device, **kw), 3.75)
    assert _same(_quantize(100.0, device, **kw), 3.75)
    for value in (-0.3, -2.0, -0.0, -1e-30, float("-inf")):
        assert _same(_quantize(value, device, **kw), 0.0)
    assert _same(_quantize(float("inf"), device, **kw), 3.75)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
def test_nonfinite_inputs(device, mode):
    """A NaN passes through; an infinity saturates, since a fixed-point word has
    no code for one."""
    assert math.isnan(_quantize(float("nan"), device, mode=mode))
    assert _same(_quantize(float("inf"), device, mode=mode), MAX)
    assert _same(_quantize(float("-inf"), device, mode=mode), MIN)


# Signalling and quiet NaNs of both signs, with small, full and mixed payloads,
# as in tests/test_binaryk_p3109.py. The float64 words add payloads in the low
# 32 bits, which a cast that went through float32 would lose. float16 and
# bfloat16 are left out: storing into them changes a NaN's payload by itself.
NANS32 = torch.tensor(
    [0x7F800001, 0xFF800001, 0x7FC00000, 0xFFC00000, 0x7FFFFFFF, 0xFF812345], dtype=torch.int64
).to(torch.int32)
NANS64 = torch.tensor(
    [
        0x7FF0000000000001,
        0x7FF8000000000000,
        0x7FFFFFFFFFFFFFFF,
        0x7FF0000100000000,
        0x7FF4000000012345,
        0x7FF8000000000001,
    ],
    dtype=torch.int64,
)
NANS64 = torch.cat([NANS64, NANS64 | torch.tensor(-(2**63), dtype=torch.int64)])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("is_signed", [True, False], ids=["signed", "unsigned"])
@pytest.mark.parametrize(
    "nans,wl,fl", [(NANS32, 8, 4), (NANS32, 4, 2), (NANS64, 8, 4), (NANS64, 40, 30)]
)
def test_nan_passes_through_whole(device, mode, is_signed, nans, wl, fl):
    """A NaN comes back bit for bit: payload, signalling bit and sign.

    The directed modes are where that is not free: a NaN compares false against
    zero, so it takes the arm that works on a magnitude and negates it back, and
    a float negation may canonicalize the payload. An unsigned format turns a
    negative input into zero, but a NaN with its sign bit set is unordered, not
    negative, so it passes through too."""
    x = nans.view(torch.float32 if nans.dtype is torch.int32 else torch.float64).to(device)
    got = fixedpoint_quantize(x, wl, fl, is_signed=is_signed, rounding_mode=mode)
    assert torch.equal(got.cpu().view(nans.dtype), nans)


# ---------------------------------------------------------------------------
# The other deterministic modes, in steps of 0.25: 0.3 is 1.2 steps, 0.375 and
# 0.625 are the ties 1.5 and 2.5, and below the step 0.1 is 0.4 steps and
# 0.125 the tie 0.5. RNA breaks a tie away from zero; RU, RD and RZ take one
# neighbour whatever the distance; RO takes the one with an odd code whenever
# the value is inexact, which under the step is always the step itself (code
# 1). A zero result is +0.0 in every mode.
OTHER_MODE_CASES = [
    # (value, mode, expected)
    (0.3, RoundMode.RNA, 0.25),
    (0.375, RoundMode.RNA, 0.5),  # tie -> away from zero
    (0.625, RoundMode.RNA, 0.75),  # tie -> away from zero, not to even
    (-0.375, RoundMode.RNA, -0.5),
    (0.1, RoundMode.RNA, 0.0),
    (0.125, RoundMode.RNA, 0.25),  # half a step: tie -> the step, not zero
    (-0.125, RoundMode.RNA, -0.25),
    (0.3, RoundMode.RU, 0.5),
    (0.375, RoundMode.RU, 0.5),
    (0.625, RoundMode.RU, 0.75),
    (-0.3, RoundMode.RU, -0.25),  # up is toward zero for a negative value
    (-0.375, RoundMode.RU, -0.25),
    (0.1, RoundMode.RU, 0.25),
    (1e-30, RoundMode.RU, 0.25),  # any positive value leaves zero
    (-0.1, RoundMode.RU, 0.0),  # +0.0, not -0.0
    (-1e-30, RoundMode.RU, 0.0),
    (0.3, RoundMode.RD, 0.25),
    (0.375, RoundMode.RD, 0.25),
    (0.625, RoundMode.RD, 0.5),
    (-0.3, RoundMode.RD, -0.5),  # down is away from zero for a negative value
    (-0.375, RoundMode.RD, -0.5),
    (0.1, RoundMode.RD, 0.0),
    (-0.1, RoundMode.RD, -0.25),
    (-1e-30, RoundMode.RD, -0.25),
    (0.3, RoundMode.RZ, 0.25),
    (0.625, RoundMode.RZ, 0.5),
    (-0.3, RoundMode.RZ, -0.25),
    (-0.625, RoundMode.RZ, -0.5),
    (0.2, RoundMode.RZ, 0.0),
    (-0.2, RoundMode.RZ, 0.0),  # +0.0, not -0.0
    (0.25, RoundMode.RO, 0.25),  # exact: left alone
    (0.5, RoundMode.RO, 0.5),  # exact, even code 2: still left alone
    (0.3, RoundMode.RO, 0.25),  # 1.2 steps: code 1 is odd
    (0.375, RoundMode.RO, 0.25),  # 1.5 steps: code 1 is odd
    (0.625, RoundMode.RO, 0.75),  # 2.5 steps: code 3 is odd
    (-0.3, RoundMode.RO, -0.25),
    (0.1, RoundMode.RO, 0.25),  # under the step: the step, code 1
    (1e-30, RoundMode.RO, 0.25),
    (-0.1, RoundMode.RO, -0.25),
    (0.0, RoundMode.RO, 0.0),  # zero is exact
    (-0.0, RoundMode.RO, 0.0),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value,mode,expected", OTHER_MODE_CASES)
def test_other_modes(device, value, mode, expected):
    assert _same(_quantize(value, device, mode=mode), expected)


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


def _reference(
    x: float, wl: int, fl: int, is_signed: bool, symmetric: bool, mode: RoundMode
) -> float:
    """``x`` rounded onto the fixed-point grid in ``mode``, then saturated."""
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
    # the code below the value, and how far past it the value is, in [0, 1)
    code, rest = divmod(Fraction(x) / step, 1)
    half = Fraction(1, 2)
    if rest == 0:
        up = False  # on the grid: every mode leaves it alone
    elif mode is RoundMode.RNE:
        up = rest > half or (rest == half and code % 2 == 1)
    elif mode is RoundMode.RNA:
        up = rest > half or (rest == half and x > 0)
    elif mode is RoundMode.RU:
        up = True
    elif mode is RoundMode.RD:
        up = False
    elif mode is RoundMode.RZ:
        up = x < 0
    elif mode is RoundMode.RO:
        up = code % 2 == 0
    else:
        raise ValueError(f"no reference for {mode}")
    code += up
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
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("wl,fl,is_signed,symmetric", FORMATS)
def test_against_the_exact_reference(device, mode, dtype, wl, fl, is_signed, symmetric):
    """Every float16 value (in float16) or 5000 random ones (in the other dtypes)
    rounds as the reference rounds it, word for word, signed zeros included."""
    values = _float16_values() if dtype is torch.float16 else _random_values(wl * 100 + fl)
    x = torch.tensor(values, dtype=torch.float64).to(dtype).to(device)
    got = fixedpoint_quantize(
        x, wl, fl, is_signed=is_signed, symmetric=symmetric, rounding_mode=mode
    )
    want = torch.tensor(
        [_reference(v, wl, fl, is_signed, symmetric, mode) for v in x.double().tolist()],
        dtype=torch.float64,
    )
    want = want.to(torch.float32).to(dtype) if dtype is not torch.float64 else want
    same = (got.cpu() == want) & (torch.signbit(got.cpu()) == torch.signbit(want))
    same |= torch.isnan(got.cpu()) & torch.isnan(want)
    assert bool(same.all()), f"first mismatch at x={x[~same][0].item()!r}"


# Formats only binary64 carries: more than binary32's 24 bits of precision, a
# step below its floor, or a top past its largest value. Each is inside
# binary64's own bounds (53 bits, a step no finer than 2**-1021, a top no
# higher than 2**1023).
WIDE_FORMATS = [
    # (wl, fl, is_signed, symmetric)
    (40, 30, True, False),  # 39 magnitude bits
    (40, 30, True, True),
    (54, 20, True, False),  # 53 magnitude bits: all of binary64's precision
    (53, 0, False, False),  # 53 magnitude bits, unsigned
    (30, 1000, True, False),  # a step of 2**-1000, a top of 2**-971
    (12, -900, True, False),  # a step of 2**900, a top of 2**910
]


def _wide_values(wl: int, fl: int, is_signed: bool) -> list[float]:
    """float64 values across a wide format: random significands from three
    binades under the step to one past the top, both signs, and the grid points
    and midpoints at both ends with a float64 ulp either side of each."""
    rng = random.Random(wl * 10_000 + fl)
    mag_bits = wl - (1 if is_signed else 0)
    step_exp, top_exp = -fl, mag_bits - 1 - fl
    values = [
        math.ldexp(1 + rng.random(), rng.randint(step_exp - 3, top_exp + 1)) for _ in range(4000)
    ]
    for code in (1, 2, 3, 2**mag_bits - 3, 2**mag_bits - 2, 2**mag_bits - 1):
        point = math.ldexp(code, step_exp)
        half = math.ldexp(2 * code + 1, step_exp - 1)  # rounded where it needs 54 bits
        for v in (point, half):
            values += [v, math.nextafter(v, math.inf), math.nextafter(v, 0.0)]
    return values + [-v for v in values]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("wl,fl,is_signed,symmetric", WIDE_FORMATS)
def test_formats_past_binary32(device, mode, wl, fl, is_signed, symmetric):
    """A float64 tensor rounds in binary64 to formats binary32 cannot hold, as
    the reference does. Catches a cast constant, mask or bound still sized for
    binary32 (a 32-bit word, a 23-bit mantissa) in the ``double`` instantiation."""
    x = torch.tensor(_wide_values(wl, fl, is_signed), dtype=torch.float64, device=device)
    got = fixedpoint_quantize(
        x, wl, fl, is_signed=is_signed, symmetric=symmetric, rounding_mode=mode
    ).cpu()
    want = torch.tensor(
        [_reference(v, wl, fl, is_signed, symmetric, mode) for v in x.tolist()],
        dtype=torch.float64,
    )
    same = (got == want) & (torch.signbit(got) == torch.signbit(want))
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
@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_in_place_matches_out_of_place(device, mode, dtype):
    x = torch.tensor(_random_values(11), dtype=torch.float64).to(dtype).to(device)
    expected = fixedpoint_quantize(x, 8, 4, rounding_mode=mode)
    y = x.clone()
    assert fixedpoint_quantize_(y, 8, 4, rounding_mode=mode) is y
    assert torch.equal(y.nan_to_num(0.5), expected.nan_to_num(0.5))


@pytest.mark.parametrize("device", DEVICES)
def test_a_strided_input_rounds_like_its_contiguous_copy(device):
    x = torch.tensor(_random_values(13)[:1000], device=device).reshape(40, 25).t()
    got = fixedpoint_quantize(x, 8, 4)
    assert torch.equal(
        got.nan_to_num(0.5), fixedpoint_quantize(x.contiguous(), 8, 4).nan_to_num(0.5)
    )
