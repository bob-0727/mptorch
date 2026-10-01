"""The fixed-point format object and its carrier and storage checks.

This module guards ``FixedPoint`` and the fixed-point checks of ``mptorch.number``,
calling the checks directly. The float formats' equivalents live in
``tests/test_number_formats.py`` (the format objects) and
``tests/test_format_limits.py`` (the checks), where every boundary is held
through a real call, ``Quant(fmt)(tensor)``, on every backend. The fixed-point
quantizer has a CPU kernel only so far, so its cases stay here until the CUDA
and MPS kernels exist; then they move into those files' tests, by topic, beside
the binaryK and superfp cases, leaving only what is fixed-point specific.

A fixed-point format has no exponent field: its range is its precision, the
``mag_bits`` of its largest value, that value's leading bit, and its step
``2**-fl``. The boundaries below are derived from the format's value set, not
measured against a kernel as the float formats' are. The storage rule is
already held against what storing does, over every float16 and bfloat16 value
(``test_the_fixedpoint_storage_rule_is_what_storing_does``).
"""

import itertools
import warnings

import pytest
import torch

from mptorch import FixedPoint
from mptorch.number import (
    FormatRangeWarning,
    _fixedpoint_storage_findings,
    check_fixedpoint,
    check_fixedpoint_carrier,
    check_fixedpoint_storage,
)
from mptorch.quant import Palette

F32, F64 = torch.float32, torch.float64
F16, BF16 = torch.float16, torch.bfloat16


def _silent(call):
    """Run ``call`` with ``FormatRangeWarning`` promoted to an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


def _fxp(*fmt, **kw):
    """Return a thunk that holds a fixed-point format against a carrier."""
    return lambda: check_fixedpoint(*fmt, **kw)


def _fxp_store(dtype, *fmt, **kw):
    """Return a thunk that holds a fixed-point quantizer's result against ``dtype``."""
    return lambda: check_fixedpoint_storage(*fmt, storage=dtype, elementwise=True, **kw)


# ------------------------------------------------------------------------------------
# The format object.


@pytest.mark.parametrize(
    "call",
    [
        lambda: FixedPoint(1, 0),
        lambda: FixedPoint(0, 0, is_signed=False),
        lambda: FixedPoint(8, 4, is_signed=False, symmetric=True),
        lambda: FixedPoint(8, 4, prng_bits=-1),
    ],
)
def test_fixedpoint_validates_at_construction(call):
    """A malformed format (no magnitude bits, a symmetric unsigned format,
    negative prng bits) raises when built, not at the first call."""
    with pytest.raises(ValueError):
        call()


def test_fixedpoint_is_frozen_and_hashable():
    """Equal formats hash equal and cannot be mutated, as for the float formats."""
    fmt = FixedPoint(8, 4)
    assert hash(fmt) == hash(FixedPoint(8, 4)) and fmt == FixedPoint(8, 4)
    with pytest.raises(AttributeError):
        fmt.wl = 9  # ty: ignore[invalid-assignment]


def test_fixedpoint_fields_after_fl_are_keyword_only():
    """``FixedPoint(8, 4, False)`` would read as unsigned or as symmetric depending
    on the field order, so everything after ``fl`` is keyword-only, as for
    ``BinaryK``'s fields after ``P``."""
    with pytest.raises(TypeError):
        FixedPoint(8, 4, False)  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize(
    ("fmt", "mag_bits", "step", "min_value", "max_value"),
    [
        (FixedPoint(8, 4), 7, 2.0**-4, -8.0, 7.9375),
        (FixedPoint(8, 4, symmetric=True), 7, 2.0**-4, -7.9375, 7.9375),
        (FixedPoint(8, 4, is_signed=False), 8, 2.0**-4, 0.0, 15.9375),
        (FixedPoint(4, -2), 3, 4.0, -32.0, 28.0),  # a step coarser than 1
        (FixedPoint(4, 6), 3, 2.0**-6, -0.125, 0.109375),  # pure fractions
    ],
)
def test_fixedpoint_range(fmt, mag_bits, step, min_value, max_value):
    """The range is the multiples of ``2**-fl`` a ``wl``-bit integer holds: two's
    complement signed, one code short of it symmetric, from zero unsigned."""
    assert (fmt.mag_bits, fmt.step) == (mag_bits, step)
    assert (fmt.min_value, fmt.max_value) == (min_value, max_value)


def test_palette_rejects_a_family_with_no_gemm():
    """A fixed-point format has an elementwise quantizer and no GEMM kernel yet."""
    with pytest.raises(TypeError, match="no GEMM kernel takes FixedPoint"):
        Palette([FixedPoint(8, 4)])


# ------------------------------------------------------------------------------------
# What each carrier can hold.


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        # precision: stochastic rounding keeps every magnitude bit in binary32's
        # 23 mantissa bits, as binaryK keeps its man_bits; a signed word has one more
        (_fxp(24, 0), _fxp(25, 0), "needs 24 bits of precision; binary32 has 23"),
        (_fxp(23, 0, is_signed=False), _fxp(24, 0, is_signed=False), "needs 24 bits"),
        (_fxp(53, 0, carrier=F64), _fxp(54, 0, carrier=F64), "needs 53 bits.*binary64 has 52"),
        # stochastic rounding draws its bits below the step, in the same mantissa
        (_fxp(16, 4, prng_bits=8), _fxp(17, 4, prng_bits=8), r"16 \+ 8 \(prng_bits\)"),
        # stochastic rounding's shift, one binade above the top, is a carrier value
        (_fxp(8, -120), _fxp(8, -121), "top binade, 2\\^127"),
        (_fxp(8, -1016, carrier=F64), _fxp(8, -1017, carrier=F64), "top binade, 2\\^1023"),
    ],
)
def test_fixedpoint_carrier_boundaries_raise(ok, bad, match):
    """Each error is one step past a format the carrier holds silently."""
    _silent(ok)
    with pytest.raises(ValueError, match=match):
        bad()


def test_a_fixedpoint_below_the_carriers_normals_raises():
    """A largest value whose leading bit is below 2**-126 has nothing the cast can
    place. One step up it works, below the floor, so it warns rather than raises."""
    with pytest.warns(FormatRangeWarning, match="step"):
        check_fixedpoint(8, 132)
    with pytest.raises(ValueError, match="below binary32's smallest normal"):
        check_fixedpoint(8, 133)


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        # a step below the floor the cast places values by, 2**-125
        (_fxp(8, 125), _fxp(8, 126), r"step, 2\^-126, is below 2\^-125"),
        (_fxp(8, 1021, carrier=F64), _fxp(8, 1022, carrier=F64), r"below 2\^-1021"),
    ],
)
def test_fixedpoint_carrier_boundaries_warn(ok, bad, match):
    """A step below the carrier's floor quantizes partially, and warns. The top
    has no warning: a range reaching the carrier's top binade is an error, since
    stochastic rounding's shift would not be a carrier value."""
    _silent(ok)
    with pytest.warns(FormatRangeWarning, match=match):
        bad()


def test_a_fixedpoint_binary32_finding_points_at_binary64():
    """A format binary64 holds says so when binary32 does not, since the cure is
    the carrier rather than the format."""
    with pytest.raises(ValueError, match="carrier=torch.float64"):
        check_fixedpoint(26, 0)
    with pytest.raises(ValueError) as caught:
        check_fixedpoint(55, 0, carrier=F64)
    assert "carrier=torch.float64" not in str(caught.value)


@pytest.mark.parametrize(
    ("args", "kw", "match"),
    [
        ((1, 0), {}, "wl must be >= 2"),
        ((0, 0), dict(is_signed=False), "wl must be >= 1"),
        ((8, 4), dict(is_signed=False, symmetric=True), "cannot be symmetric"),
        ((8, 4), dict(prng_bits=-1), "prng_bits must be >= 0"),
    ],
)
def test_the_fixedpoint_checks_refuse_what_the_constructor_refuses(args, kw, match):
    """``fixedpoint_quantize`` takes plain integers and never builds a
    ``FixedPoint``, so its checks refuse the layouts the constructor does."""
    with pytest.raises(ValueError, match=match):
        check_fixedpoint(*args, **kw)


def test_building_a_fixedpoint_raises_only_what_binary64_cannot_do():
    """Construction holds the format to binary64 and warns about nothing."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        FixedPoint(40, 4)  # past binary32's precision
        FixedPoint(8, -122)  # past binary32's top
        FixedPoint(8, 126)  # below binary32's floor
    assert caught == []
    with pytest.raises(ValueError, match="binary64 has 52"):
        FixedPoint(60, 4)


def test_a_fixedpoint_carrier_warning_names_the_callers_line():
    """The per-call check names the caller's line, as the float formats' do."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        check_fixedpoint_carrier(8, 126, carrier=F32)  # a step below binary32's floor
    assert [c.filename for c in caught] == [__file__]


# ------------------------------------------------------------------------------------
# What a tensor narrower than its carrier can store.


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        # Edge 1, the top: 65504 is 2047 * 2**5, off the grid of a step of 64,
        # and rounds up out of float16's range. A step of 32 holds it.
        (_fxp_store(F16, 16, -5), _fxp_store(F16, 16, -6), "may round to infinity"),
        # ...and the same from the bottom: an asymmetric format ends at -65536
        # while its top, 65472, is inside float16's range.
        (
            _fxp_store(F16, 11, -6, symmetric=True),
            _fxp_store(F16, 11, -6),
            "may round to infinity",
        ),
        # Edge 2: a largest value with more bits than the dtype, which an input
        # reaches by saturating onto it: 2047 fits float16's 11 bits, 4095 does not.
        (_fxp_store(F16, 12, 0), _fxp_store(F16, 13, 0), "not a float16 value"),
        (_fxp_store(BF16, 9, 0), _fxp_store(BF16, 10, 0), "not a bfloat16 value"),
        (_fxp_store(F32, 25, 0), _fxp_store(F32, 26, 0), "not a float32 value"),
    ],
)
def test_fixedpoint_storage_edges_warn(ok, bad, match):
    """The two range edges where a fixed-point result leaves the dtype's grid warn."""
    _silent(ok)
    with pytest.warns(FormatRangeWarning, match=match):
        bad()


def test_a_fixedpoint_below_the_dtypes_smallest_value_raises():
    """A largest value below float16's 2**-24 leaves nothing but zero to store."""
    _silent(_fxp_store(F16, 2, 24))  # {-2**-23, -2**-24, 0, 2**-24}
    with pytest.raises(ValueError, match="below float16's smallest"):
        _fxp_store(F16, 2, 25)()


def test_there_is_no_fixedpoint_gemm_storage_rule_yet():
    """``elementwise=False`` is a GEMM's rule, and there is no fixed-point GEMM."""
    with pytest.raises(NotImplementedError, match="no fixed-point GEMM"):
        check_fixedpoint_storage(8, 4, storage=F16)


def _dtype_values(dtype):
    """Every finite value of a 16-bit dtype, exactly, as float64."""
    words = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    values = words.view(dtype).to(F64)
    return values[torch.isfinite(values)]


def _any_restored(values, wl, fl, is_signed, symmetric, dtype):
    """Whether storing in ``dtype`` changes any result of quantizing ``values``.

    Each value is rounded onto the grid in float64, where every value involved
    is exact, down, up and to nearest even (the directions a rounding takes),
    saturating at the ends, and the result is stored in the dtype and read back.
    """
    mag_bits = wl - (1 if is_signed else 0)
    max_code = 2**mag_bits - 1
    min_code = -max_code if symmetric else -(2**mag_bits) if is_signed else 0
    scaled = torch.ldexp(values, torch.tensor(float(fl), dtype=F64))
    for rounding in (torch.floor, torch.ceil, torch.round):
        codes = rounding(scaled).clamp(min_code, max_code)
        results = torch.ldexp(codes, torch.tensor(float(-fl), dtype=F64))
        if not torch.equal(results.to(dtype).to(F64), results):
            return True
    return False


@pytest.mark.parametrize("dtype", [F16, BF16])
@pytest.mark.parametrize("is_signed,symmetric", [(True, False), (True, True), (False, False)])
def test_the_fixedpoint_storage_rule_is_what_storing_does(dtype, is_signed, symmetric):
    """The storage rule warns exactly for the formats whose results storing changes.

    Every finite value of the dtype is quantized, exactly, to each format over a
    grid of widths and steps, and stored: the rule must warn where some result
    comes back changed and stay silent where none does. Formats with nothing to
    store are the error's, and left out.
    """
    values = _dtype_values(dtype)
    wrong = []
    for wl, fl in itertools.product(range(2, 20), range(-20, 30)):
        error, warning = _fixedpoint_storage_findings(wl, fl, is_signed, symmetric, dtype, True)
        if error is not None:
            continue
        restored = _any_restored(values, wl, fl, is_signed, symmetric, dtype)
        if restored != (warning is not None):
            wrong.append((wl, fl, restored, warning))
    assert wrong == []
