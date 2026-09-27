#pragma once

#include "bit_helper.h"
#include "modes.h"

// The fixed-point format cast, one function per rounding mode, in the same
// shape as cast_binaryK.h and cast_superfp.h: each reads a FixedPointParamsT
// built once per tensor, and is a template on the carrier (bit_helper.h's
// FloatTraits).
//
// A fixed-point format holds the multiples of its step 2^-fl that a wl-bit
// integer holds, so it is a float that is subnormal everywhere: one fixed step
// from zero to the largest value. Each cast is therefore binaryK's subnormal
// arm (cast_binaryK.h) applied to the whole range. A value whose leading bit
// is 2^target_exp keeps exp_diff = target_exp + fl bits below it, which is
// binaryK's man_bits - (min_exp - target_exp) with the step in place of its
// smallest subnormal; a value under the step (exp_diff < 0) is zero or the
// step; and the result saturates to the ends of the range, which have no
// infinity beyond them. Only round-to-nearest-even is implemented so far: the
// other casts return their input unchanged.

// The constants the casts read, flat for the reason BinaryKParamsT is (a
// nested aggregate in the GEMM instantiations costs nvcc minutes).
template <class T>
struct FixedPointParamsT
{
    int fl;                  // exp_diff = target_exp + fl, the bits a value keeps
    int step_exponent_store; // the carrier's exponent field of the step 2^-fl
    T max_val;               // the largest value, which +inf and overflow saturate to
    T min_val;               // the smallest: 0 unsigned, -max_val symmetric, else -2^(mag_bits - fl)
};

// The carrier value whose word is `w`.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_word_value(typename FloatTraits<T>::word_t w)
{
    return reinterpret_cast<const MPTORCH_THREAD T &>(w);
}

// Builds the constants once per tensor. The two bounds are built on the word,
// not by float arithmetic, so they are exact on every backend: the largest
// value is mag_bits ones, a leading 1 at 2^top_exp and mag_bits - 1 ones
// below it, and the two's complement bottom is the power of two 2^min_exp.
// The number.py checks hold both inside the carrier's normals from below; a
// top past the carrier's largest value (a FormatRangeWarning) is an infinity,
// since the carrier cannot hold the real bound.
template <class T = float>
CUDA_HOST_DEVICE_INLINE FixedPointParamsT<T> make_fixedpoint_params(int wl, int fl, bool is_signed, bool symmetric)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    const int mag_bits = wl - (is_signed ? 1 : 0);
    const int top_exp = mag_bits - 1 - fl; // the largest value's leading bit
    const int min_exp = mag_bits - fl;     // the two's complement bottom is -2^min_exp

    FixedPointParamsT<T> p;
    p.fl = fl;
    p.step_exponent_store = F::BIAS - fl;

    if (top_exp > F::MAX_EXP)
        p.max_val = fixedpoint_word_value<T>(F::INF_BITS);
    else
    {
        const word_t exponent = (word_t)(top_exp + F::BIAS) << F::MAN_BITS;
        const word_t ones = (((word_t)1 << (mag_bits - 1)) - 1) << (F::MAN_BITS - (mag_bits - 1));
        p.max_val = fixedpoint_word_value<T>(exponent | ones);
    }

    if (!is_signed)
        p.min_val = T(0);
    else if (symmetric)
        p.min_val = -p.max_val;
    else if (min_exp > F::MAX_EXP)
        p.min_val = fixedpoint_word_value<T>(F::SIGN_MASK | F::INF_BITS);
    else
        p.min_val = fixedpoint_word_value<T>(F::SIGN_MASK | ((word_t)(min_exp + F::BIAS) << F::MAN_BITS));

    return p;
}

// A rounded value past either end becomes that end. Rounding can carry a value
// one step past the top (1.875 to 2.0 in wl=4, fl=2), and a value far above
// the range is left unrounded, since its exp_diff reaches the carrier's
// mantissa, and lands here.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_saturate(T quantized, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    if (quantized > p.max_val)
        return p.max_val;
    if (quantized < p.min_val)
        return p.min_val;
    return quantized;
}

// A NaN or an infinity, the words with an all-ones exponent field: a NaN (a
// nonzero mantissa) passes through, and an infinity saturates to the end of
// the range on its side, since a fixed-point word has no code for one.
template <class T>
CUDA_HOST_DEVICE_INLINE T fixedpoint_nonfinite(T x, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;
    const word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    if ((target & F::MAN_MASK) != 0)
        return x;
    return (target & F::SIGN_MASK) ? p.min_val : p.max_val;
}

// Rounds to nearest, ties to the even code. binaryK's subnormal arm with
// exp_diff = target_exp + fl, then the saturation to the range.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_nearest_even(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    using F = FloatTraits<T>;
    using word_t = typename F::word_t;

    // an unsigned format has no negative values
    if (MPTORCH_IS_NEGATIVE(x) && !is_signed)
        return T(0);

    word_t target = reinterpret_cast<const MPTORCH_THREAD word_t &>(x);
    int target_exp = (int)((target >> F::MAN_BITS) & F::FIELD_MASK) - F::BIAS;

    if (target_exp == F::INF_EXP)
        return fixedpoint_nonfinite(x, p);

    // bits kept below the leading one, down to the step; negative under the step
    int exp_diff = target_exp + p.fl;
    // zero unless the value is at least the step, or above half of it (half is
    // the tie, and zero the even code)
    int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << (F::EXP_BITS + 1)) > 0));

    // rounded only when it is not zero
    word_t quantize_bits = not_uflow ? round_bitwise_nearest_even(target, exp_diff) : word_t(0);
    // lifts a value rounded up to the step onto it, and makes a rounded zero +0.0
    quantize_bits = clip_subnormal_range_exponent(target, quantize_bits, p.step_exponent_store);
    T quantized = reinterpret_cast<const MPTORCH_THREAD T &>(quantize_bits);

    return fixedpoint_saturate(quantized, p);
}

// PLACEHOLDER: returns its input unchanged until it is written.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_nearest_away(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    return x;
}

// PLACEHOLDER: returns its input unchanged until it is written.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_up(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    return x;
}

// PLACEHOLDER: returns its input unchanged until it is written.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_down(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    return x;
}

// PLACEHOLDER: returns its input unchanged until it is written.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_zero(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    return x;
}

// PLACEHOLDER: returns its input unchanged until it is written.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_odd(T x, bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    return x;
}

// PLACEHOLDER: returns its input unchanged until it is written.
template <class T>
CUDA_HOST_DEVICE_INLINE T cast_fixedpoint_stochastic(T x, typename FloatTraits<T>::word_t rand_prob, int prng_bits,
                                                     bool is_signed, const MPTORCH_THREAD FixedPointParamsT<T> &p)
{
    return x;
}
