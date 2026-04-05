#!/usr/bin/env python3
"""
Monte Carlo Error Analysis for Approximate Multipliers (AxMs)
=============================================================
Scans all .c multiplier implementations under this directory, compiles each
one via gcc into a temporary shared library, runs a simulation over the full
input space (8-bit) or a large random sample (wider bit-widths), and
computes the error distribution statistics:

    mu      : mean error  (bias)
    sigma^2 : variance of error
    sigma   : standard deviation of error

Results are written to  multiplier_error_stats.csv  in this directory.

Usage:
    python3 axm_monte_carlo.py [--samples N]

    --samples N   Number of Monte Carlo samples for bit-widths > 8.
                  Default: 100000.  8-bit multipliers always use the full
                  exhaustive input space (65536 pairs).
"""

import os
import re
import sys
import csv
import ctypes
import tempfile
import argparse
import subprocess
from pathlib import Path

import numpy as np

VSRC_DIR   = Path(__file__).parent
OUTPUT_CSV = VSRC_DIR / "multiplier_error_stats.csv"
DEFAULT_N  = 100_000
SEED       = 42

# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

CTYPE_MAP = {
    "int8_t"   : ctypes.c_int8,
    "int16_t"  : ctypes.c_int16,
    "int32_t"  : ctypes.c_int32,
    "int64_t"  : ctypes.c_int64,
    "uint8_t"  : ctypes.c_uint8,
    "uint16_t" : ctypes.c_uint16,
    "uint32_t" : ctypes.c_uint32,
    "uint64_t" : ctypes.c_uint64,
}

NUMPY_MAP = {
    "int8_t"   : np.int8,
    "int16_t"  : np.int16,
    "int32_t"  : np.int32,
    "int64_t"  : np.int64,
    "uint8_t"  : np.uint8,
    "uint16_t" : np.uint16,
    "uint32_t" : np.uint32,
    "uint64_t" : np.uint64,
}

# Width of each C type in bits
TYPE_BITS = {
    "int8_t"   : 8,   "uint8_t"  : 8,
    "int16_t"  : 16,  "uint16_t" : 16,
    "int32_t"  : 32,  "uint32_t" : 32,
    "int64_t"  : 64,  "uint64_t" : 64,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_c_signature(c_path):
    """
    Return (ret_type_str, func_name, arg_type_str) by scanning for the
    function definition.  Handles the four argument conventions used across
    EvoApproxLib .c files:

        int16_t  mulXXs_ID(int8_t A, int8_t B)          -- standard (A, B)
        uint64_t mulXXu_ID(uint64_t a, uint64_t b)       -- lowercase (a, b)
        uint64_t mulXXu_ID(const uint64_t B,const uint64_t A)  -- reversed, const
        uint16_t mulXXu_ID(const uint16_t A,const uint16_t B)  -- const (A, B)
    """
    text = c_path.read_text(errors="replace")
    m = re.search(
        r"(u?int\d+_t)\s+(mul\w+)\s*"
        r"\(\s*(?:const\s+)?(u?int\d+_t)\s+[AaBb]\s*,"
        r"\s*(?:const\s+)?(u?int\d+_t)\s+[AaBb]\s*\)",
        text,
    )
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def parse_func_name(func_name):
    """
    Extract (bit_width, sign) from names like mul8s_1KV6 or mul12u_2PP.
    sign is 's' (signed) or 'u' (unsigned).
    Returns (None, None) if the name does not match the expected pattern.
    """
    m = re.match(r"mul(\d+)([su])_", func_name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def input_range(bit_width, sign):
    """Return (lo, hi) inclusive integer range for operand inputs."""
    if sign == "s":
        lo = -(1 << (bit_width - 1))
        hi =  (1 << (bit_width - 1)) - 1
    else:
        lo = 0
        hi = (1 << bit_width) - 1
    return lo, hi


def sign_extend(approx, bit_width, sign, ret_type_str):
    """
    Correct the raw approx array (int64) for the signed result encoding.

    EvoApproxLib multipliers store the product in a C container that may be
    wider than the true result width (2*bit_width bits), or may be unsigned.
    In both cases the upper bits are 0 and the two's-complement sign bit is
    at position (2*bit_width - 1), NOT at the C type's MSB.

    Sign extension is needed when sign=='s' AND either:
      (a) the return type is unsigned  (uint*_t), OR
      (b) the return type is signed but wider than the result
          (e.g. int32_t holding a 24-bit 12x12 product).

    When the return type is signed and exactly as wide as the result
    (e.g. int16_t for 8x8, int32_t for 16x16), ctypes already delivers the
    correct signed value — no further adjustment.
    """
    if sign != "s":
        return approx          # unsigned multipliers: always non-negative

    result_bits   = 2 * bit_width
    ret_bits      = TYPE_BITS[ret_type_str]
    ret_is_signed = ret_type_str.startswith("int")

    if ret_is_signed and result_bits == ret_bits:
        return approx          # ctypes already sign-extended correctly

    # Need to sign-extend from bit (result_bits - 1)
    sign_val = np.int64(1 << (result_bits - 1))
    full_val = np.int64(1 << result_bits)
    neg_mask = (approx & sign_val).astype(bool)
    approx   = approx.copy()
    approx[neg_mask] -= full_val
    return approx


def compile_shared_lib(c_path, so_path):
    proc = subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", so_path, str(c_path)],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(c_path, n_samples, rng):
    """
    Compile one .c multiplier, evaluate it over inputs, and return a stats dict.

    8-bit  => exhaustive over all (hi-lo+1)^2 pairs (exact population stats).
    wider  => Monte Carlo with n_samples random pairs.

    Error is defined as:  approximate_result - exact_result
    """
    ret_type_str, func_name, arg_type_str = parse_c_signature(c_path)
    if func_name is None:
        return None
    bit_width, sign = parse_func_name(func_name)
    if bit_width is None:
        return None

    lo, hi = input_range(bit_width, sign)

    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as tmp:
        so_path = tmp.name

    try:
        if not compile_shared_lib(c_path, so_path):
            print(f"  [WARN] Compilation failed: {c_path.name}", file=sys.stderr)
            return None

        lib  = ctypes.CDLL(so_path)
        func = getattr(lib, func_name)
        func.restype  = CTYPE_MAP[ret_type_str]
        func.argtypes = [CTYPE_MAP[arg_type_str], CTYPE_MAP[arg_type_str]]
        ctype_a = CTYPE_MAP[arg_type_str]

        # Build input arrays ------------------------------------------------
        if bit_width <= 8:
            vals = np.arange(lo, hi + 1, dtype=np.int64)
            As_grid, Bs_grid = np.meshgrid(vals, vals)
            As       = As_grid.ravel()
            Bs       = Bs_grid.ravel()
            sim_type = "exhaustive"
        else:
            As       = rng.integers(lo, hi + 1, size=n_samples, dtype=np.int64)
            Bs       = rng.integers(lo, hi + 1, size=n_samples, dtype=np.int64)
            sim_type = f"MC n={len(As):,}"

        n_used   = len(As)
        accurate = As * Bs       # exact integer multiply (int64 — no overflow)

        # Call approximate function -----------------------------------------
        np_dtype = NUMPY_MAP[arg_type_str]
        As_cast  = As.astype(np_dtype)
        Bs_cast  = Bs.astype(np_dtype)

        approx = np.empty(n_used, dtype=np.int64)
        for i in range(n_used):
            approx[i] = func(ctype_a(int(As_cast[i])), ctype_a(int(Bs_cast[i])))

        # Fix signed encoding: sign-extend where ctypes didn't do it -------
        approx = sign_extend(approx, bit_width, sign, ret_type_str)

        errors = approx - accurate
        mu     = float(np.mean(errors))
        sigma2 = float(np.var(errors,  ddof=0))
        sigma  = float(np.std(errors,  ddof=0))

        return dict(func_name=func_name, bit_width=bit_width, sign=sign,
                    mu=mu, sigma2=sigma2, sigma=sigma,
                    n_used=n_used, sim_type=sim_type)
    finally:
        os.unlink(so_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_N,
                        help=f"Monte Carlo samples for bit-widths > 8 (default {DEFAULT_N})")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    # Discover .c files; deduplicate by function name (same multiplier may
    # appear under multiple pareto_* subdirectories).
    multipliers = {}
    for c_path in sorted(VSRC_DIR.rglob("*.c")):
        _, func_name, _ = parse_c_signature(c_path)
        if func_name is None:
            continue
        bit_width, sign = parse_func_name(func_name)
        if bit_width is None:
            continue   # skip non-standard names (e.g. mul8_364)

        pareto_dir = c_path.parent.name         # e.g. "pareto_pwr_mae"
        category   = c_path.parent.parent.name  # e.g. "8x8_signed"

        if func_name not in multipliers:
            multipliers[func_name] = dict(c_path=c_path, category=category,
                                          pareto_dirs=[pareto_dir])
        else:
            multipliers[func_name]["pareto_dirs"].append(pareto_dir)

    total = len(multipliers)
    print(f"Found {total} unique multipliers under {VSRC_DIR}")
    print(f"Monte Carlo samples for bit-width > 8: {args.samples:,}")
    print(f"Output CSV: {OUTPUT_CSV}\n")

    rows = []
    for idx, (func_name, info) in enumerate(sorted(multipliers.items()), 1):
        c_path      = info["c_path"]
        category    = info["category"]
        pareto_dirs = ";".join(sorted(set(info["pareto_dirs"])))

        print(f"[{idx:3d}/{total}] {func_name:<22s} ({category})", end="  ", flush=True)

        result = run_simulation(c_path, args.samples, rng)
        if result is None:
            print("FAILED")
            continue

        print(f"mu={result['mu']:+.4f}  sigma2={result['sigma2']:.4f}  [{result['sim_type']}]")

        rows.append({
            "multiplier_name" : func_name,
            "category"        : category,
            "pareto_dirs"     : pareto_dirs,
            "bit_width"       : result["bit_width"],
            "signed"          : (result["sign"] == "s"),
            "mean_error"      : result["mu"],
            "variance_error"  : result["sigma2"],
            "std_error"       : result["sigma"],
            "n_samples"       : result["n_used"],
            "simulation_type" : result["sim_type"],
        })

    fieldnames = [
        "multiplier_name", "category", "pareto_dirs",
        "bit_width", "signed",
        "mean_error", "variance_error", "std_error",
        "n_samples", "simulation_type",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)}/{total} multipliers written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
