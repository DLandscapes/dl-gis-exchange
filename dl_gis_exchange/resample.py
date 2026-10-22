# DL GIS Exchange - resample.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""0.7.0: resample a terrain grid to a new cell size, keeping its outer upper-left corner (pure numpy, no Blender).

The two directions follow established GIS practice (research note "DL - resample and feature research - 001"):
  refine  (new cell < source cell): BILINEAR interpolation between the source cell centres (GRASS r.import recommends
          bilinear for DEMs; GDAL "-r bilinear"). It adds no measured detail - the new cells are interpolated.
  coarsen (new cell > source cell): AREA-WEIGHTED MEAN of the source cells each new cell overlaps (GDAL "-r average",
          GRASS r.resamp.stats -w), nodata cells left out. Over the same extent this keeps the volume.
Grid convention as everywhere in the extension: (ul_E, ul_N) = the OUTER upper-left corner, rows from north to south.
"""
from __future__ import annotations

import math

import numpy as np

MAX_CELLS = 50_000_000


class ResampleError(ValueError):
    """A problem the user can act on; the message says what to do."""


def target_size(rows: int, cols: int, d0: float, d1: float):
    """(rows1, cols1, nested): the new grid covers the source extent from the same corner; where the extent is not a
    whole number of new cells, the last row / column reaches past it."""
    fx, fy = cols * d0 / d1, rows * d0 / d1
    cols1 = int(round(fx)) if abs(fx - round(fx)) < 1e-6 else int(math.ceil(fx))
    rows1 = int(round(fy)) if abs(fy - round(fy)) < 1e-6 else int(math.ceil(fy))
    ratio = d1 / d0
    nested = abs(ratio - round(ratio)) < 1e-9 or abs(1.0 / ratio - round(1.0 / ratio)) < 1e-9
    return max(rows1, 1), max(cols1, 1), nested


def cells_from_lattice(xy_local: np.ndarray, z: np.ndarray, origin, ul_E, ul_N, d, rows, cols, corners: bool):
    """The grid's CELL values (rows x cols, NaN = no data) from a mesh's vertices (local x, y + origin): at cell centres
    the vertex height is the cell value; at cell corners a cell is the mean of its four corners (missing -> NaN).
    Raises ResampleError when a vertex lies off the lattice (the mesh was moved or edited in x / y)."""
    oe, on = origin
    half = 0.0 if corners else 0.5
    fc = (xy_local[:, 0] + oe - ul_E) / d - half
    fr = (ul_N - (xy_local[:, 1] + on)) / d - half
    c, r = np.rint(fc).astype(np.int64), np.rint(fr).astype(np.int64)
    off = max(float(np.abs(fc - c).max() * d), float(np.abs(fr - r).max() * d)) if len(c) else 0.0
    nr, nc = rows + (1 if corners else 0), cols + (1 if corners else 0)
    if off > 1e-3 or (len(c) and (c.min() < 0 or r.min() < 0 or c.max() >= nc or r.max() >= nr)):
        raise ResampleError("The mesh is no longer the imported grid (vertices moved sideways or the object was moved). "
                            "Use 'Grid from selection' to sample it onto a new grid instead.")
    lat = np.full((nr, nc), np.nan)
    lat[r, c] = z
    if not corners:
        return lat
    return (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:]) / 4.0      # NaN if any corner is missing


def bilinear(z: np.ndarray, d0: float, d1: float, rows1: int, cols1: int) -> np.ndarray:
    """Bilinear interpolation between source cell centres at the new cell centres. A new cell whose centre falls in a
    source cell without data stays without data; missing neighbours are left out of the weights (as GDAL does).
    Beyond the outermost source centres the edge value is held (no extrapolation)."""
    rows, cols = z.shape

    def axis(n_new, n_src):
        p = (np.arange(n_new) + 0.5) * d1 / d0 - 0.5           # position in source-centre index units
        home = np.floor((np.arange(n_new) + 0.5) * d1 / d0).astype(np.int64)
        p = np.clip(p, 0.0, n_src - 1.0)
        i0 = np.minimum(np.floor(p).astype(np.int64), max(n_src - 2, 0))
        t = p - i0
        i1 = np.minimum(i0 + 1, n_src - 1)
        return i0, i1, t, home

    r0, r1, ty, hr = axis(rows1, rows)
    c0, c1, tx, hc = axis(cols1, cols)
    out = np.zeros((rows1, cols1))
    wsum = np.zeros((rows1, cols1))
    for rr, wy in ((r0, 1.0 - ty), (r1, ty)):
        for cc, wx in ((c0, 1.0 - tx), (c1, tx)):
            v = z[np.ix_(rr, cc)]
            w = wy[:, None] * wx[None, :]
            ok = ~np.isnan(v) & (w > 0)
            out += np.where(ok, v * w, 0.0)
            wsum += np.where(ok, w, 0.0)
    res = np.where(wsum > 0, out / np.where(wsum > 0, wsum, 1.0), np.nan)
    inside = (hr[:, None] < rows) & (hc[None, :] < cols)
    home_nan = np.ones((rows1, cols1), dtype=bool)
    home_nan[np.ix_(hr < rows, hc < cols)] = np.isnan(z[np.ix_(hr[hr < rows], hc[hc < cols])])
    return np.where(inside & ~home_nan, res, np.nan)


def _overlap(n_new: int, n_src: int, d0: float, d1: float) -> np.ndarray:
    """(n_new x n_src) lengths by which new cell j and source cell i overlap along one axis."""
    a0 = np.arange(n_new)[:, None] * d1
    b0 = np.arange(n_src)[None, :] * d0
    return np.clip(np.minimum(a0 + d1, b0 + d0) - np.maximum(a0, b0), 0.0, None)


def area_mean(z: np.ndarray, d0: float, d1: float, rows1: int, cols1: int) -> np.ndarray:
    """Area-weighted mean of the overlapped source cells with data; no data where none overlaps."""
    wy = _overlap(rows1, z.shape[0], d0, d1)
    wx = _overlap(cols1, z.shape[1], d0, d1)
    valid = ~np.isnan(z)
    num = wy @ np.where(valid, z, 0.0) @ wx.T
    den = wy @ valid.astype(np.float64) @ wx.T
    return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)


def resample(z: np.ndarray, d0: float, d1: float, method: str = "AUTO"):
    """(new cells, method used, rows1, cols1, nested). method: AUTO | BILINEAR | AREA_MEAN."""
    rows1, cols1, nested = target_size(z.shape[0], z.shape[1], d0, d1)
    if rows1 * cols1 > MAX_CELLS:
        raise ResampleError(f"{rows1:,} x {cols1:,} = {rows1 * cols1 / 1e6:,.0f} million cells is too many cells "
                            f"(limit {MAX_CELLS / 1e6:.0f} million): choose a larger cell size or clip the terrain first")
    if method == "AUTO":
        method = "BILINEAR" if d1 < d0 else "AREA_MEAN"
    if abs(d1 - d0) < 1e-12:
        return z.astype(np.float64).copy(), "copy", rows1, cols1, True
    out = bilinear(z, d0, d1, rows1, cols1) if method == "BILINEAR" else area_mean(z, d0, d1, rows1, cols1)
    return out, method, rows1, cols1, nested
