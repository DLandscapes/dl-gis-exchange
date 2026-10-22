# DL GIS Exchange - meshing.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Raster grid -> quad mesh (numpy, then bulk foreach_set into bpy.types.Mesh).

Pattern from DL-Terrain Growth's build_demo.py (vertices at cell centres, quads built in numpy, foreach_set);
extended with nodata holes, the one-corner convention of geotiff_io, and a local origin.

Vertices sit at CELL CENTRES in local scene metres:  x = centre_E - origin_E,  y = centre_N - origin_N,  z = height.
The scene stays near (0, 0) and the UTM origin lives in double-precision ID properties (float32 would quantise
northings around 7.7e6 to 0.5 m - see the DL float32 rule).
Quads are wound counter-clockwise seen from above (normals point up): (r, c) -> (r+1, c) -> (r+1, c+1) -> (r, c+1).
"""
from __future__ import annotations

import numpy as np


def corner_heights(z: np.ndarray) -> np.ndarray:
    """0.5.2: heights at the CELL CORNERS, shape (rows + 1, cols + 1): the mean of the 1 to 4 neighbouring cells that
    have data, NaN where none has. With these, grid_to_quads called with the raster's outer corner shifted half a cell
    north-west places one vertex on every cell corner and one quad on every cell, so the mesh covers the raster's
    full extent (a user's requirement, 2026-10-13: it stays the same size). Heights are averaged, not exact."""
    z = np.asarray(z, dtype=np.float64)
    ny, nx = z.shape
    p = np.full((ny + 2, nx + 2), np.nan)
    p[1:-1, 1:-1] = z
    stack = np.stack([p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]])      # the 4 cells around each corner
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(stack)
        n = valid.sum(axis=0)
        s = np.where(valid, stack, 0.0).sum(axis=0)
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out.astype(np.float32)


def grid_to_quads(z: np.ndarray, ul_E: float, ul_N: float, dx: float, dy: float,
                  origin_E: float, origin_N: float, nodata_mode: str = "HOLES"):
    """Return (verts float32 (n,3), quads int32 (m,4), info dict).

    nodata_mode: "HOLES" - cells with NaN get no vertex, quads touching them are dropped;
                 "FLAT"  - NaN cells get the lowest height of the raster (a flat floor), every quad kept.
    """
    z = np.asarray(z, dtype=np.float32)
    ny, nx = z.shape
    if ny < 2 or nx < 2:
        raise ValueError("raster window smaller than 2 x 2 cells")
    valid = ~np.isnan(z)
    n_nodata = int((~valid).sum())
    info = {"rows": ny, "cols": nx, "n_nodata": n_nodata, "nodata_mode": nodata_mode}

    if nodata_mode == "FLAT" and n_nodata:
        floor = float(np.nanmin(z)) if valid.any() else 0.0
        z = np.where(valid, z, np.float32(floor))
        valid = np.ones_like(valid)
        info["flat_floor"] = floor
    elif nodata_mode not in ("HOLES", "FLAT"):
        raise ValueError(f"nodata_mode {nodata_mode!r}")

    # centres in local coordinates (float64 maths, float32 storage)
    xs = (ul_E - origin_E) + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    ys = (ul_N - origin_N) - (np.arange(ny, dtype=np.float64) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    verts = np.column_stack([X.ravel(), Y.ravel(), z.astype(np.float64).ravel()])

    # all quads, counter-clockwise from above
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    v00 = (j * nx + i).ravel()
    quads = np.column_stack([v00, v00 + nx, v00 + nx + 1, v00 + 1]).astype(np.int64)

    kept = np.arange(ny * nx)
    if n_nodata and nodata_mode == "HOLES":
        ok = (valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:]).ravel()
        quads = quads[ok]
        used = np.zeros(ny * nx, dtype=bool)
        used[quads.ravel()] = True
        remap = np.full(ny * nx, -1, dtype=np.int64)
        remap[used] = np.arange(int(used.sum()))
        verts = verts[used]
        quads = remap[quads]
        kept = np.flatnonzero(used)
        verts[:, 2] = np.where(np.isnan(verts[:, 2]), 0.0, verts[:, 2])  # cannot happen, belt and braces
    info["kept"] = kept   # flat cell index (row * cols + col) of each vertex, in vertex order (0.3.0)
    info["n_verts"] = int(len(verts))
    info["n_quads"] = int(len(quads))
    return verts.astype(np.float32), quads.astype(np.int32), info


def mesh_from_quads(name: str, verts: np.ndarray, quads: np.ndarray, smooth: bool = True):
    """Build a bpy mesh from the arrays with bulk foreach_set (bmesh is far slower on large grids)."""
    import bpy
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.ascontiguousarray(verts, dtype=np.float32).ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(quads, dtype=np.int32).ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate(verbose=False)
    if smooth and len(quads):
        me.polygons.foreach_set("use_smooth", np.ones(len(quads), dtype=bool))
    return me
