# DL GIS Exchange - difference.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Difference raster: NEW surface minus EXISTING surface on one grid - cut and fill.

    difference = z_new - z_existing      positive = fill (the new surface lies above), negative = cut
A cell where either surface has no data is nodata. Volumes: fill = sum of the positive differences x cell area,
cut = sum of the negative differences x cell area (both reported as positive numbers), net = fill - cut.
Both surfaces are sampled by ray-casting (sampling.py), the HIGHEST hit of each.

The optional mesh (0.4.3: its height IS the difference by default - flat at 0, fill up, cut down - or draped on the
existing surface on request), with a float point attribute "difference" (metres) and a colour attribute "cutfill":
cut in red, fill in blue, zero white, scaled to the largest |difference|; a material "DL cut fill" shows it and the
operator switches the 3D views' solid shading to Attribute colours.
The optional solids (0.4.3, 2026-10-12): two closed meshes between the surfaces - "Fill solid"
(positive, where the new surface lies above) and "Cut solid" (negative, where it lies below), each with its volume
from bmesh, flat materials "DL fill solid" / "DL cut solid"; see solid_arrays.
"""
from __future__ import annotations

import numpy as np

CUT_RGB = (0.80, 0.25, 0.20)
FILL_RGB = (0.20, 0.45, 0.85)


def difference(z_existing: np.ndarray, z_new: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, dict]:
    """(difference array with NaN where either is NaN, stats dict with volumes in m3 and areas in m2)."""
    a = np.asarray(z_existing, dtype=np.float64)
    b = np.asarray(z_new, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"surfaces sampled on different grids: {a.shape} vs {b.shape}")
    d = b - a
    valid = ~np.isnan(d)
    cell = float(dx) * float(dy)
    pos = np.where(valid & (d > 0), d, 0.0)
    neg = np.where(valid & (d < 0), -d, 0.0)
    stats = {
        "cells_compared": int(valid.sum()),
        "cells_nodata": int((~valid).sum()),
        "area_m2": float(valid.sum() * cell),
        "fill_m3": float(pos.sum() * cell),
        "cut_m3": float(neg.sum() * cell),
        "net_m3": float((pos.sum() - neg.sum()) * cell),
        "max_fill_m": float(pos.max()) if valid.any() else 0.0,
        "max_cut_m": float(neg.max()) if valid.any() else 0.0,
    }
    return d, stats


def cutfill_colours(diff_values: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """RGBA per value: cut red, fill blue, zero white, linear to +-scale (default: the largest |difference|)."""
    v = np.asarray(diff_values, dtype=np.float64)
    if scale is None or scale <= 0:
        scale = float(np.nanmax(np.abs(v))) if v.size and np.isfinite(v).any() else 1.0
        scale = scale or 1.0
    t = np.clip(np.nan_to_num(v, nan=0.0) / scale, -1.0, 1.0)
    rgba = np.ones((len(v), 4), dtype=np.float32)
    cut, fill = np.array(CUT_RGB), np.array(FILL_RGB)
    neg = t < 0
    pos = t > 0
    rgba[neg, :3] = 1.0 + (cut - 1.0) * (-t[neg])[:, None]
    rgba[pos, :3] = 1.0 + (fill - 1.0) * (t[pos])[:, None]
    return rgba, float(scale)


def cutfill_material(name="DL cut fill"):
    """A material that shows the 'cutfill' colour attribute (Material Preview / Rendered); reused when it exists."""
    import bpy
    mat = bpy.data.materials.get(name)
    if mat is not None and mat.use_nodes:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    att = nt.nodes.new("ShaderNodeVertexColor")
    att.layer_name = "cutfill"
    att.location = (-300, 300)
    if bsdf is not None:
        nt.links.new(att.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.9
    return mat


def show_attribute_colours(context):
    """Switch the solid shading of the current screen's 3D views to the colour attribute, so the cut / fill colours
    show at once (reported to the user). Returns how many views were switched."""
    n = 0
    screen = getattr(context, "screen", None)
    if screen is None:
        return 0
    for area in screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D" and space.shading.type == "SOLID" and space.shading.color_type != "VERTEX":
                    space.shading.color_type = "VERTEX"
                    n += 1
    return n


def plain_material(name, rgb):
    """A flat-coloured material (viewport colour too), reused when it exists."""
    import bpy
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.9
        mat.diffuse_color = (*rgb, 1.0)
    return mat


SOLID_MIN_M = 0.001   # a difference smaller than 1 mm is noise (float32 vertices), not cut or fill


def solid_arrays(d, z_existing, z_new, kind, ul_E, ul_N, dx, dy, origin_E, origin_N, min_m=SOLID_MIN_M):
    """A closed solid between the two surfaces (0.4.3): kind "fill" where new > existing, "cut" where new < existing.
    A grid quad belongs to the solid when ANY of its four corners (vertices at cell centres) differs by more than
    min_m in that direction and all four corners have data; the solid's top is max(new, existing) for fill (bottom
    existing), its bottom min(new, existing) for cut (top existing), so the thickness tapers to zero at the solid's
    edge and the volume follows the raster's. Faces: tops counter-clockwise from above, bottoms reversed, walls
    along the quad set's boundary facing outward. Where two quads touch at ONE corner only (a diagonal contact) that
    vertex is doubled so every part stays watertight (found on a test file, 2026-10-12). Local metres.
    Returns (verts (n,3) float64, faces list, info)."""
    d = np.asarray(d, dtype=np.float64)
    ze = np.asarray(z_existing, dtype=np.float64); zn = np.asarray(z_new, dtype=np.float64)
    ny, nx = d.shape
    valid = ~np.isnan(d)
    if kind == "fill":
        on = valid & (d > min_m)
        zt, zb = np.where(valid, np.maximum(zn, ze), np.nan), ze
    else:
        on = valid & (d < -min_m)
        zt, zb = ze, np.where(valid, np.minimum(zn, ze), np.nan)
    all4 = valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:]
    any_on = on[:-1, :-1] | on[1:, :-1] | on[1:, 1:] | on[:-1, 1:]
    ok = all4 & any_on                                   # quads of the solid, shape (ny-1, nx-1)
    if not ok.any():
        return np.zeros((0, 3)), [], {"n_quads": 0, "n_verts": 0, "n_split": 0}
    # the four quads around every node (r, c): a = (r-1, c-1), b = (r-1, c), c_ = (r, c), dd = (r, c-1)
    okp = np.pad(ok, 1, constant_values=False)           # okp[qr+1, qc+1] = ok[qr, qc]
    a = okp[0:ny, 0:nx]; b = okp[0:ny, 1:nx + 1]; c_ = okp[1:ny + 1, 1:nx + 1]; dd = okp[1:ny + 1, 0:nx]
    used = a | b | c_ | dd
    split = (a & c_ & ~b & ~dd) | (b & dd & ~a & ~c_)    # diagonal contact: two fans -> two vertices
    flat_used = np.flatnonzero(used.ravel())
    n0 = len(flat_used)
    idx0 = np.full(ny * nx, -1, dtype=np.int64); idx0[flat_used] = np.arange(n0)
    flat_split = np.flatnonzero(split.ravel())
    idx1 = idx0.copy(); idx1[flat_split] = n0 + np.arange(len(flat_split))
    nodes = np.concatenate([flat_used, flat_split])      # top vertex i <-> node nodes[i]; bottom = i + nv
    nv = len(nodes)
    rr, cc = np.divmod(nodes, nx)
    xs = (ul_E - origin_E) + (cc + 0.5) * dx
    ys = (ul_N - origin_N) - (rr + 0.5) * dy
    verts = np.vstack([np.column_stack([xs, ys, zt.ravel()[nodes]]), np.column_stack([xs, ys, zb.ravel()[nodes]])])
    qr, qc = np.nonzero(ok)
    # corner 0 (qr, qc) and corner 3 (qr, qc+1): the quad lies BELOW the node (row qr == node row) -> copy 1;
    # corners 1 (qr+1, qc) and 2 (qr+1, qc+1): the quad lies above the node -> copy 0 (identical unless split)
    v0 = idx1[qr * nx + qc]; v1 = idx0[(qr + 1) * nx + qc]; v2 = idx0[(qr + 1) * nx + qc + 1]; v3 = idx1[qr * nx + qc + 1]
    tops = np.column_stack([v0, v1, v2, v3])                               # counter-clockwise from above
    bottoms = np.column_stack([v3, v2, v1, v0]) + nv                       # reversed
    faces = [list(map(int, f)) for f in tops] + [list(map(int, f)) for f in bottoms]
    # walls on quad edges whose neighbour is not ok: edge 0-1 west, 1-2 south, 2-3 east, 3-0 north
    west = ~okp[qr + 1, qc]; south = ~okp[qr + 2, qc + 1]; east = ~okp[qr + 1, qc + 2]; north = ~okp[qr, qc + 1]
    for sel, p, q in ((west, v0, v1), (south, v1, v2), (east, v2, v3), (north, v3, v0)):
        pa, qb = p[sel], q[sel]
        faces += [[int(y), int(x), int(x) + nv, int(y) + nv] for x, y in zip(pa, qb)]   # outward for an edge x->y
    return verts, faces, {"n_quads": int(ok.sum()), "n_verts": 2 * nv, "n_split": int(len(flat_split))}


def solid_volume(me):
    """(volume in m3 from the closed mesh, is_manifold) through bmesh; normals recalculated outward first."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    vol = float(bm.calc_volume(signed=True))
    manifold = all(e.is_manifold for e in bm.edges) and all(v.is_manifold for v in bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return abs(vol), manifold


def add_attributes(me, diff_flat: np.ndarray, kept: np.ndarray):
    """Write "difference" (FLOAT, POINT) and "cutfill" (colour, POINT) onto a mesh built from the kept cells."""
    vals = np.asarray(diff_flat, dtype=np.float64)[kept]
    att = me.attributes.new("difference", "FLOAT", "POINT")
    att.data.foreach_set("value", np.nan_to_num(vals, nan=0.0).astype(np.float32))
    col = me.color_attributes.new("cutfill", "FLOAT_COLOR", "POINT")
    rgba, scale = cutfill_colours(vals)
    col.data.foreach_set("color", rgba.ravel())
    me.color_attributes.active_color = col
    return scale
