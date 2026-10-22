# DL GIS Exchange - vector_mesh.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shapefile features -> one Blender mesh per layer, in local scene metres.

Points   -> vertices.                 Lines -> chains of edges.
Polygons -> faces. Measured 2026-10-10 (probe 03, 03b; research note 002):
  - a feature WITHOUT holes becomes one n-gon per ring (exact, the fewest faces);
  - a feature WITH holes becomes triangles from mathutils.geometry.delaunay_2d_cdt, output type 2 (constraint edges
    and faces, holes open): exact in Blender 4.5 and 5.2; type 5 (n-gons) filled the real holes and the dissolve
    step differs between versions - both left out;
  - "which ring is a hole" is NOT decided by winding (unreliable in shapefiles); a ring is a hole when it lies inside
    another ring of the same feature (point-in-polygon of its first vertex).
Attributes: the dbf fields become Blender attributes on the feature's domain (POINT for points, EDGE for lines,
FACE for polygons): numbers FLOAT / INT, text STRING, plus "fid" (INT, the feature index in the file).
"""
from __future__ import annotations

import numpy as np


def point_in_ring(pt, ring) -> bool:
    """Even-odd rule."""
    x, y = pt
    xs, ys = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / (yj - ys + 1e-300) + xs)
    return bool(cross.sum() % 2)


def classify_rings(parts):
    """[(ring, depth)] - depth 0 = outer, 1 = hole, 2 = island in a hole ..."""
    out = []
    for i, r in enumerate(parts):
        depth = 0
        for j, o in enumerate(parts):
            if i != j and len(o) >= 3 and point_in_ring(r[0], o):
                depth += 1
        out.append((r, depth))
    return out


def polygon_faces(parts, mode="AUTO"):
    """(verts (n,2), faces [[i,...]]) for one feature.  mode: AUTO (n-gons, triangles where holes) | TRIANGLES |
    OUTER (outer rings only, holes filled)."""
    rings = classify_rings(parts)
    has_holes = any(d % 2 == 1 for _, d in rings)
    if mode == "OUTER" or (mode == "AUTO" and not has_holes):
        verts, faces = [], []
        for r, d in rings:
            if d % 2 == 0:
                base = len(verts)
                verts.extend(map(tuple, r))
                faces.append(list(range(base, base + len(r))))
        return np.array(verts, dtype=np.float64).reshape(-1, 2), faces
    # triangles with holes open: CDT output type 2 (measured 2026-10-10)
    from mathutils import Vector
    from mathutils.geometry import delaunay_2d_cdt
    v_in, e_in, f_in = [], [], []
    for r, d in rings:
        base = len(v_in)
        v_in.extend(Vector((float(x), float(y))) for x, y in r)
        n = len(r)
        e_in.extend((base + i, base + (i + 1) % n) for i in range(n))
        f_in.append(list(range(base, base + n)))
    res = delaunay_2d_cdt(v_in, e_in, f_in, 2, 1e-6)
    verts = np.array([(v.x, v.y) for v in res[0]], dtype=np.float64).reshape(-1, 2)
    faces = [list(f) for f in res[2]]
    return verts, faces


def build_layer(layer, origin_E, origin_N, faces_mode="AUTO", z_mode="ZERO", z_field=None, z_scale=1.0):
    """Arrays for the mesh: verts (n,3), edges [(a,b)], faces [[...]], per-element feature index, in LOCAL metres."""
    verts, edges, faces = [], [], []
    v_fid, e_fid, f_fid = [], [], []
    for feat in layer.features:
        z = 0.0
        if z_mode == "FIELD" and z_field:
            try:
                z = float(feat.attrs.get(z_field) or 0.0) * z_scale
            except (TypeError, ValueError):
                z = 0.0
        if layer.kind == "point":
            for p in feat.parts:
                for x, y in p:
                    verts.append((x - origin_E, y - origin_N, z)); v_fid.append(feat.fid)
        elif layer.kind == "line":
            for p in feat.parts:
                base = len(verts)
                for x, y in p:
                    verts.append((x - origin_E, y - origin_N, z)); v_fid.append(feat.fid)
                for i in range(len(p) - 1):
                    edges.append((base + i, base + i + 1)); e_fid.append(feat.fid)
        else:
            # triangulate in LOCAL coordinates: mathutils is single precision, and a CDT at E 655,000 / N 7,738,000
            # came out 18 m2 off on an 88,000 m2 polygon while the same rings minus the origin matched GDAL exactly
            # (probe 04, 2026-10-10)
            local_parts = [p - np.array([origin_E, origin_N]) for p in feat.parts]
            fv, ff = polygon_faces(local_parts, faces_mode)
            base = len(verts)
            for x, y in fv:
                verts.append((x, y, z)); v_fid.append(feat.fid)
            for f in ff:
                faces.append([base + i for i in f]); f_fid.append(feat.fid)
    return (np.array(verts, dtype=np.float64).reshape(-1, 3), edges, faces,
            np.array(v_fid, dtype=np.int64), np.array(e_fid, dtype=np.int64), np.array(f_fid, dtype=np.int64))


def mesh_from_arrays(name, verts, edges, faces):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [tuple(e) for e in edges], faces)
    me.validate(verbose=False)
    me.update()
    return me


def add_attributes(me, layer, fids_by_domain):
    """dbf fields as attributes on the layer's domain + "fid"; fids_by_domain = {"POINT": arr, "EDGE": arr, "FACE": arr}."""
    domain = {"point": "POINT", "line": "EDGE", "polygon": "FACE"}[layer.kind]
    fids = fids_by_domain[domain]
    count = {"POINT": len(me.vertices), "EDGE": len(me.edges), "FACE": len(me.polygons)}[domain]
    if len(fids) != count:
        return 0   # validate() changed the counts: skip attributes rather than write wrong ones
    by_fid = {f.fid: f.attrs for f in layer.features}
    att = me.attributes.new("fid", "INT", domain)
    att.data.foreach_set("value", fids.astype(np.int32))
    written = 1
    for name, ftype, decimals in layer.fields:
        safe = name[:63] or "field"
        if safe in me.attributes:
            safe = safe + "_"
        vals = [by_fid.get(int(i), {}).get(name) for i in fids]
        if ftype == "N" and decimals == 0:
            a = me.attributes.new(safe, "INT", domain)
            out = []
            for v in vals:
                try:
                    out.append(int(float(v)) if v not in (None, "") else 0)
                except (TypeError, ValueError):
                    out.append(0)
            a.data.foreach_set("value", np.clip(np.array(out, dtype=np.int64), -2**31, 2**31 - 1).astype(np.int32))
        elif ftype in ("N", "F"):
            a = me.attributes.new(safe, "FLOAT", domain)
            out = []
            for v in vals:
                try:
                    out.append(float(v) if v not in (None, "") else 0.0)
                except (TypeError, ValueError):
                    out.append(0.0)
            a.data.foreach_set("value", np.array(out, dtype=np.float32))
        elif ftype == "L":
            a = me.attributes.new(safe, "BOOLEAN", domain)
            a.data.foreach_set("value", np.array([bool(v) for v in vals], dtype=bool))
        else:
            a = me.attributes.new(safe, "STRING", domain)
            for d, v in zip(a.data, vals):
                d.value = ("" if v is None else str(v)).encode("utf-8")   # StringAttributeValue.value takes bytes (4.5 + 5.2)
        written += 1
    return written
