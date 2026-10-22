# DL GIS Exchange - gpkg_io.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read GeoPackage (.gpkg) vector layers into the same Layer / Feature records as vector_io's shapefile reader,
without GDAL: Python's own sqlite3 opens the file, the GeoPackage binary header (OGC 12-128r19, section 2.1.3) and
the WKB geometry (OGC 06-103r4 / ISO 19125) are parsed here. Z and M are dropped, as the shapefile reader drops them.

Measured on a teaching data package (2026-10-12): every layer EPSG:25833, geometry types POINT, LINESTRING, MULTIPOINT,
MULTILINESTRING, MULTIPOLYGON, little-endian headers, envelope type 0 or 1, column names with Norwegian letters.
"""
from __future__ import annotations

import os
import sqlite3
import struct

import numpy as np

from .vector_io import Feature, Layer, VectorError

_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
_KIND_OF = {1: "point", 2: "line", 3: "polygon", 4: "point", 5: "line", 6: "polygon", 7: "collection"}
_NUMERIC_INT = ("INT", "INTEGER", "MEDIUMINT", "SMALLINT", "TINYINT", "BOOLEAN", "BIGINT")
_NUMERIC_REAL = ("REAL", "DOUBLE", "FLOAT")


def list_layers(path: str):
    """[(table, geometry type name, srs_id, feature count)] of the vector layers in a GeoPackage."""
    if not os.path.isfile(path):
        raise VectorError(f"File not found: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            rows = con.execute("SELECT c.table_name, g.geometry_type_name, g.srs_id FROM gpkg_contents c "
                               "JOIN gpkg_geometry_columns g ON g.table_name = c.table_name "
                               "WHERE c.data_type = 'features' ORDER BY c.table_name").fetchall()
        except sqlite3.DatabaseError as ex:
            raise VectorError(f"Not a GeoPackage (no gpkg_contents / gpkg_geometry_columns): {ex}") from ex
        out = []
        for table, gtype, srs in rows:
            n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            out.append((table, (gtype or "GEOMETRY").upper(), int(srs) if srs is not None else None, int(n)))
        return out
    finally:
        con.close()


def _srs(con, srs_id):
    """(epsg or None, name) for a gpkg_spatial_ref_sys row."""
    row = con.execute("SELECT organization, organization_coordsys_id, srs_name FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
                      (srs_id,)).fetchone()
    if row is None:
        return None, f"srs {srs_id}"
    org, code, name = row
    epsg = int(code) if org and org.upper() == "EPSG" and code is not None and int(code) > 0 else None
    return epsg, name or (f"EPSG:{epsg}" if epsg else f"srs {srs_id}")


def parse_wkb(buf: bytes, off: int = 0):
    """One WKB geometry at buf[off:] -> (base type 1..7, geometry, next offset). Points -> (x, y); line strings ->
    (n, 2) arrays; polygons -> [rings]; multi types -> [members]; collections -> [(type, geometry), ...]."""
    big = buf[off] == 0
    end = ">" if big else "<"
    t = struct.unpack_from(end + "I", buf, off + 1)[0]
    off += 5
    has_z = bool(t & 0x80000000)
    has_m = bool(t & 0x40000000)
    if t & 0x20000000:          # EWKB SRID
        off += 4
    t &= 0x0FFFFFFF
    dims = t // 1000
    base = t % 1000
    has_z = has_z or dims in (1, 3)
    has_m = has_m or dims in (2, 3)
    ncoord = 2 + int(has_z) + int(has_m)
    fmt = end + "d" * ncoord
    size = 8 * ncoord

    def read_points(o):
        n = struct.unpack_from(end + "I", buf, o)[0]
        o += 4
        arr = np.frombuffer(buf, dtype=(">f8" if big else "<f8"), count=n * ncoord, offset=o).reshape(n, ncoord)[:, :2]
        return np.array(arr, dtype=np.float64), o + n * size

    if base == 1:
        vals = struct.unpack_from(fmt, buf, off)
        return 1, (vals[0], vals[1]), off + size
    if base == 2:
        pts, off = read_points(off)
        return 2, pts, off
    if base == 3:
        nr = struct.unpack_from(end + "I", buf, off)[0]
        off += 4
        rings = []
        for _ in range(nr):
            pts, off = read_points(off)
            rings.append(pts)
        return 3, rings, off
    if base in (4, 5, 6, 7):
        n = struct.unpack_from(end + "I", buf, off)[0]
        off += 4
        members = []
        for _ in range(n):
            bt, g, off = parse_wkb(buf, off)
            members.append((bt, g) if base == 7 else g)
        return base, members, off
    raise VectorError(f"WKB geometry type {t} is not supported")


def parse_gpb(blob: bytes):
    """A GeoPackage geometry blob -> (base type, geometry, srs_id) or (None, None, srs_id) when empty."""
    if blob is None or len(blob) < 8 or blob[:2] != b"GP":
        raise VectorError("geometry blob without the GeoPackage header")
    flags = blob[3]
    little = bool(flags & 1)
    env = (flags >> 1) & 7
    empty = bool(flags & 0x10)
    srs_id = struct.unpack_from("<i" if little else ">i", blob, 4)[0]
    if env not in _ENVELOPE_BYTES:
        raise VectorError(f"GeoPackage envelope type {env} is not valid")
    if empty:
        return None, None, srs_id
    off = 8 + _ENVELOPE_BYTES[env]
    bt, geom, _ = parse_wkb(blob, off)
    return bt, geom, srs_id


def geometry_parts(bt: int, geom):
    """(kind, parts) of a parsed geometry in vector_io's convention: points -> one (n, 2) part; lines -> one part per
    polyline; polygons -> one part per RING with the closing point removed. Collections are flattened by kind."""
    if bt == 1:
        return "point", [np.array([geom], dtype=np.float64)]
    if bt == 4:
        pts = [g for g in geom if g is not None]
        return "point", [np.array(pts, dtype=np.float64)] if pts else []
    if bt == 2:
        return "line", [geom] if len(geom) >= 2 else []
    if bt == 5:
        return "line", [g for g in geom if len(g) >= 2]
    if bt in (3, 6):
        polys = [geom] if bt == 3 else geom
        parts = []
        for rings in polys:
            for r in rings:
                if len(r) > 1 and np.allclose(r[0], r[-1]):
                    r = r[:-1]
                if len(r) >= 3:
                    parts.append(r)
        return "polygon", parts
    if bt == 7:
        kinds = {}
        for mbt, mg in geom:
            k, p = geometry_parts(mbt, mg)
            kinds.setdefault(k, []).extend(p)
        if not kinds:
            return "point", []
        k = max(kinds, key=lambda kk: len(kinds[kk]))
        return k, kinds[k]
    raise VectorError(f"geometry type {bt} is not supported")


def _field_type(declared: str):
    d = (declared or "").upper().split("(")[0].strip()
    if d in _NUMERIC_INT:
        return "N", 0
    if d in _NUMERIC_REAL:
        return "F", 6
    return "C", 0


def read_gpkg(path: str, table: str | None = None) -> Layer:
    """One layer of a GeoPackage (the first when none is named) as a Layer; the kind from the declared geometry type,
    or from the majority of the features for a GEOMETRY column; features of another kind are skipped with a note."""
    layers = list_layers(path)
    if not layers:
        raise VectorError("The GeoPackage has no vector layers")
    if table is None:
        table = layers[0][0]
    meta = next((l for l in layers if l[0] == table), None)
    if meta is None:
        raise VectorError(f"Layer '{table}' is not in {os.path.basename(path)}: {', '.join(l[0] for l in layers)}")
    _, gtype, srs_id, _ = meta
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        gcol = con.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?", (table,)).fetchone()[0]
        info = con.execute(f'PRAGMA table_info("{table}")').fetchall()      # cid, name, type, notnull, dflt, pk
        pk = next((r[1] for r in info if r[5]), None)
        fields = [(r[1], *_field_type(r[2])) for r in info if r[1] != gcol and r[1] != pk]
        names = [f[0] for f in fields]
        cols = ", ".join(f'"{c}"' for c in ([pk] if pk else []) + [gcol] + names)
        epsg, crs_name = _srs(con, srs_id)
        declared = {"POINT": "point", "MULTIPOINT": "point", "LINESTRING": "line", "MULTILINESTRING": "line",
                    "POLYGON": "polygon", "MULTIPOLYGON": "polygon"}.get(gtype)
        feats_raw = []
        kinds = {"point": 0, "line": 0, "polygon": 0}
        skipped = 0
        for i, row in enumerate(con.execute(f"SELECT {cols} FROM \"{table}\"")):
            fid = int(row[0]) if pk else i
            blob = row[1 if pk else 0]
            vals = row[2:] if pk else row[1:]
            if blob is None:
                skipped += 1
                continue
            bt, geom, _ = parse_gpb(blob)
            if bt is None:
                skipped += 1
                continue
            kind, parts = geometry_parts(bt, geom)
            if not parts:
                skipped += 1
                continue
            kinds[kind] += 1
            attrs = {}
            for n, v in zip(names, vals):
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                attrs[n] = v
            feats_raw.append((kind, Feature(fid, parts, attrs)))
    finally:
        con.close()
    kind = declared or (max(kinds, key=kinds.get) if feats_raw else "point")
    feats = [f for k, f in feats_raw if k == kind]
    other = len(feats_raw) - len(feats)
    notes = []
    if skipped:
        notes.append(f"{skipped} empty features skipped")
    if other:
        notes.append(f"{other} features of another geometry type than {kind} skipped")
    if epsg is None:
        notes.append(f"CRS of srs {srs_id} is not an EPSG code ({crs_name}); the scene's CRS is assumed")
    if feats:
        allpts = np.vstack([p for f in feats for p in f.parts])
        bb = (float(allpts[:, 0].min()), float(allpts[:, 1].min()), float(allpts[:, 0].max()), float(allpts[:, 1].max()))
    else:
        bb = (0.0, 0.0, 0.0, 0.0)
    return Layer(kind, feats, fields, epsg, crs_name, bb, f"{os.path.basename(path)}:{table}", "utf-8", notes)
