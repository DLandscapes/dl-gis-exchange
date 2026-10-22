# DL GIS Exchange - geojson_io.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read GeoJSON (.geojson / .json) into vector_io's Layer / Feature records, without GDAL.

Two kinds of file exist. Files after RFC 7946 (2016) carry WGS84 longitude / latitude and no CRS member; files after
the 2008 GeoJSON draft may carry a "crs" member naming a projected CRS (QGIS and ogr2ogr write "urn:ogc:def:crs:EPSG::25833"
when asked for the legacy form). Measured on a teaching data package (2026-10-12): every file names EPSG:25833 that way.
A lon/lat file is projected here to a UTM zone (projection.py): the scene's EPSG when it is a UTM zone, else the zone
of the file's centre; a projected "crs" is taken as it is. Z (a third coordinate) is dropped, as the shapefile reader
drops it.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .vector_io import Feature, Layer, VectorError
from . import projection

_KIND = {"Point": "point", "MultiPoint": "point", "LineString": "line", "MultiLineString": "line",
         "Polygon": "polygon", "MultiPolygon": "polygon"}


def crs_epsg(obj) -> tuple[int | None, str]:
    """(epsg, name) from a GeoJSON "crs" member; (4326, 'WGS84 lon/lat (RFC 7946)') when there is none."""
    crs = obj.get("crs") if isinstance(obj, dict) else None
    if not crs:
        return 4326, "WGS84 lon/lat (RFC 7946)"
    name = ""
    if isinstance(crs, dict):
        name = str((crs.get("properties") or {}).get("name") or crs.get("name") or "")
    else:
        name = str(crs)
    m = re.search(r"EPSG:{1,2}(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1)), name
    if "CRS84" in name.upper():
        return 4326, name
    return None, name


def _parts(gtype: str, coords):
    """(kind, parts) in vector_io's convention from a GeoJSON geometry's coordinates."""
    if gtype == "Point":
        return "point", [np.array([coords[:2]], dtype=np.float64)]
    if gtype == "MultiPoint":
        pts = [c[:2] for c in coords]
        return "point", [np.array(pts, dtype=np.float64)] if pts else []
    if gtype == "LineString":
        a = np.array([c[:2] for c in coords], dtype=np.float64)
        return "line", [a] if len(a) >= 2 else []
    if gtype == "MultiLineString":
        out = []
        for line in coords:
            a = np.array([c[:2] for c in line], dtype=np.float64)
            if len(a) >= 2:
                out.append(a)
        return "line", out
    if gtype in ("Polygon", "MultiPolygon"):
        polys = [coords] if gtype == "Polygon" else coords
        out = []
        for rings in polys:
            for ring in rings:
                a = np.array([c[:2] for c in ring], dtype=np.float64)
                if len(a) > 1 and np.allclose(a[0], a[-1]):
                    a = a[:-1]
                if len(a) >= 3:
                    out.append(a)
        return "polygon", out
    raise VectorError(f"GeoJSON geometry type {gtype} is not supported")


def _geometry_parts(geom):
    if not geom:
        return None, []
    gtype = geom.get("type")
    if gtype == "GeometryCollection":
        kinds = {}
        for g in geom.get("geometries") or []:
            k, p = _geometry_parts(g)
            if k:
                kinds.setdefault(k, []).extend(p)
        if not kinds:
            return None, []
        k = max(kinds, key=lambda kk: len(kinds[kk]))
        return k, kinds[k]
    if gtype not in _KIND:
        raise VectorError(f"GeoJSON geometry type {gtype} is not supported")
    return _parts(gtype, geom.get("coordinates") or [])


def _fields(features):
    """Field list with a type inferred from the values: any float -> F, only ints / bools -> N, else C."""
    seen = {}
    order = []
    for f in features:
        for k, v in (f.get("properties") or {}).items():
            if k not in seen:
                seen[k] = set()
                order.append(k)
            if v is None:
                continue
            seen[k].add("f" if isinstance(v, float) else "i" if isinstance(v, (bool, int)) else "c")
    out = []
    for k in order:
        s = seen[k]
        if s and s <= {"f", "i"}:
            out.append((k, "F", 6) if "f" in s else (k, "N", 0))
        else:
            out.append((k, "C", 0))
    return out


def read_geojson(path: str, target_epsg: int | None = None) -> Layer:
    """All features of a GeoJSON file as one Layer of the majority kind; lon/lat files projected to target_epsg (a UTM
    zone) or, without one, to the UTM zone of the file's centre."""
    if not os.path.isfile(path):
        raise VectorError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as fh:
        try:
            obj = json.load(fh)
        except json.JSONDecodeError as ex:
            raise VectorError(f"Not valid JSON: {ex}") from ex
    if not isinstance(obj, dict):
        raise VectorError("Not a GeoJSON object")
    t = obj.get("type")
    if t == "FeatureCollection":
        features = obj.get("features") or []
    elif t == "Feature":
        features = [obj]
    elif t in _KIND or t == "GeometryCollection":
        features = [{"type": "Feature", "geometry": obj, "properties": {}}]
    else:
        raise VectorError(f"GeoJSON type {t} is not supported")
    epsg, crs_name = crs_epsg(obj)
    fields = _fields(features)
    names = [f[0] for f in fields]
    raw = []
    kinds = {"point": 0, "line": 0, "polygon": 0}
    skipped = 0
    for i, f in enumerate(features):
        kind, parts = _geometry_parts(f.get("geometry"))
        if kind is None or not parts:
            skipped += 1
            continue
        kinds[kind] += 1
        fid = f.get("id")
        fid = int(fid) if isinstance(fid, int) and not isinstance(fid, bool) else i
        props = f.get("properties") or {}
        attrs = {n: props.get(n) for n in names}
        raw.append((kind, Feature(fid, parts, attrs)))
    if not raw:
        raise VectorError("The GeoJSON file has no features with geometry")
    kind = max(kinds, key=kinds.get)
    feats = [f for k, f in raw if k == kind]
    notes = []
    if skipped:
        notes.append(f"{skipped} features without geometry skipped")
    other = len(raw) - len(feats)
    if other:
        notes.append(f"{other} features of another geometry type than {kind} skipped")
    if epsg == 4326:
        allpts = np.vstack([p for f in feats for p in f.parts])
        lon_c, lat_c = float(allpts[:, 0].mean()), float(allpts[:, 1].mean())
        if abs(lon_c) > 180.0 or abs(lat_c) > 90.0:
            raise VectorError("The file has no CRS member and its coordinates are not longitude / latitude; "
                              "save it with a projected CRS (QGIS: RFC7946 off) or add a crs member")
        target = target_epsg if projection.utm_from_epsg(target_epsg) else projection.utm_epsg_for(lon_c, lat_c)
        for f in feats:
            f.parts = [np.column_stack(projection.lonlat_to_epsg(p[:, 0], p[:, 1], target)) for p in f.parts]
        notes.append(f"WGS84 lon/lat projected to EPSG:{target} inside Blender (Transverse Mercator series; ETRS89 and "
                     f"WGS84 treated as one datum, as GDAL's default does)")
        epsg, crs_name = target, f"EPSG:{target} (from {crs_name})"
    elif epsg is None:
        notes.append(f"CRS '{crs_name}' is not an EPSG code; the scene's CRS is assumed")
    allpts = np.vstack([p for f in feats for p in f.parts])
    bb = (float(allpts[:, 0].min()), float(allpts[:, 1].min()), float(allpts[:, 0].max()), float(allpts[:, 1].max()))
    return Layer(kind, feats, fields, epsg, crs_name, bb, os.path.basename(path), "utf-8", notes)
