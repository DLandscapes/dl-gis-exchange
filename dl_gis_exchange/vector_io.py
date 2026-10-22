# DL GIS Exchange - vector_io.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read ESRI shapefiles (.shp + .dbf + .prj) into plain Python features, without GDAL.

Built on pyshp (MIT, Joel Lawhead and contributors, https://github.com/GeospatialPython/pyshp), bundled as a wheel.
Measured 2026-10-10 (research note 002):
  - .dbf field names and text can be cp1252 (Norwegian ø): the .cpg file decides, else utf-8, cp1252, latin-1 are tried;
  - ring winding in shapefiles is unreliable (GDAL warns and autocorrects), so a hole is NOT decided here by winding:
    polygon rings are handed to Blender's constrained Delaunay, which nests them geometrically (vector_mesh.py).
Coordinates stay in the file's CRS (map units); the add-on subtracts the scene origin when it builds the mesh.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

# ESRI WKT names -> EPSG for the CRS this tool meets most (no AUTHORITY in ESRI .prj files)
PRJ_NAMES = {
    "ETRS_1989_UTM_Zone_32N": 25832, "ETRS_1989_UTM_Zone_33N": 25833, "ETRS_1989_UTM_Zone_35N": 25835,
    "ETRS89_UTM_zone_32N": 25832, "ETRS89_UTM_zone_33N": 25833, "ETRS89_UTM_zone_35N": 25835,
    "WGS_1984_UTM_Zone_32N": 32632, "WGS_1984_UTM_Zone_33N": 32633, "WGS_1984_UTM_Zone_35N": 32635,
    "ETRS_1989_LAEA": 3035, "ETRS89_LAEA_Europe": 3035,
}


class VectorError(ValueError):
    """A problem the user can act on; the message says what to do."""


@dataclass
class Feature:
    fid: int
    parts: list                 # list of numpy (n, 2) arrays in map units: points -> one part of n points;
                                # lines -> one part per polyline; polygons -> one part per RING (closing point removed)
    attrs: dict                 # dbf record as {field name: value}


@dataclass
class Layer:
    kind: str                   # "point" | "line" | "polygon"
    features: list
    fields: list                # [(name, type letter C/N/F/L/D, decimals)]
    epsg: int | None
    crs_name: str
    bbox: tuple                 # (xmin, ymin, xmax, ymax) in map units
    source: str
    encoding: str
    notes: list = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return sum(len(p) for f in self.features for p in f.parts)


def _open(path: str):
    import shapefile
    enc = None
    cpg = os.path.splitext(path)[0] + ".cpg"
    if os.path.exists(cpg):
        try:
            enc = open(cpg, "r", encoding="ascii", errors="ignore").read().strip() or None
        except OSError:
            enc = None
    tried = []
    for e in ([enc] if enc else []) + ["utf-8", "cp1252", "latin-1"]:
        if e in tried:
            continue
        tried.append(e)
        try:
            rd = shapefile.Reader(path, encoding=e)
            rd.fields              # the dbf header
            for _ in rd.iterRecords():   # EVERY record: an OSM file declared utf-8 held one Latin-1 record (2026-10-10)
                pass
            return rd, e
        except Exception:      # noqa: BLE001 - try the next encoding
            continue
    # nothing decodes every record strictly: keep the declared (or utf-8) encoding and replace the bad characters
    e = enc or "utf-8"
    try:
        rd = shapefile.Reader(path, encoding=e, encodingErrors="replace")
        rd.fields
        return rd, e + " (bad characters replaced)"
    except Exception as ex:    # noqa: BLE001
        raise VectorError(f"Cannot read the attribute table of {os.path.basename(path)} (encodings tried: {', '.join(tried)}): {ex}")


def read_prj(path: str) -> tuple[int | None, str]:
    """(EPSG code or None, CRS name) from the .prj beside the shapefile."""
    prj = os.path.splitext(path)[0] + ".prj"
    if not os.path.exists(prj):
        return None, ""
    try:
        wkt = open(prj, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return None, ""
    name = ""
    m = re.match(r'\s*(PROJCS|GEOGCS|PROJCRS|GEOGCRS)\["([^"]+)"', wkt)
    if m:
        name = m.group(2)
    # an explicit authority wins (the LAST one in the string is the top-level CRS)
    auth = re.findall(r'AUTHORITY\["EPSG",\s*"?(\d+)"?\]', wkt)
    if auth:
        return int(auth[-1]), name
    ids = re.findall(r'ID\["EPSG",\s*(\d+)\]', wkt)
    if ids:
        return int(ids[-1]), name
    if name in PRJ_NAMES:
        return PRJ_NAMES[name], name
    m2 = re.search(r"UTM[_ ]Zone[_ ](\d+)N", name, re.IGNORECASE)
    if m2 and ("ETRS" in name.upper()):
        return 25800 + int(m2.group(1)), name
    if m2 and ("WGS" in name.upper()):
        return 32600 + int(m2.group(1)), name
    return None, name


def read_shapefile(path: str) -> Layer:
    """All features of a shapefile; the kind from the shape type; Z and M dropped (they come back as options later)."""
    import shapefile
    rd, enc = _open(path)
    st = rd.shapeType
    if st in (1, 11, 21, 8, 18, 28):
        kind = "point"
    elif st in (3, 13, 23):
        kind = "line"
    elif st in (5, 15, 25):
        kind = "polygon"
    else:
        raise VectorError(f"Shape type {rd.shapeTypeName} is not supported (points, lines and polygons are)")
    fields = [(f[0], f[1], f[3]) for f in rd.fields if f[0] != "DeletionFlag"]
    names = [f[0] for f in fields]
    feats = []
    notes = []
    skipped = 0
    for i, sr in enumerate(rd.iterShapeRecords()):
        s = sr.shape
        if s.shapeType == 0 or not s.points:
            skipped += 1
            continue
        pts = np.asarray(s.points, dtype=np.float64)[:, :2]
        if kind == "point":
            parts = [pts]
        else:
            bounds = list(s.parts) + [len(pts)]
            parts = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                part = pts[a:b]
                if kind == "polygon" and len(part) > 1 and np.allclose(part[0], part[-1]):
                    part = part[:-1]
                if (kind == "polygon" and len(part) >= 3) or (kind == "line" and len(part) >= 2):
                    parts.append(part)
            if not parts:
                skipped += 1
                continue
        attrs = {}
        for n, v in zip(names, sr.record):
            if isinstance(v, bytes):
                v = v.decode(enc.split(" ")[0], "replace")
            attrs[n] = v
        feats.append(Feature(i, parts, attrs))
    if skipped:
        notes.append(f"{skipped} empty features skipped")
    if "replaced" in enc:
        notes.append(f"attribute text in a mixed encoding: read as {enc}")
    epsg, crs_name = read_prj(path)
    if epsg is None:
        notes.append("CRS not read from the .prj (unknown name); the scene's CRS is assumed")
    bb = tuple(float(v) for v in rd.bbox) if rd.bbox else (0.0, 0.0, 0.0, 0.0)
    return Layer(kind, feats, fields, epsg, crs_name, bb, os.path.basename(path), enc, notes)
