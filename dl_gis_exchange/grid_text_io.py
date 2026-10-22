# DL GIS Exchange - grid_text_io.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""0.6.4: terrain grids as TEXT - ESRI ASCII grid (.asc) and XYZ point grids (.xyz) - read into the same Raster as a
GeoTIFF (geotiff_io.Raster), so meshing, scene origin, sampling and export stay one path. No GDAL.

ESRI ASCII grid: a header (ncols, nrows, xllcorner | xllcenter, yllcorner | yllcenter, cellsize - or dx and dy -,
NODATA_value optional; keys in any case, separated by spaces or tabs) and nrows lines of ncols values from north to
south. The CRS comes from a .prj beside the file: WKT (an EPSG authority, or an ETRS89 / WGS 84 / NAD83 UTM name), or the
older Arc/Info form ("Projection UTM", "Zone 30", "Datum ...ETRS89", "Units METERS"). Without one the CRS is unknown.
XYZ: one point per line, x y z separated by spaces, tabs, commas, or semicolons (then a decimal comma is allowed); an
optional header line. The points must lie on a regular lattice; they are read as CELL CENTRES (GDAL's XYZ convention);
missing points become holes. An XYZ file never states a CRS.
Feet: a .prj whose linear unit is not the metre is refused, as for GeoTIFF (0.6.1).
"""
from __future__ import annotations

import os
import re

import numpy as np

try:
    from . import geotiff_io
except ImportError:  # run as a plain script
    import geotiff_io  # type: ignore

Raster, GeoTiffError = geotiff_io.Raster, geotiff_io.GeoTiffError
MAX_CELLS = 200_000_000


def _numbers(body: str) -> np.ndarray:
    """All numbers of a blank-separated text block, flat, float64. numpy's C loadtxt: 0.7 s for the 8.5 million values
    of a Basque 1 m sheet, against 12.8 s for np.fromstring (measured 2026-10-21); rows of unequal length (some ASCII
    grids wrap a row over several lines) fall back to a plain split."""
    import io
    try:
        return np.loadtxt(io.StringIO(body), dtype=np.float64, ndmin=1).ravel()
    except ValueError:
        try:
            return np.array(body.split(), dtype=np.float64)
        except ValueError as ex:
            raise GeoTiffError(f"A value is not a number: {ex}") from None


# ------------------------------------------------------------------------------------------------ CRS from a .prj
def _utm_epsg(datum: str, zone: int, south: bool = False):
    d = datum.upper().replace("_", "").replace(" ", "")
    if "ETRS" in d and not south and 28 <= zone <= 38:
        return 25800 + zone                       # ETRS89 / UTM zone 28N-38N = EPSG:25828-25838
    if ("WGS84" in d or "WGS1984" in d) and 1 <= zone <= 60:
        return (32700 if south else 32600) + zone  # WGS 84 / UTM zone N: 32601-32660, S: 32701-32760
    if ("NAD83" in d or "NAD1983" in d) and not south and 1 <= zone <= 23:
        return 26900 + zone                       # NAD83 / UTM zone 1N-23N = EPSG:26901-26923
    return None


def epsg_from_prj(path: str):
    """(epsg or None, crs name, note). Raises GeoTiffError for a non-metre linear unit."""
    if not os.path.exists(path):
        return None, "", "no .prj beside the file: CRS unknown"
    text = open(path, "r", encoding="utf-8", errors="ignore").read().strip()
    if not text:
        return None, "", "empty .prj: CRS unknown"
    if text.upper().startswith(("PROJCS", "PROJCRS", "GEOGCS", "GEOGCRS", "COMPD_CS", "COMPOUNDCRS")):
        if text.upper().startswith(("GEOGCS", "GEOGCRS")):
            raise GeoTiffError("The grid's .prj describes a geographic CRS (degrees). Reproject it in QGIS to a metric CRS first.")
        name = (re.match(r'\w+\[\s*"([^"]*)"', text) or [None, ""])[1]
        units = re.findall(r'(?:UNIT|LENGTHUNIT)\[\s*"([^"]*)"\s*,\s*([0-9.eE+-]+)', text)
        lin = [(u, float(f)) for u, f in units if "degree" not in u.lower() and "radian" not in u.lower()]
        if lin and abs(lin[-1][1] - 1.0) > 1e-9:
            raise GeoTiffError(f"The grid's .prj uses {lin[-1][0]} units ({lin[-1][1]:g} m), not metres. DL GIS Exchange works in "
                               "metres: reproject it in QGIS to a metric CRS first.")
        codes = re.findall(r'(?:AUTHORITY|ID)\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]', text)
        if codes:
            return int(codes[-1]), name, f".prj: EPSG:{codes[-1]}"
        m = re.search(r"(ETRS[_ ]?(?:19)?89|WGS[_ ]?(?:19)?84|NAD[_ ]?(?:19)?83)[ _/]*UTM[_ ]zone[_ ](\d+)\s*([NS])", name, re.I)
        if m:
            e = _utm_epsg(m.group(1), int(m.group(2)), m.group(3).upper() == "S")
            if e:
                return e, name, f".prj names {name!r}: EPSG:{e}"
        return None, name, f".prj names {name!r} without an EPSG code: CRS unknown"
    kv = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            kv[parts[0].lower()] = parts[1].strip()
    units = kv.get("units", "METERS").upper()
    if units not in ("METERS", "METRES", "METER", "METRE"):
        raise GeoTiffError(f"The grid's .prj says Units {units}, not metres. Reproject it in QGIS to a metric CRS first.")
    proj = kv.get("projection", "").upper()
    if proj == "UTM" and "zone" in kv:
        z = int(float(kv["zone"]))
        e = _utm_epsg(kv.get("datum", ""), abs(z), south=z < 0)
        name = f"UTM zone {abs(z)}{'S' if z < 0 else 'N'}, {kv.get('datum', '')}"
        if e:
            return e, name, f"Arc/Info .prj ({name}): EPSG:{e}"
        return None, name, f"Arc/Info .prj ({name}): no EPSG code for this datum - CRS unknown"
    return None, proj.title(), f"Arc/Info .prj (Projection {proj or '?'}): CRS unknown"


# ------------------------------------------------------------------------------------------------ ESRI ASCII grid
def read_asc(path: str) -> Raster:
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        text = fh.read()
    head, pos = {}, 0
    lines = text.split("\n", 12)
    for i, line in enumerate(lines[:12]):
        parts = line.split()
        if len(parts) == 2 and re.match(r"^[A-Za-z_]+$", parts[0]):
            head[parts[0].lower()] = parts[1]
            pos += len(line) + 1
        else:
            break
    try:
        ncols, nrows = int(head["ncols"]), int(head["nrows"])
        if "cellsize" in head:
            dx = dy = float(head["cellsize"])
        else:
            dx, dy = float(head["dx"]), float(head["dy"])
    except (KeyError, ValueError) as ex:
        raise GeoTiffError(f"Not an ESRI ASCII grid header (missing or bad {ex}): ncols, nrows, cellsize expected") from None
    if dx <= 0 or dy <= 0:
        raise GeoTiffError("ESRI ASCII grid: the cell size must be positive")
    if ncols * nrows > MAX_CELLS:
        raise GeoTiffError(f"ESRI ASCII grid of {ncols:,} x {nrows:,} cells is too large")
    if "xllcorner" in head:
        x0 = float(head["xllcorner"])
    elif "xllcenter" in head:
        x0 = float(head["xllcenter"]) - dx / 2.0
    else:
        raise GeoTiffError("ESRI ASCII grid: neither xllcorner nor xllcenter in the header")
    if "yllcorner" in head:
        y0 = float(head["yllcorner"])
    elif "yllcenter" in head:
        y0 = float(head["yllcenter"]) - dy / 2.0
    else:
        raise GeoTiffError("ESRI ASCII grid: neither yllcorner nor yllcenter in the header")
    body = text[pos:]
    decimal_comma = "," in body[:20000]
    if decimal_comma:                    # an ASCII grid separates by blanks, so a comma is a decimal comma (Basque 2017 grids)
        body = body.replace(",", ".")
    vals = _numbers(body)
    if vals.size < ncols * nrows:
        raise GeoTiffError(f"ESRI ASCII grid: {vals.size:,} values for {ncols} x {nrows} = {ncols * nrows:,} cells")
    z = vals[: ncols * nrows].reshape(nrows, ncols).astype(np.float32)
    notes = ["ESRI ASCII grid" + (" (lower-left given as a cell centre)" if "xllcenter" in head else "")
             + (", decimal commas" if decimal_comma else "")]
    nodata, nodata_from = None, ""
    if "nodata_value" in head:
        nodata, nodata_from = float(head["nodata_value"]), "tag"
        z[z == np.float32(nodata)] = np.nan
    else:
        for s in geotiff_io.NODATA_SENTINELS:
            hit = z == np.float32(s)
            if hit.any():
                z[hit] = np.nan
                nodata, nodata_from = s, "sentinel"
                notes.append(f"no NODATA_value; {int(hit.sum())} cells at {s:g} treated as nodata")
                break
    epsg, crs_name, note = epsg_from_prj(os.path.splitext(path)[0] + ".prj")
    notes.append(note)
    return Raster(z=z, ul_E=x0, ul_N=y0 + nrows * dy, dx=dx, dy=dy, epsg=epsg, crs_name=crs_name, raster_type="area",
                  nodata=nodata, nodata_from=nodata_from, source=os.path.basename(path), notes=notes)


# ------------------------------------------------------------------------------------------------ XYZ
def read_xyz(path: str) -> Raster:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lines = text.lstrip("﻿").split("\n", 2)
    first = lines[0].strip()
    header = bool(first) and not re.match(r"^[\s+\-]*[0-9.]", first)
    body = lines[1] + "\n" + (lines[2] if len(lines) > 2 else "") if header else text.lstrip("﻿")
    sample = next((ln for ln in body.split("\n", 5) if ln.strip()), "")
    if sample.count(";") == 2:
        body = body.replace(",", ".").replace(";", " ")             # semicolons + decimal commas
    elif sample.count(",") == 2:
        body = body.replace(",", " ")
    vals = _numbers(body)
    if vals.size == 0 or vals.size % 3:
        raise GeoTiffError(f"XYZ: expected three numbers per line (x y z); found {vals.size:,} numbers in total")
    x, y, zz = vals[0::3], vals[1::3], vals[2::3]
    ux = np.unique(np.round(x, 6))
    uy = np.unique(np.round(y, 6))
    if ux.size < 2 or uy.size < 2:
        raise GeoTiffError("XYZ: the points do not span a grid (one row or one column)")
    dx, dy = float(np.diff(ux).min()), float(np.diff(uy).min())
    fc, fr = (x - ux[0]) / dx, (uy[-1] - y) / dy
    c, r = np.rint(fc).astype(np.int64), np.rint(fr).astype(np.int64)
    off = max(float(np.abs(fc - c).max()), float(np.abs(fr - r).max()))
    if off > 0.01:
        raise GeoTiffError(f"XYZ: the points are not on a regular grid (a point lies {off:.2f} cells off the lattice "
                           f"of {dx:g} x {dy:g}). Scattered points need gridding first, e.g. in QGIS (Interpolation).")
    cols, rows = int(c.max()) + 1, int(r.max()) + 1
    if cols * rows > MAX_CELLS:
        raise GeoTiffError(f"XYZ grid of {cols:,} x {rows:,} cells is too large")
    flat = r * cols + c
    if np.unique(flat).size != flat.size:
        raise GeoTiffError("XYZ: two or more points share one grid cell (duplicates)")
    z = np.full(rows * cols, np.nan, dtype=np.float32)
    z[flat] = zz.astype(np.float32)
    z = z.reshape(rows, cols)
    notes = [f"XYZ: {zz.size:,} points read as cell centres of a {dx:g} x {dy:g} grid"
             + (f"; {rows * cols - zz.size:,} cells without a point left empty" if rows * cols > zz.size else "")]
    nodata, nodata_from = None, ""
    for s in geotiff_io.NODATA_SENTINELS:
        hit = z == np.float32(s)
        if hit.any():
            z[hit] = np.nan
            nodata, nodata_from = s, "sentinel"
            notes.append(f"{int(hit.sum())} points at {s:g} treated as nodata")
            break
    notes.append("XYZ states no CRS")
    return Raster(z=z, ul_E=float(ux[0]) - dx / 2.0, ul_N=float(uy[-1]) + dy / 2.0, dx=dx, dy=dy, epsg=None, crs_name="",
                  raster_type="area", nodata=nodata, nodata_from=nodata_from, source=os.path.basename(path), notes=notes)


# ------------------------------------------------------------------------------------------------ one entry point
TEXT_EXT = (".asc", ".xyz")


def read_raster(path: str) -> Raster:
    """GeoTIFF, ESRI ASCII grid or XYZ by the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".asc":
        return read_asc(path)
    if ext == ".xyz":
        return read_xyz(path)
    return geotiff_io.read_geotiff(path)
