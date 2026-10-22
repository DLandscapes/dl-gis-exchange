# DL GIS Exchange - geotiff_io.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read a GeoTIFF raster into a numpy array plus its georeference, without GDAL.

Built on tifffile (BSD-3-Clause, Christoph Gohlke), bundled as a wheel, with the pure-Python LZW and
floating-point-predictor decoders from tiffcodecs.py plugged in, so the files QGIS and GDAL write by default
(LZW, predictor 3) open too - measured 2026-10-09, probe 01.

ONE convention, everywhere in this extension (the GeoTIFF / GDAL one):
    ul_E, ul_N  = the OUTER upper-left corner of the upper-left cell (north edge, west edge), in CRS units
    dx, dy      = cell size, both positive; row 0 is the northernmost row
    cell (r, c) covers  E = ul_E + c*dx .. ul_E + (c+1)*dx,  N = ul_N - (r+1)*dy .. ul_N - r*dy
    its centre is       E = ul_E + (c + 0.5)*dx,             N = ul_N - (r + 0.5)*dy
PixelIsPoint rasters (GTRasterTypeGeoKey 1025 = 2) are converted to this convention on reading: their tiepoint
names the CENTRE of the upper-left cell, so the corner lies half a cell further north-west (OGC 19-008r4 §7.2.2).
Nodata becomes NaN.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

try:
    from . import tiffcodecs
except ImportError:  # run as a plain script (probes)
    import tiffcodecs  # type: ignore

# sentinel values treated as nodata when the file carries NO GDAL_NODATA tag (same sweep as DL-TerrainSlicer)
NODATA_SENTINELS = (-9999.0, -32767.0, -32768.0, -3.4028235e38, 3.4028235e38)

# 0.6.1: EPSG linear units (GeoTIFF ProjLinearUnitsGeoKey 3076 / VerticalUnitsGeoKey 4099) - metres per unit
METRE = 9001
FOOT_UNITS = {9002: ("foot", 0.3048), 9003: ("US survey foot", 1200.0 / 3937.0), 9005: ("Clarke's foot", 0.3047972654),
              9014: ("fathom", 1.8288), 9030: ("nautical mile", 1852.0), 9036: ("kilometre", 1000.0)}


@dataclass
class Raster:
    z: np.ndarray                 # float32, shape (rows, cols), NaN = nodata
    ul_E: float                   # outer upper-left corner, east
    ul_N: float                   # outer upper-left corner, north
    dx: float
    dy: float
    epsg: int | None              # projected CRS code, or None when the file does not say
    crs_name: str = ""            # GTCitationGeoKey when present
    raster_type: str = "area"     # "area" | "point" as the file declared it
    nodata: float | None = None   # the value that became NaN (tag or sentinel), None if none
    nodata_from: str = ""         # "tag" | "sentinel" | ""
    source: str = ""              # file name
    notes: list[str] = field(default_factory=list)
    # 0.6.3: a projected CRS WITHOUT an EPSG code is kept as the file's own GeoKeys, so an export can write it back:
    # {"dir": [34735 shorts], "doubles": [34736], "ascii": "34737 text"}; None when the file names an EPSG code
    geokeys: dict | None = None

    @property
    def rows(self) -> int:
        return int(self.z.shape[0])

    @property
    def cols(self) -> int:
        return int(self.z.shape[1])

    def centre_E(self, c) -> np.ndarray:
        return self.ul_E + (np.asarray(c, dtype=np.float64) + 0.5) * self.dx

    def centre_N(self, r) -> np.ndarray:
        return self.ul_N - (np.asarray(r, dtype=np.float64) + 0.5) * self.dy


class GeoTiffError(ValueError):
    """A problem the user can act on; the message says what to do."""


def read_geotiff(path: str) -> Raster:
    """Read band 1 of a GeoTIFF as float32 with NaN for nodata, plus its georeference."""
    import tifffile
    tiffcodecs.install()

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        meta = dict(tf.geotiff_metadata or {})
        if not meta:
            raise GeoTiffError("No GeoTIFF tags found (tag 34735 missing): this is a plain TIFF. "
                               "Save it with georeference from QGIS (Export > Save As > GeoTIFF).")
        nd_tag = page.tags.get(42113)
        raw_keys = {t: page.tags.get(t) for t in (34735, 34736, 34737)}     # 0.6.3: the GeoKeys as written
        raw_keys = {t: (v.value if v is not None else None) for t, v in raw_keys.items()}
        data = page.asarray()

    notes: list[str] = []
    # ---- the array: first band only, float32
    if data.ndim == 3:
        if page.planarconfig == 2 or data.shape[0] < data.shape[-1]:
            data = data[0]
        else:
            data = data[..., 0]
        notes.append("multi-band raster: band 1 used")
    if data.ndim != 2:
        raise GeoTiffError(f"Unexpected raster shape {data.shape}")
    z = data.astype(np.float32, copy=True)

    # ---- nodata
    nodata = None
    nodata_from = ""
    if nd_tag is not None:
        try:
            nodata = float(str(nd_tag.value).strip().strip("\x00"))
            nodata_from = "tag"
        except ValueError:
            notes.append(f"GDAL_NODATA tag unreadable: {nd_tag.value!r}")
    if nodata is not None:
        if np.isnan(nodata):
            pass  # NaN already is nodata
        else:
            z[z == np.float32(nodata)] = np.nan
    else:
        for s in NODATA_SENTINELS:
            hit = z == np.float32(s)
            if hit.any():
                z[hit] = np.nan
                nodata, nodata_from = s, "sentinel"
                notes.append(f"no nodata tag; {int(hit.sum())} cells at sentinel {s:g} treated as nodata")
                break

    # ---- georeference
    model_type = int(meta.get("GTModelTypeGeoKey", 0) or 0)
    raster_key = int(meta.get("GTRasterTypeGeoKey", 1) or 1)
    epsg = meta.get("ProjectedCSTypeGeoKey")
    epsg = int(epsg) if epsg not in (None, 0, 32767) else None
    geog = meta.get("GeographicTypeGeoKey")
    # 0.6.1: GTModelTypeGeoKey decides (1 projected, 2 geographic). Only a file WITHOUT a model type counts as degrees
    # because of a geographic code alone. 0.4.0-0.6.0 refused a user-defined projected CRS on a geographic base as
    # "degrees" (the USGS Alaska IfSAR DTM, NAD83 / Alaska Albers, test-site gate 2026-10-20).
    if model_type == 2 or (model_type == 0 and epsg is None and geog not in (None, 0, 32767)):
        raise GeoTiffError(f"Raster is in degrees (geographic CRS{' EPSG:' + str(geog) if geog else ''}). "
                           "Reproject it in QGIS to a metric CRS such as EPSG:25833 first.")
    if epsg is None:
        notes.append("CRS not stated in the file (ProjectedCSTypeGeoKey missing or user-defined)")
    # 0.6.1: the CRS's linear unit. The extension works in metres throughout (scene origin, sampling, export), so a
    # raster whose coordinates are in feet is refused, not scaled: 0.6.0 imported the USGS Alaska lidar (State Plane,
    # US survey feet) as if in metres - 3.28 times too large and too high (test-site gate 2026-10-20).
    lin = int(meta.get("ProjLinearUnitsGeoKey", 0) or 0)
    if lin == 32767:
        size = meta.get("ProjLinearUnitSizeGeoKey")
        if size is None or abs(float(size) - 1.0) > 1e-9:
            raise GeoTiffError(f"The raster's coordinates are in a user-defined unit ({size} m). DL GIS Exchange works in "
                               "metres: reproject it in QGIS to a metric CRS (Raster > Projections > Warp) first.")
    elif lin in FOOT_UNITS:
        name, m = FOOT_UNITS[lin]
        raise GeoTiffError(f"The raster's coordinates are in {name} units ({m:g} m), not metres. DL GIS Exchange works in "
                           "metres: reproject it in QGIS to a metric CRS (Raster > Projections > Warp, e.g. the local "
                           "UTM zone) first.")
    elif lin not in (0, METRE):
        raise GeoTiffError(f"The raster's linear unit (EPSG unit {lin}) is not metres. Reproject it in QGIS to a metric CRS first.")
    elif lin == 0 and model_type == 1:
        notes.append("linear unit not stated in the file: metres assumed")
    # 0.6.1: heights in feet under a metric CRS are converted (only z changes; the grid stays as it is)
    vu = int(meta.get("VerticalUnitsGeoKey", 0) or 0)
    if vu in FOOT_UNITS:
        vname, vm = FOOT_UNITS[vu]
        z = z * np.float32(vm)
        notes.append(f"heights converted from {vname} units to metres (x {vm:.9g}; VerticalUnitsGeoKey) - feet in the file")
    elif vu not in (0, METRE, 32767):
        notes.append(f"vertical unit EPSG {vu} not recognised: heights taken as metres")
    crs_name = str(meta.get("GTCitationGeoKey", "") or meta.get("PCSCitationGeoKey", "") or "")

    scale = meta.get("ModelPixelScale")
    tie = meta.get("ModelTiepoint")
    trans = meta.get("ModelTransformation")
    if scale is not None and tie is not None:
        scale = [float(v) for v in scale]
        tie = [float(v) for v in tie]
        i, j = tie[0], tie[1]
        dx, dy = abs(scale[0]), abs(scale[1])
        if dx == 0 or dy == 0:
            raise GeoTiffError("ModelPixelScale has a zero cell size")
        # tiepoint (i, j) -> (E, N): the point at raster coordinate (i, j) sits at (E, N)
        ul_E = tie[3] - i * dx
        ul_N = tie[4] + j * dy
    elif trans is not None:
        m = [float(v) for v in np.asarray(trans).ravel()]
        if len(m) != 16:
            raise GeoTiffError("ModelTransformation is not a 4x4 matrix")
        a, b, d_, e, f, h = m[0], m[1], m[3], m[4], m[5], m[7]
        if abs(b) > 1e-12 or abs(e) > 1e-12:
            raise GeoTiffError("Rotated raster (ModelTransformation has off-diagonal terms): "
                               "warp it north-up in QGIS first.")
        dx, dy = abs(a), abs(f)
        ul_E, ul_N = d_, h
        if f > 0:
            notes.append("ModelTransformation with positive dy: rows run south to north; array flipped")
            z = z[::-1]
    else:
        raise GeoTiffError("No georeference (neither ModelTiepoint + ModelPixelScale nor ModelTransformation).")

    raster_type = "point" if raster_key == 2 else "area"
    if raster_type == "point":
        # the tiepoint named the CENTRE of cell (0,0): move to the outer corner
        ul_E -= dx / 2.0
        ul_N += dy / 2.0
        notes.append("PixelIsPoint raster: corner taken half a cell north-west of the tiepoint")

    geokeys = None
    if epsg is None and model_type == 1 and raw_keys.get(34735):
        geokeys = {"dir": [int(v) for v in raw_keys[34735]],
                   "doubles": [float(v) for v in (raw_keys.get(34736) or ())],
                   "ascii": str(raw_keys.get(34737) or "")}
        notes.append("user-defined projected CRS: its GeoKeys are kept and written back on export")
    return Raster(z=z, ul_E=float(ul_E), ul_N=float(ul_N), dx=float(dx), dy=float(dy), epsg=epsg,
                  crs_name=crs_name, raster_type=raster_type, nodata=nodata, nodata_from=nodata_from,
                  source=os.path.basename(path), notes=notes, geokeys=geokeys)


def window(r: Raster, e_min: float, e_max: float, n_min: float, n_max: float) -> tuple[int, int, int, int] | None:
    """Row/column slice (r0, r1, c0, c1) of the cells whose CENTRES lie inside the box, or None if no overlap."""
    c0 = int(np.ceil((e_min - r.ul_E) / r.dx - 0.5))
    c1 = int(np.floor((e_max - r.ul_E) / r.dx - 0.5)) + 1
    r0 = int(np.ceil((r.ul_N - n_max) / r.dy - 0.5))
    r1 = int(np.floor((r.ul_N - n_min) / r.dy - 0.5)) + 1
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, r.cols), min(r1, r.rows)
    if c1 - c0 < 2 or r1 - r0 < 2:
        return None
    return r0, r1, c0, c1


def crop(r: Raster, r0: int, r1: int, c0: int, c1: int) -> Raster:
    """A new Raster for the slice, with the corner moved accordingly."""
    out = Raster(z=np.ascontiguousarray(r.z[r0:r1, c0:c1]), ul_E=r.ul_E + c0 * r.dx, ul_N=r.ul_N - r0 * r.dy,
                 dx=r.dx, dy=r.dy, epsg=r.epsg, crs_name=r.crs_name, raster_type=r.raster_type, nodata=r.nodata,
                 nodata_from=r.nodata_from, source=r.source, notes=list(r.notes), geokeys=r.geokeys)
    out.notes.append(f"clipped to rows {r0}..{r1 - 1}, columns {c0}..{c1 - 1} of {r.rows} x {r.cols}")
    return out


def stride(r: Raster, step: int) -> Raster:
    """Every step-th cell, cell size multiplied; the kept cells keep their own centres (corner stays)."""
    if step <= 1:
        return r
    out = Raster(z=np.ascontiguousarray(r.z[::step, ::step]), ul_E=r.ul_E, ul_N=r.ul_N, dx=r.dx, dy=r.dy,
                 epsg=r.epsg, crs_name=r.crs_name, raster_type=r.raster_type, nodata=r.nodata,
                 nodata_from=r.nodata_from, source=r.source, notes=list(r.notes), geokeys=r.geokeys)
    # kept cell k sits at original index k*step; its centre = ul + (k*step + 0.5)*d. Express that with a new
    # cell size step*d and a shifted corner so that centre = ul' + (k + 0.5)*d':
    #   ul' + (k + 0.5)*step*d = ul + (k*step + 0.5)*d  ->  ul' = ul + 0.5*d - 0.5*step*d
    out.ul_E = r.ul_E + 0.5 * r.dx - 0.5 * step * r.dx
    out.ul_N = r.ul_N - 0.5 * r.dy + 0.5 * step * r.dy
    out.dx, out.dy = r.dx * step, r.dy * step
    out.notes.append(f"step {step}: every {step}. cell, cell size {out.dx:g} x {out.dy:g}")
    return out


# ====================================================================================================== writing
USER_DEFINED = 32767   # GeoTIFF: "user-defined" code when the CRS is not known (OGC 19-008r4, B.4)


def _passthrough_keys(geokeys: dict):
    """0.6.3: the source's GeoKey directory with GTRasterTypeGeoKey forced to PixelIsArea (this writer always writes the
    outer-corner convention), plus its double and text parameter arrays - returned as (directory, doubles, text)."""
    d = [int(v) for v in geokeys["dir"]]
    head, entries = d[:4], [d[i:i + 4] for i in range(4, 4 + 4 * d[3], 4)]
    if any(e[0] == 1025 for e in entries):
        entries = [[1025, 0, 1, 1] if e[0] == 1025 else e for e in entries]
    else:
        entries = sorted(entries + [[1025, 0, 1, 1]], key=lambda e: e[0])
    out = [head[0], head[1], head[2], len(entries)]
    for e in entries:
        out.extend(e)
    return out, [float(v) for v in geokeys.get("doubles") or ()], str(geokeys.get("ascii") or "")


def write_geotiff(path: str, z: np.ndarray, ul_E: float, ul_N: float, dx: float, dy: float, epsg: int | None,
                  nodata: float = -9999.0, compression: str = "deflate", geokeys: dict | None = None) -> dict:
    """Write a float32 GeoTIFF (PixelIsArea, north-up) with the five tags QGIS and SAGA need, through tifffile.

    Measured 2026-10-09 (probe 01, GDAL 3.13.1 via QGIS 3.44): tags 33550 + 33922 + 34735 (1024, 1025, 3072, 3076)
    + 42113 give EPSG, transform and nodata back exactly. The floating-point predictor is NOT written: tifffile
    cannot encode it without imagecodecs. NaN cells become `nodata`.
    """
    import tifffile
    a = np.asarray(z, dtype=np.float32)
    if a.ndim != 2:
        raise GeoTiffError(f"expected a 2-D array, got shape {a.shape}")
    a = np.where(np.isnan(a), np.float32(nodata), a).astype(np.float32)
    code = int(epsg) if epsg else USER_DEFINED
    keys = [(1024, 0, 1, 1),        # GTModelTypeGeoKey: projected
            (1025, 0, 1, 1),        # GTRasterTypeGeoKey: PixelIsArea
            (3072, 0, 1, code),     # ProjectedCSTypeGeoKey: EPSG code or user-defined
            (3076, 0, 1, 9001)]     # ProjLinearUnitsGeoKey: metre
    gk = [1, 1, 0, len(keys)]
    for k in keys:
        gk.extend(k)
    doubles, text, crs = [], "", f"EPSG:{code}" if epsg else "user-defined (no parameters)"
    if not epsg and geokeys:                    # 0.6.3: write the source's own CRS back (no EPSG code exists for it)
        gk, doubles, text = _passthrough_keys(geokeys)
        crs = "user-defined (the source's GeoKeys)"
    extratags = [
        (33550, "d", 3, (float(dx), float(dy), 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, float(ul_E), float(ul_N), 0.0), True),
        (34735, "H", len(gk), tuple(gk), True),
        (42113, "s", 0, repr(float(nodata)) if float(nodata) != int(nodata) else str(int(nodata)), True),
    ]
    if doubles:
        extratags.append((34736, "d", len(doubles), tuple(doubles), True))
    if text:
        extratags.append((34737, "s", 0, text, True))
    kw = dict(extratags=extratags, metadata=None, photometric="minisblack")
    if compression in ("deflate", "zlib"):
        kw.update(compression="zlib", predictor=False)
    elif compression not in (None, "", "none"):
        raise GeoTiffError(f"compression {compression!r} not supported (deflate or none)")
    tifffile.imwrite(path, a, **kw)
    return {"rows": int(a.shape[0]), "cols": int(a.shape[1]), "epsg": code, "crs": crs, "nodata": float(nodata),
            "n_nodata": int((a == np.float32(nodata)).sum()), "bytes": os.path.getsize(path)}


def grid_for_box(e_min: float, e_max: float, n_min: float, n_max: float, cell: float,
                 align: bool = True, margin_cells: int = 0) -> tuple[float, float, int, int]:
    """(ul_E, ul_N, rows, cols) of the raster that covers the box at `cell`.

    align=True snaps the corner outward to whole multiples of the cell size in map coordinates, so the raster lines
    up with other rasters of that cell size (hoydedata tiles start on whole metres)."""
    if cell <= 0:
        raise GeoTiffError("cell size must be positive")
    if e_max <= e_min or n_max <= n_min:
        raise GeoTiffError("the bounding box has no area")
    if align:
        ul_E = np.floor(e_min / cell) * cell
        ul_N = np.ceil(n_max / cell) * cell
    else:
        ul_E, ul_N = round(e_min, 3), round(n_max, 3)   # object positions are float32: keep whole millimetres
    ul_E -= margin_cells * cell
    ul_N += margin_cells * cell
    cols = int(np.ceil((e_max - ul_E) / cell - 1e-9)) + margin_cells
    rows = int(np.ceil((ul_N - n_min) / cell - 1e-9)) + margin_cells
    return float(ul_E), float(ul_N), max(rows, 1), max(cols, 1)
