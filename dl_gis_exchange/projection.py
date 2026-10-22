# DL GIS Exchange - projection.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Geographic lon/lat -> UTM easting/northing without GDAL or pyproj (0.5.0, for GeoJSON files that follow RFC 7946
and therefore carry WGS84 longitude / latitude).

Transverse Mercator after Krüger's series as given by Karney (2011, "Transverse Mercator with an accuracy of a few
nanometers", J. Geodesy 85, 475-485; the same formulation as GeographicLib's and PROJ's "tmerc" algorithm "poder/engsager"
family to sixth order in n). Within a UTM zone and its neighbours the series is exact to well under a millimetre; the gate
compares it with GDAL on real data (rule V7).

Datum: ETRS89 (EPSG 258xx, GRS80 ellipsoid) and WGS84 (EPSG 326xx / 327xx) are treated as one datum here, as GDAL's
default pipeline does without a time-dependent transformation; the two differ by a few decimetres today (plate motion
since 1989), which matters for surveying, not for a terrain model built from 1 m rasters. Said in the import's note.
"""
from __future__ import annotations

import math

import numpy as np

GRS80 = (6378137.0, 1.0 / 298.257222101)
WGS84 = (6378137.0, 1.0 / 298.257223563)


def utm_from_epsg(epsg: int | None):
    """(zone, is_north, (a, f)) for a UTM EPSG code, or None when the code is not a UTM zone."""
    if not epsg:
        return None
    e = int(epsg)
    if 25828 <= e <= 25838:
        return e - 25800, True, GRS80
    if 32601 <= e <= 32660:
        return e - 32600, True, WGS84
    if 32701 <= e <= 32760:
        return e - 32700, False, WGS84
    return None


def utm_epsg_for(lon: float, lat: float) -> int:
    """The WGS84 UTM zone EPSG code (326xx north, 327xx south) that holds a lon/lat; Norway's special zones ignored."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def tm_forward(lon_deg, lat_deg, lon0_deg: float, a: float, f: float, k0: float = 0.9996, fe: float = 500000.0, fn: float = 0.0):
    """Transverse Mercator forward (Krüger series, 6th order): arrays of easting, northing in metres."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    n = f / (2.0 - f)
    n2, n3, n4, n5, n6 = n ** 2, n ** 3, n ** 4, n ** 5, n ** 6
    A = a / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0)
    alpha = (
        n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0 - 127.0 * n5 / 288.0 + 7891.0 * n6 / 37800.0,
        13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0 + 281.0 * n5 / 630.0 - 1983433.0 * n6 / 1935360.0,
        61.0 * n3 / 240.0 - 103.0 * n4 / 140.0 + 15061.0 * n5 / 26880.0 + 167603.0 * n6 / 181440.0,
        49561.0 * n4 / 161280.0 - 179.0 * n5 / 168.0 + 6601661.0 * n6 / 7257600.0,
        34729.0 * n5 / 80640.0 - 3418889.0 * n6 / 1995840.0,
        212378941.0 * n6 / 319334400.0,
    )
    e = math.sqrt(f * (2.0 - f))
    lam = lon - math.radians(lon0_deg)
    s = np.sin(lat)
    t = np.sinh(np.arctanh(s) - e * np.arctanh(e * s))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1.0 + t * t))
    xi = xi_p.copy()
    eta = eta_p.copy()
    for j, aj in enumerate(alpha, start=1):
        xi += aj * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta += aj * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    return fe + k0 * A * eta, fn + k0 * A * xi


def lonlat_to_epsg(lon_deg, lat_deg, epsg: int):
    """Project arrays of lon/lat to a UTM EPSG code; raises ValueError for a non-UTM code."""
    z = utm_from_epsg(epsg)
    if z is None:
        raise ValueError(f"EPSG:{epsg} is not a UTM zone; only UTM targets (EPSG 258xx, 326xx, 327xx) are built in")
    zone, north, (a, f) = z
    lon0 = zone * 6.0 - 183.0
    return tm_forward(lon_deg, lat_deg, lon0, a, f, 0.9996, 500000.0, 0.0 if north else 10000000.0)
