# DL GIS Exchange — credits

DL GIS Exchange is made by Digital Landscapes (https://digital-landscapes.com) and released under the GNU General Public
License, version 3 or (at your option) any later version — see `LICENSE.txt`. It stands on open-source work by others;
thank you.

Copyright 2026 Marc Ihle (Digital Landscapes).

## Bundled, unmodified, in `wheels\`
- **tifffile** 2026.3.3 — Copyright (c) 2008-2026, Christoph Gohlke. BSD-3-Clause; its licence travels inside the wheel.
  https://github.com/cgohlke/tifffile
- **pyshp** 3.1.6 — Copyright (c) 2013 Joel Lawhead. MIT; its licence travels inside the wheel.
  https://github.com/GeospatialPython/pyshp

## Adapted
- `tiffcodecs.py`: the LZW decoder is a port of the pure-Python decoder of older tifffile releases (BSD-3-Clause,
  Christoph Gohlke); the file came from Digital Landscapes' DL-TerrainSlicer.

## Built on
- **Blender** and its Python API (`bpy`, `mathutils`, `bmesh`) — Blender Foundation, GPL. https://www.blender.org
- **numpy** and Python's own **sqlite3** module, as shipped with Blender.

## Standards and methods implemented here
- GeoTIFF: OGC GeoTIFF Standard 1.1 (19-008r4).
- GeoPackage: OGC GeoPackage Encoding Standard (12-128r19); geometry as Well-Known Binary (OGC 06-103r4).
- GeoJSON: RFC 7946 (2016) and the 2008 GeoJSON specification's `crs` member.
- Transverse Mercator (lon/lat to UTM for RFC 7946 files): the Krüger series as given by C. F. F. Karney (2011),
  "Transverse Mercator with an accuracy of a few nanometers", Journal of Geodesy 85, 475–485.

## Used for testing only (not bundled)
GDAL / rasterio, QGIS and SAGA GIS checked every export; test data from Kartverket (CC BY 4.0, ©Kartverket) and
OpenStreetMap (© OpenStreetMap contributors, ODbL) is not part of this extension.

## How it was made
Built by Digital Landscapes with the help of an AI coding assistant (Claude by Anthropic, in Claude Code): Digital
Landscapes set the goals, took the design decisions, and reviewed and tested every version; the assistant wrote much of
the code. Every version passed a test gate in Blender 4.5 LTS and 5.2 LTS against GDAL's readings of the same files.
