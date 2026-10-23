# DL GIS Exchange

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23575333.svg)](https://doi.org/10.5281/zenodo.23575333)

A Blender extension by [Digital Landscapes](https://digital-landscapes.com) that moves terrain and map data between GIS
and Blender in both directions — no GDAL, no QGIS plug-in, nothing else to install.

- **Terrain rasters in:** GeoTIFF, ESRI ASCII grid (`.asc`) and XYZ grids become quad meshes at their map position.
- **GIS vectors in:** shapefile, GeoPackage and GeoJSON become points, edges and faces — flat, draped on a terrain, or
  lifted by an attribute.
- **Rasters out:** any selection in Blender (a modelled terrain, a design, a sculpted landform) is sampled onto a grid
  and written as a GeoTIFF that QGIS and SAGA GIS open in place.
- **Cut and fill:** a difference raster (new minus existing) with cut and fill volumes, and closed cut / fill solids.
- **Resample grid:** a new cell size from 0.10 m to 100 m, finer by bilinear interpolation, coarser by the
  area-weighted mean (the volume is kept).

One scene origin in double precision keeps map coordinates (for example UTM, 7 digits) exact while Blender works in
small local numbers. A coordinate system without an EPSG code is kept and written back on export. Metric coordinate
systems only: a raster in feet is refused with a message to reproject it first.

## Install

Needs **Blender 4.5 LTS or newer** (4.5 is the last Blender for Intel Macs; 5.2 LTS works too). Pure Python, nothing
compiled: it is tested on Windows in Blender 4.5 and 5.2; macOS and Linux have not been tested yet - the self-test
below tells within seconds whether it works on your computer, and we are glad to receive its report.

1. Download `DL GIS Exchange 1.0.1.zip` from the [latest release](../../releases/latest). Keep it as a `.zip` — do not
   unpack it.
2. In Blender: **Edit › Preferences › Get Extensions**, the drop-down arrow at the top right › **Install from Disk…**,
   and choose the zip. (Or drag the zip into the Blender window.)
3. The tools are in the 3D view's sidebar (press **N**), tab **DL GIS**.

A zip installed this way does not update itself: install the next release's zip the same way.

## Check that it works on your computer

The release also has `dl_gis_exchange_selftest.py` (the same file is in [`selftest/`](selftest/)). In Blender: the
**Scripting** workspace › Text Editor › **Text › Open…** › the file › **Run Script** (▶). It takes a few seconds, writes
small test files into a temporary folder, runs 12 checks, and shows the result in a pop-up and in a text called
"DL GIS Exchange self-test report". Your scene is not changed. If something fails, please send us that report text —
it names your operating system and Blender version, nothing personal.

## Manual

The user manual (PDF) is attached to each release.

## Cite

If you use DL GIS Exchange in teaching or research, please cite it — see [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button gives APA and BibTeX):

> Ihle, M. (2026). *DL GIS Exchange* (Version 1.0.1) [Computer software]. Digital Landscapes.
> https://doi.org/10.5281/zenodo.23575333

To cite the tool in general rather than this version, use the DOI for all versions,
[10.5281/zenodo.23575333](https://doi.org/10.5281/zenodo.23575333) - it always resolves to the newest release.

## Licence

GNU General Public License, version 3 or later — see [`LICENSE`](LICENSE). Copyright 2026 Marc Ihle (Digital
Landscapes). Bundled open-source work by others (tifffile, pyshp, an LZW decoder port) is listed with its licences in
[`dl_gis_exchange/CREDITS.md`](dl_gis_exchange/CREDITS.md).

## Repository layout

| Folder / file | What it is |
|---|---|
| `dl_gis_exchange/` | the extension's source — exactly what is inside the release zip |
| `selftest/` | the self-test script |
| `CHANGELOG.md` | what changed in each release |
| `CITATION.cff` | how to cite it |
