# Changelog

## 1.0.0 — 2026-10-22

First public release.

- Import terrain rasters — GeoTIFF (also LZW or Deflate with the floating-point predictor, as QGIS and GDAL write
  them, and Cloud Optimized GeoTIFF), ESRI ASCII grid with its `.prj`, XYZ grids — as quad meshes at their map
  position; vertices at cell corners (default, the mesh keeps the raster's size) or at cell centres (exact heights);
  nodata as holes or as a flat floor at the raster's lowest height.
- Import GIS vectors — shapefile (`.prj` read), GeoPackage (one layer or all), GeoJSON (a `crs` member honoured,
  RFC 7946 longitude / latitude projected to the scene's UTM zone) — flat, draped on a terrain, or lifted by an
  attribute; attributes kept as mesh attributes.
- One scene origin in double precision; set by the first import, by hand, or from a file's lower-left corner.
- Grid from selection and Export GeoTIFF: any selected geometry (also Geometry Nodes instances) ray-cast onto a grid
  of a chosen cell size, cropped to a terrain or a drawn extent.
- Difference raster (new minus existing) with cut / fill volumes, a coloured difference mesh and closed cut / fill
  solids.
- Resample grid: 0.10–100 m; finer = bilinear interpolation (no new measured detail, and the tool says so), coarser =
  area-weighted mean (volume kept).
- A coordinate system without an EPSG code is kept from the source and written back on export. Rasters in feet are
  refused with a clear message (reproject them to a metric CRS first); heights in feet are converted.
- After an import the 3D view frames the new object and extends its clip distance.

Development versions 0.1.0–0.7.0 were not published.
