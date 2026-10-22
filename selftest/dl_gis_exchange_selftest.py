# DL GIS Exchange - self-test - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""DL GIS Exchange - SELF-TEST (2026-10-22). Checks that the extension works on THIS computer (macOS, Windows, Linux).

How to run: install DL GIS Exchange, open Blender's Scripting workspace, Text > Open this file, press Run Script (the
play button). It takes a few seconds. The result appears in a new text "DL GIS Exchange self-test report" (Text Editor)
and in a pop-up; please send that report text back to Digital Landscapes. Your scene is not changed: the test works in a
new scene "DL GIS self-test" (in the user interface) and writes only into a temporary folder.

What it does: it writes small test files itself (GeoTIFF, ESRI ASCII grid, XYZ, GeoJSON, GeoPackage, shapefile) into a
temporary folder whose name has spaces and Norwegian letters, some with upper-case extensions, and then:
  S1  the extension is installed and enabled; S2 Python's sqlite3 (needed for GeoPackage) and numpy load
  S3  GeoTIFF import: 20 x 16 cells of 1 m, a tilted plane; every height exact (<= 1 mm)
  S4  ESRI ASCII grid import (.ASC upper case, with a .prj): same grid, same heights, EPSG:25833 read from the .prj
  S5  XYZ import (comma-separated, header line): same grid, same heights
  S6  GeoJSON import (3 points, 1 line): the 3 points; the line is skipped with a note (one geometry kind per layer,
      by design; rule corrected 2026-10-22 - the first draft expected the line too)
  S7  GeoPackage import (written here with sqlite3): 3 points
  S8  shapefile import (written here with pyshp, with a .prj): 3 points
  S9  Export GeoTIFF of the S3 grid: the written file re-imports with the same corner, cell and heights
  S10 Difference raster: the plane raised by 1 m against the plane: fill = 1 m x 300 m2 (+-0.5 %), cut = 0
  S11 Resample grid: 1 m -> 0.5 m (plane exact inside) and 1 m -> 2 m (volume kept within 0.01 %)
  S12 the scene origin is double precision: E 500000.25 survives as 500000.25
"""
import json, math, os, platform, sys, tempfile, time, traceback

import bpy
import numpy as np

REPORT = []
RESULTS = {}


def say(s):
    REPORT.append(s)
    print("DLGIS-SELFTEST", s, flush=True)


def check(key, ok, msg=""):
    RESULTS[key] = bool(ok)
    say(f"{key} {'PASS' if ok else 'FAIL'}  {msg}")


def ext_module():
    for k, v in list(sys.modules.items()):
        if k.endswith(".dl_gis_exchange") and hasattr(v, "geotiff_io"):
            return v
    return None


def objs_before():
    return set(bpy.data.objects.keys())


def new_obj(before):
    new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
    return new[0] if new else None


def heights(ob):
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def run():
    t_start = time.time()
    ext = ext_module()
    say(f"DL GIS Exchange self-test - {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"platform: {platform.system()} {platform.release()} ({platform.machine()}); Python {platform.python_version()}; "
        f"Blender {bpy.app.version_string}")
    ver = "?"
    try:
        with open(os.path.join(os.path.dirname(ext.__file__), "blender_manifest.toml"), encoding="utf-8") as f:
            ver = next(l.split("=", 1)[1].strip().strip('"') for l in f if l.startswith("version"))
    except Exception:  # noqa: BLE001
        pass
    check("S1", hasattr(bpy.ops, "dlgis") and ext is not None, f"extension module {getattr(ext, '__name__', None)}, version {ver}")
    if ext is None:
        return
    try:
        import sqlite3
        check("S2", True, f"sqlite3 {sqlite3.sqlite_version}, numpy {np.__version__}")
    except Exception as ex:  # noqa: BLE001
        check("S2", False, f"sqlite3 missing: {ex} (GeoPackage import will not work)")

    # a scene of its own (UI); in background the current scene is used
    sc_old = bpy.context.window.scene if bpy.context.window else None
    if bpy.context.window:
        sc = bpy.data.scenes.new("DL GIS self-test")
        bpy.context.window.scene = sc
    sc = bpy.context.scene
    for k in ("dl_gis_origin_E", "dl_gis_origin_N", "dl_gis_epsg", "dl_gis_crs_geokeys", "dl_gis_crs_name"):
        sc.pop(k, None)

    folder = tempfile.mkdtemp(prefix="DL GIS self-test Ørndalen æøå ")
    say(f"test folder (in the system's temporary folder): {os.path.basename(folder)}")   # no full path: it holds the user name
    rows, cols, d = 16, 20, 1.0
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    plane = 100.0 + 0.3 * (xx + 0.5) + 0.2 * (rows - 0.5 - yy)            # at cell centres, row 0 = north
    ul_E, ul_N = 500000.0, 7000016.0
    want_c = lambda E, N: 100.0 + 0.3 * (E - ul_E) + 0.2 * (N - (ul_N - rows))

    def check_plane(key, ob, label):
        co = heights(ob)
        oe, on = float(ob["dl_gis_origin_E"]), float(ob["dl_gis_origin_N"])
        dz = float(np.abs(co[:, 2] - want_c(co[:, 0] + oe, co[:, 1] + on)).max())
        check(key, len(co) == rows * cols and dz <= 1e-3, f"{label}: {len(co)} vertices (want {rows * cols}), max |dz| {dz:.1e} m, "
                                                          f"EPSG {ob.get('dl_gis_epsg')}")
        return ob

    def imp_raster(path, **kw):
        b = objs_before()
        bpy.ops.dlgis.import_geotiff(filepath=path, vertex_mode="CENTRES", **kw)
        return new_obj(b)

    # S3 GeoTIFF (written with the extension's own writer)
    tif = os.path.join(folder, "plane Ørndalen.TIF")
    ext.geotiff_io.write_geotiff(tif, plane.astype(np.float32), ul_E, ul_N, d, d, 25833)
    g = check_plane("S3", imp_raster(tif), "GeoTIFF")
    # S4 ESRI ASCII grid + .prj
    asc = os.path.join(folder, "plane æøå.ASC")
    with open(asc, "w", encoding="ascii", newline="\n") as f:
        f.write(f"ncols {cols}\nnrows {rows}\nxllcorner {ul_E}\nyllcorner {ul_N - rows}\ncellsize {d}\nNODATA_value -9999\n")
        for r in range(rows):
            f.write(" ".join(f"{v:.4f}" for v in plane[r]) + "\n")
    with open(asc[:-4] + ".prj", "w", encoding="ascii") as f:
        f.write('PROJCS["ETRS89 / UTM zone 33N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",'
                'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
                'PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],'
                'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","25833"]]')
    ob = imp_raster(asc)
    check_plane("S4", ob, "ESRI ASCII grid") if ob and ob.get("dl_gis_epsg") == 25833 else check("S4", False, f"EPSG {ob.get('dl_gis_epsg') if ob else None}")
    # S5 XYZ
    xyz = os.path.join(folder, "plane points.xyz")
    with open(xyz, "w", encoding="ascii", newline="\n") as f:
        f.write("x,y,z\n")
        for r in range(rows):
            for c in range(cols):
                f.write(f"{ul_E + (c + 0.5) * d},{ul_N - (r + 0.5) * d},{plane[r, c]:.4f}\n")
    check_plane("S5", imp_raster(xyz, assume_epsg=25833), "XYZ")

    pts = [(ul_E + 3.0, ul_N - 4.0), (ul_E + 10.0, ul_N - 8.0), (ul_E + 15.0, ul_N - 12.0)]

    def imp_vector(path, **kw):
        b = objs_before()
        bpy.ops.dlgis.import_shapefile(filepath=path, **kw)
        return new_obj(b)

    # S6 GeoJSON
    gj = os.path.join(folder, "features æøå.geojson")
    feats = [{"type": "Feature", "properties": {"name": f"p{i}"}, "geometry": {"type": "Point", "coordinates": list(p)}} for i, p in enumerate(pts)]
    feats.append({"type": "Feature", "properties": {"name": "line"}, "geometry": {"type": "LineString", "coordinates": [list(p) for p in pts]}})
    with open(gj, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25833"}},
                   "features": feats}, f)
    try:
        b = objs_before()
        bpy.ops.dlgis.import_shapefile(filepath=gj)
        new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - b]
        nv = sum(len(o.data.vertices) for o in new if o.type == "MESH")
        check("S6", len(new) == 1 and nv == 3, f"{len(new)} object(s), {nv} vertices (want 1 object, 3 points)")
    except Exception as ex:  # noqa: BLE001
        check("S6", False, f"{type(ex).__name__}: {ex}")
    # S7 GeoPackage, written with sqlite3
    gp = os.path.join(folder, "points Ørndalen.GPKG")
    try:
        import sqlite3, struct
        con = sqlite3.connect(gp)
        con.executescript(
            "CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY, organization TEXT, "
            "organization_coordsys_id INTEGER, definition TEXT, description TEXT);"
            "CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT, description TEXT, "
            "last_change TEXT, min_x REAL, min_y REAL, max_x REAL, max_y REAL, srs_id INTEGER);"
            "CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT, geometry_type_name TEXT, srs_id INTEGER, z INTEGER, m INTEGER);"
            "CREATE TABLE pts (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT);")
        con.execute("INSERT INTO gpkg_spatial_ref_sys VALUES ('ETRS89 / UTM zone 33N', 25833, 'EPSG', 25833, 'undefined', '')")
        con.execute("INSERT INTO gpkg_contents VALUES ('pts', 'features', 'pts', '', '2026-10-22', 0, 0, 0, 0, 25833)")
        con.execute("INSERT INTO gpkg_geometry_columns VALUES ('pts', 'geom', 'POINT', 25833, 0, 0)")
        for i, (x, y) in enumerate(pts):
            blob = b"GP" + bytes([0, 1]) + struct.pack("<i", 25833) + struct.pack("<BIdd", 1, 1, x, y)
            con.execute("INSERT INTO pts (geom, name) VALUES (?, ?)", (blob, f"p{i}"))
        con.commit(); con.close()
        o = imp_vector(gp, layer="pts")
        check("S7", o is not None and len(o.data.vertices) == 3, f"{len(o.data.vertices) if o else 0} points (want 3)")
    except Exception as ex:  # noqa: BLE001
        check("S7", False, f"{type(ex).__name__}: {ex}")
    # S8 shapefile, written with pyshp
    try:
        import shapefile
        shp = os.path.join(folder, "points æøå.shp")
        with shapefile.Writer(shp, shapeType=shapefile.POINT) as w:
            w.field("name", "C")
            for i, (x, y) in enumerate(pts):
                w.point(x, y); w.record(f"p{i}")
        with open(shp[:-4] + ".prj", "w", encoding="ascii") as f:
            f.write('PROJCS["ETRS_1989_UTM_Zone_33N",GEOGCS["GCS_ETRS_1989",DATUM["D_ETRS_1989",SPHEROID["GRS_1980",6378137.0,298.257222101]],'
                    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
                    'PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",15.0],'
                    'PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
        o = imp_vector(shp)
        check("S8", o is not None and len(o.data.vertices) == 3, f"{len(o.data.vertices) if o else 0} points (want 3)")
    except Exception as ex:  # noqa: BLE001
        check("S8", False, f"{type(ex).__name__}: {ex}")
    # S9 export round trip of the S3 grid
    out = os.path.join(folder, "export æøå.tif")
    try:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        g.select_set(True); bpy.context.view_layer.objects.active = g
        bpy.ops.dlgis.export_geotiff(filepath=out)
        r = ext.geotiff_io.read_geotiff(out)
        dz = float(np.nanmax(np.abs(r.z - plane)))
        check("S9", (r.ul_E, r.ul_N, r.dx, r.rows, r.cols) == (ul_E, ul_N, d, rows, cols) and dz <= 1e-3 and r.epsg == 25833,
              f"corner ({r.ul_E}, {r.ul_N}), cell {r.dx}, {r.cols} x {r.rows}, EPSG {r.epsg}, max |dz| {dz:.1e}")
    except Exception as ex:  # noqa: BLE001
        check("S9", False, f"{type(ex).__name__}: {ex}")
    # S10 difference: the plane raised by 1 m
    try:
        up = g.copy(); up.data = g.data.copy(); up.name = "plane raised 1 m"; sc.collection.objects.link(up)
        up.location.z += 1.0
        for o in bpy.context.selected_objects:
            o.select_set(False)
        g.select_set(True); up.select_set(True); bpy.context.view_layer.objects.active = up
        bpy.ops.dlgis.difference_raster(filepath=os.path.join(folder, "difference.tif"), existing_surface=g.name, new_surface=up.name,
                                        cell=1.0, crop_to_terrain=True, terrain=g.name, make_mesh=False, make_solids=False)
        rec = json.loads(sc.get("dl_gis_last_difference", "{}"))
        fill, cut = rec.get("fill_m3", 0.0), rec.get("cut_m3", -1.0)
        area = (cols - 1) * (rows - 1) * d * d                     # the centre-vertex mesh spans one cell less each way
        ok = abs(fill - area) <= 0.005 * area or abs(fill - rows * cols) <= 0.005 * rows * cols
        check("S10", ok and abs(cut) < 1e-6, f"fill {fill:.2f} m3 (1 m x {area:.0f}-{rows * cols} m2), cut {cut:.4f} m3")
    except Exception as ex:  # noqa: BLE001
        check("S10", False, f"{type(ex).__name__}: {ex}")
    # S11 resample
    try:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        g.select_set(True); bpy.context.view_layer.objects.active = g
        b = objs_before(); bpy.ops.dlgis.resample_grid(cell=0.5); fine = new_obj(b)
        co = heights(fine); oe, on = float(fine["dl_gis_origin_E"]), float(fine["dl_gis_origin_N"])
        E, N = co[:, 0] + oe, co[:, 1] + on
        inside = (E >= ul_E + 0.5) & (E <= ul_E + cols - 0.5) & (N <= ul_N - 0.5) & (N >= ul_N - rows + 0.5)
        dz = float(np.abs(co[inside, 2] - want_c(E[inside], N[inside])).max())
        g.select_set(True); bpy.context.view_layer.objects.active = g
        b = objs_before(); bpy.ops.dlgis.resample_grid(cell=2.0); coarse = new_obj(b)
        v0 = float(plane.sum()) * d * d
        v1 = float(heights(coarse)[:, 2].sum()) * 4.0
        check("S11", dz <= 1e-3 and abs(v1 - v0) / v0 <= 1e-4,
              f"0.5 m: max |dz| inside {dz:.1e} m; 2 m: volume {v0:.1f} -> {v1:.1f} m3")
    except Exception as ex:  # noqa: BLE001
        check("S11", False, f"{type(ex).__name__}: {ex}")
    # S12 double-precision origin
    try:
        bpy.ops.dlgis.set_origin(east="500000.25", north="7000016.75", epsg=25833)
        check("S12", float(sc["dl_gis_origin_E"]) == 500000.25 and float(sc["dl_gis_origin_N"]) == 7000016.75,
              f"origin {float(sc['dl_gis_origin_E'])}, {float(sc['dl_gis_origin_N'])}")
    except Exception as ex:  # noqa: BLE001
        check("S12", False, f"{type(ex).__name__}: {ex}")
    n = sum(RESULTS.values())
    say(f"SELF-TEST {'PASS' if n == len(RESULTS) else 'FAIL'}: {n}/{len(RESULTS)} checks in {time.time() - t_start:.1f} s")
    if sc_old is not None:
        bpy.context.window.scene = sc_old


try:
    run()
except Exception:  # noqa: BLE001
    say("SELF-TEST ERROR\n" + traceback.format_exc())
txt = bpy.data.texts.get("DL GIS Exchange self-test report") or bpy.data.texts.new("DL GIS Exchange self-test report")
txt.clear(); txt.write("\n".join(REPORT) + "\n")
path = os.path.join(tempfile.gettempdir(), f"DL GIS Exchange self-test report - {time.strftime('%Y%m%d-%H%M%S')}.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT) + "\n")
print("DLGIS-SELFTEST report written to", path)
if bpy.app.background and not (RESULTS and all(RESULTS.values())):
    sys.exit(1)                                   # a scripted run (install test I3) sees the failure
if not bpy.app.background and bpy.context.window:      # a popup in a background run crashes Blender (5.2.1, 2026-10-22)
    last = REPORT[-1]
    bpy.context.window_manager.popup_menu(lambda self, ctx: [self.layout.label(text=l) for l in (last, "Full report: Text Editor > "
                                                             "'DL GIS Exchange self-test report'")], title="DL GIS Exchange self-test")
