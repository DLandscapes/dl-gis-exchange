# DL GIS Exchange - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
# 1.0.0 (2026-10-22): first public release. Terrain rasters (GeoTIFF, ESRI ASCII grid, XYZ) in as quad meshes at
#                     their map position (one scene origin in double precision); GIS vectors (shapefile, GeoPackage,
#                     GeoJSON) in as points, edges and faces, flat, draped or from a field; a grid from any selection
#                     out as GeoTIFF; difference raster with cut / fill volumes and solids; resample grid (0.10-100 m);
#                     a CRS without an EPSG code kept and written back. Development versions 0.1.0-0.7.0 were not
#                     published.
"""GeoTIFF terrain into Blender as a quad mesh, and back out again.

Georeference model
  - The SCENE has one origin (E, N in the raster's CRS, plus the EPSG code) stored as double-precision ID
    properties: scene["dl_gis_origin_E"], ["dl_gis_origin_N"], ["dl_gis_epsg"]. Local x, y = E - origin_E, N - origin_N.
    Set by the first import (centre of that raster, whole metres), or by hand in the DL GIS panel.
  - Each imported or sampled grid OBJECT carries its own facts: corner, cell size, EPSG, source, nodata, raster type.
Beginner-first: two or three choices on the first level; everything else behind Advanced.
"""
import math
import os
import time

import bpy
import numpy as np
from mathutils import Vector   # 0.6.5: view framing
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

import json

from . import difference, geojson_io, geotiff_io, gpkg_io, grid_text_io, meshing, projection, resample, sampling, vector_io, vector_mesh

ORIGIN_E, ORIGIN_N, EPSG = "dl_gis_origin_E", "dl_gis_origin_N", "dl_gis_epsg"


# ------------------------------------------------------------------------------------------------ scene origin
def scene_origin(scene):
    """(E, N, epsg) or None when the scene has no origin yet."""
    if ORIGIN_E in scene and ORIGIN_N in scene:
        return float(scene[ORIGIN_E]), float(scene[ORIGIN_N]), int(scene.get(EPSG, 0)) or None
    return None


CRS_KEYS, CRS_NAME = "dl_gis_crs_geokeys", "dl_gis_crs_name"


def set_scene_origin(scene, e, n, epsg, crs_keys="keep", crs_name=""):
    """0.6.3: crs_keys = the GeoKeys of a user-defined projected CRS (dict, stored with the origin and written back on
    export), None (remove them) or "keep" (leave them). An EPSG code always removes them - the code is the CRS then."""
    scene[ORIGIN_E] = float(e)   # Python floats become DOUBLE id-properties (float32 would step 0.5 m at N 7.7e6)
    scene[ORIGIN_N] = float(n)
    scene[EPSG] = int(epsg or 0)
    if epsg or crs_keys is None:
        scene.pop(CRS_KEYS, None)
        scene.pop(CRS_NAME, None)
    elif isinstance(crs_keys, dict):
        scene[CRS_KEYS] = json.dumps(crs_keys)
        scene[CRS_NAME] = str(crs_name or "")


def scene_geokeys(scene):
    """0.6.3: the stored GeoKeys of the scene's user-defined CRS, or None."""
    raw = scene.get(CRS_KEYS)
    return json.loads(raw) if raw else None


# ------------------------------------------------------------------------------------------------ view (0.6.5)
def is_startup_cube(o):
    """Blender's untouched default cube: not earlier GIS work, so it does not trigger the "no scene origin" warning.
    a user imported into the startup scene (2026-10-21) and got that warning for the cube."""
    if o.type != "MESH" or not o.name.startswith("Cube") or "dl_gis_source" in o or o.modifiers:
        return False
    if len(o.data.vertices) != 8 or o.location.length > 1e-6 or Vector(o.rotation_euler).length > 1e-6:
        return False
    return all(abs(s - 1.0) < 1e-6 for s in o.scale) and all(abs(d - 2.0) < 1e-4 for d in o.dimensions)


def frame_view_on(context, ob):
    """Centre every 3D view on the object, set the view distance from its size and raise the clip end so the whole
    object is drawn (0.6.5: a 1.9 km terrain 157-440 m high stayed invisible behind Blender's default view, whose clip end
    is 1000 m). Camera views are left alone. No window (background) - nothing to do."""
    wm = context.window_manager
    if wm is None or not wm.windows:
        return
    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre, diag = (lo + hi) / 2.0, max((hi - lo).length, 1.0)
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            sp = area.spaces.active
            r3 = sp.region_3d
            if r3 is None or r3.view_perspective == "CAMERA":
                continue
            r3.view_location = centre
            r3.view_distance = diag
            need = 3.0 * diag
            if sp.clip_end < need:
                sp.clip_end = float(10 ** math.ceil(math.log10(need)))      # a round number: 10 km, 100 km
                if sp.clip_end / max(sp.clip_start, 1e-9) > 1e6:             # keep the depth range sane
                    sp.clip_start = sp.clip_end / 1e6
            area.tag_redraw()


def selection_box_world(context, exclude=()):
    """Axis-aligned world bounding box (xmin, xmax, ymin, ymax) of the selected objects, or None."""
    from mathutils import Vector
    pts = []
    for ob in context.selected_objects:
        if ob in exclude:
            continue
        if ob.type == "EMPTY":
            pts.append(ob.matrix_world.translation.copy())
        elif ob.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            pts.extend(ob.matrix_world @ Vector(c) for c in ob.bound_box)
    if not pts:
        return None
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def store_facts(ob, source, ul_E, ul_N, dx, dy, rows, cols, epsg, origin, nodata, nodata_mode, raster_type, step=1, notes=()):
    ob["dl_gis_source"] = source
    ob["dl_gis_epsg"] = int(epsg or 0)
    ob["dl_gis_ul_E"] = float(ul_E)
    ob["dl_gis_ul_N"] = float(ul_N)
    ob["dl_gis_cell_x"] = float(dx)
    ob["dl_gis_cell_y"] = float(dy)
    ob["dl_gis_rows"] = int(rows)
    ob["dl_gis_cols"] = int(cols)
    ob["dl_gis_step"] = int(step)
    ob["dl_gis_nodata"] = float(nodata) if nodata is not None else float("nan")
    ob["dl_gis_nodata_mode"] = nodata_mode
    ob["dl_gis_raster_type"] = raster_type
    ob["dl_gis_origin_E"] = float(origin[0])
    ob["dl_gis_origin_N"] = float(origin[1])
    ob["dl_gis_notes"] = "; ".join(notes)


LAST_EXPORT_DIR, LAST_IMPORT_DIR = "dl_gis_last_export_dir", "dl_gis_last_import_dir"


def default_export_path(scene, ob):
    """Where the export dialog opens (0.2.1): the last export folder, else the last import folder, else beside the
    saved .blend, else the user's Documents - and the active object's name as the file name."""
    folder = None
    for key in (LAST_EXPORT_DIR, LAST_IMPORT_DIR):
        d = scene.get(key)
        if d and os.path.isdir(str(d)):
            folder = str(d)
            break
    if folder is None and bpy.data.filepath:
        folder = os.path.dirname(bpy.data.filepath)
    if folder is None:
        docs = os.path.join(os.path.expanduser("~"), "Documents")
        folder = docs if os.path.isdir(docs) else os.path.expanduser("~")
    stem = ob.name if ob is not None else "terrain"
    stem = "".join(c for c in stem if c not in '\\/:*?"<>|').strip() or "terrain"
    return os.path.join(folder, f"{stem} - export.tif")


def has_grid_facts(ob):
    return ob is not None and all(k in ob for k in ("dl_gis_ul_E", "dl_gis_ul_N", "dl_gis_cell_x", "dl_gis_rows", "dl_gis_cols"))


def link_new_object(context, name, me):
    ob = bpy.data.objects.new(name, me)
    context.collection.objects.link(ob)
    for o in context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    context.view_layer.objects.active = ob
    return ob


# ------------------------------------------------------------------------------------------------ import
def file_corner(path):
    """0.5.1: (E, N, epsg, what) of a GIS file's LOWER-LEFT corner - a GeoTIFF's outer south-west corner, a vector
    file's bounding-box minimum (first layer of a GeoPackage). Files made in a CAD or 3D tool often place the model
    with the model frame's corner at local (0, 0), so that corner is the origin they need (probe 12 / 13, 2026-10-13)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff", ".asc", ".xyz"):
        r = grid_text_io.read_raster(path)      # 0.6.4: text grids too
        return (round(r.ul_E, 3), round(r.ul_N - r.rows * r.dy, 3), r.epsg, f"south-west corner of {r.source} ({r.cols} x {r.rows} cells)",
                r.geokeys, r.crs_name)                       # 0.6.3: a user-defined CRS travels with the origin
    if ext == ".gpkg":
        layer = gpkg_io.read_gpkg(path)
    elif ext in (".geojson", ".json"):
        layer = geojson_io.read_geojson(path)
    else:
        layer = vector_io.read_shapefile(path)
    return (round(layer.bbox[0], 3), round(layer.bbox[1], 3), layer.epsg, f"lower-left corner of {layer.source}'s extent ({len(layer.features)} features)",
            None, "")


def origin_from_object(context, ob, e_min, n_min, half_x=0.0, half_y=0.0):
    """0.5.1: the scene origin that puts a file whose lower-left corner is (e_min, n_min) onto an existing object:
    origin = file corner (+ half a cell for raster vertices at cell centres) - the object's lower-left world corner."""
    b = sampling.selection_box([ob], context.evaluated_depsgraph_get())
    if b is None:
        return None, None
    return round(e_min + half_x - b[0], 3), round(n_min + half_y - b[2], 3), b


NO_ORIGIN_WARNING = ("This file already had {n} mesh objects but no scene origin: the import was placed with the origin at the "
                     "file's centre (E {e:,.0f}, N {nn:,.0f}). If those objects come from GIS data, undo and use 'Place on existing "
                     "object' in this dialog, or set the origin from a file's corner first (DL GIS > Scene origin)")


class DLGIS_OT_import_geotiff(bpy.types.Operator, ImportHelper):
    """Import a terrain raster - GeoTIFF, ESRI ASCII grid (.asc) or XYZ grid (.xyz) - as a quad mesh, one vertex per cell"""
    bl_idname = "dlgis.import_geotiff"
    bl_label = "Import terrain raster as quad mesh"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".tif"
    filter_glob: StringProperty(default="*.tif;*.tiff;*.asc;*.xyz", options={"HIDDEN"})   # 0.6.4: + .asc, .xyz
    place_on: StringProperty(
        name="Place on existing object",
        description="For a file that already holds this terrain without a scene origin: pick the object this raster should lie on; "
                    "the scene origin is set so that the import lands on it, and every later import lines up (0.5.1)")

    vertex_mode: EnumProperty(
        name="Vertices",
        description="Where the mesh's vertices sit (0.5.2)",
        items=[("CORNERS", "At cell corners (one face per cell)",
                "One vertex on every cell corner, one face per cell: a 54 x 192 raster gives a 54 x 192 m mesh of 54 x 192 faces; "
                "a vertex's height is the mean of the cells around it"),
               ("CENTRES", "At cell centres (exact heights)",
                "One vertex at every cell centre with the raster's own height: the mesh ends half a cell inside the raster's "
                "edge (53 x 191 m for 54 x 192 cells); exports and differences round-trip exactly")],
        default="CORNERS")
    nodata_mode: EnumProperty(
        name="Nodata",
        description="What happens where the raster has no data",
        items=[("HOLES", "Holes", "Cells without data are left out: you will see holes in the mesh"),
               ("FLAT", "Flat floor", "Cells without data get the lowest height of the raster: a flat floor fills the holes")],
        default="HOLES")
    clip_to_selection: BoolProperty(
        name="Clip to selection",
        description="Import only the part of the raster under the selected objects (needs a scene origin from an earlier import)",
        default=False)
    advanced: BoolProperty(name="Advanced", description="Show the settings below", default=False)
    step: IntProperty(
        name="Step", description="Use every n-th cell: 1 = every cell, 4 = every fourth (16 times fewer faces)",
        default=1, min=1, max=64)
    max_cells: IntProperty(
        name="Max cells (millions)",
        description="Rasters with more cells are imported with a larger step automatically, so Blender stays responsive",
        default=4, min=1, max=64)
    origin_mode: EnumProperty(
        name="Scene origin",
        description="Where local (0, 0) lies in map coordinates",
        items=[("KEEP", "Keep", "Keep the scene origin set by an earlier import, so tiles line up"),
               ("RESET", "Reset to this raster", "Put the scene origin at the centre of this raster")],
        default="KEEP")
    smooth: BoolProperty(name="Smooth shading", description="Smooth instead of faceted shading", default=True)
    frame_view: BoolProperty(name="Frame the view", description="Centre the 3D view on the import and extend the view clip end so the whole terrain is drawn (0.6.5)", default=True)
    assume_epsg: IntProperty(
        name="EPSG if the file has none",
        description="For an XYZ file or an ESRI ASCII grid without a .prj: the projected CRS it is in, for example 25832. "
                    "0 = take the scene's EPSG (0.6.4)",
        default=0, min=0, max=32766)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        if os.path.splitext(self.filepath)[1].lower() in grid_text_io.TEXT_EXT:   # 0.6.4: text grids may carry no CRS
            lay.prop(self, "assume_epsg")
        lay.prop(self, "vertex_mode")
        lay.prop(self, "nodata_mode")
        lay.prop(self, "clip_to_selection")
        if scene_origin(context.scene) is None or self.place_on:
            lay.prop_search(self, "place_on", bpy.data, "objects")
        box = lay.box()
        box.prop(self, "advanced", toggle=True)
        col = box.column()
        col.enabled = self.advanced
        col.prop(self, "step")
        col.prop(self, "max_cells")
        col.prop(self, "origin_mode")
        col.prop(self, "smooth")
        col.prop(self, "frame_view")

    def execute(self, context):
        path = self.filepath
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            r = grid_text_io.read_raster(path)       # 0.6.4: GeoTIFF, ESRI ASCII grid or XYZ
        except geotiff_io.GeoTiffError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        except Exception as ex:  # noqa: BLE001 - show the user something useful
            self.report({"ERROR"}, f"Could not read {os.path.basename(path)}: {type(ex).__name__}: {ex}")
            return {"CANCELLED"}

        scene = context.scene
        if os.path.splitext(path)[1].lower() in grid_text_io.TEXT_EXT and r.epsg is None:   # 0.6.4: a text grid without a CRS
            so = scene_origin(scene)
            if self.assume_epsg:
                r.epsg = int(self.assume_epsg)
                r.notes.append(f"CRS not in the file: EPSG:{r.epsg} as set in the import dialog")
            elif so is not None and so[2]:
                r.epsg = int(so[2])
                r.notes.append(f"CRS not in the file: the scene's EPSG:{r.epsg} taken")
            else:
                r.notes.append("CRS not in the file and none set: exports will carry no CRS (set 'EPSG if the file has none')")
        origin = scene_origin(scene)
        step = self.step if self.advanced else 1
        origin_mode = self.origin_mode if self.advanced else "KEEP"
        smooth = self.smooth if self.advanced else True
        extra_notes = []
        n_meshes_before = sum(1 for o in scene.objects if o.type == "MESH" and not is_startup_cube(o))   # 0.6.5: not the startup cube

        if self.place_on:                                  # 0.5.1: the origin that lands this raster on an existing object
            target = bpy.data.objects.get(self.place_on)
            if target is None:
                self.report({"ERROR"}, f"Object '{self.place_on}' to place the raster on was not found")
                return {"CANCELLED"}
            # the mesh's box is that of the cells WITH data (a raster with a nodata margin starts inside its outer
            # corner - gate G11 run 1 was 7 m off on the test patch): first / last data column and row
            has = ~np.isnan(r.z)
            cols_with = np.flatnonzero(has.any(axis=0))
            rows_with = np.flatnonzero(has.any(axis=1))
            if cols_with.size == 0:
                self.report({"ERROR"}, "No cells with data in this raster")
                return {"CANCELLED"}
            c0, c1, r0, r1 = int(cols_with[0]), int(cols_with[-1]), int(rows_with[0]), int(rows_with[-1])
            # the target's vertices sit at cell centres (meshes made by others, this tool's CENTRES mode) unless it
            # says it was made with vertices at cell corners (0.5.2)
            at_corners = str(target.get("dl_gis_vertex_mode", "centres")) == "corners"
            half = 0.0 if at_corners else 0.5
            e_first = r.ul_E + (c0 + half) * r.dx                # west-most vertex of the data
            n_first = r.ul_N - (r1 + 1.0 - half) * r.dy          # south-most vertex of the data
            oe_new, on_new, b = origin_from_object(context, target, e_first, n_first)
            if oe_new is None:
                self.report({"ERROR"}, f"{target.name} has no extent to place the raster on")
                return {"CANCELLED"}
            span_x, span_y = (c1 - c0 + (1 if at_corners else 0)) * r.dx, (r1 - r0 + (1 if at_corners else 0)) * r.dy
            dw = abs((b[1] - b[0]) - span_x)
            dh = abs((b[3] - b[2]) - span_y)
            if dw > 2 * r.dx or dh > 2 * r.dy:
                extra_notes.append(f"{target.name} is {b[1] - b[0]:.1f} x {b[3] - b[2]:.1f} m, the raster's vertices span "
                                   f"{span_x:.1f} x {span_y:.1f} m: sizes differ, aligned at the lower-left corner anyway")
            if origin is not None and (abs(origin[0] - oe_new) > 1e-3 or abs(origin[1] - on_new) > 1e-3):
                extra_notes.append(f"scene origin moved from E {origin[0]:,.3f} N {origin[1]:,.3f} to E {oe_new:,.3f} N {on_new:,.3f}: "
                                   f"objects imported with the old origin keep their local place and no longer match their map position")
            set_scene_origin(scene, oe_new, on_new, r.epsg or (origin[2] if origin else None),
                             crs_keys=r.geokeys if r.geokeys else "keep", crs_name=r.crs_name)
            origin = scene_origin(scene)
            extra_notes.append(f"scene origin set to E {oe_new:,.3f} N {on_new:,.3f} so that {r.source} lies on {target.name}")

        if self.clip_to_selection:
            box = selection_box_world(context)
            if origin is None:
                self.report({"WARNING"}, "Clip to selection needs a scene origin from an earlier import; whole raster imported")
            elif box is None:
                self.report({"WARNING"}, "Nothing selected to clip to; whole raster imported")
            else:
                oe, on, _ = origin
                w = geotiff_io.window(r, box[0] + oe, box[1] + oe, box[2] + on, box[3] + on)
                if w is None:
                    self.report({"ERROR"}, "The selection does not overlap the raster")
                    return {"CANCELLED"}
                r = geotiff_io.crop(r, *w)

        n_cells = r.rows * r.cols
        auto = 1
        while n_cells / (auto * auto) > self.max_cells * 1_000_000:
            auto += 1
        if auto > step:
            self.report({"WARNING"}, f"{n_cells:,} cells: imported with step {auto} to stay under {self.max_cells} million; set Step under Advanced to change")
            step = auto
        r = geotiff_io.stride(r, step)

        if origin is None or origin_mode == "RESET":
            ce = round(r.ul_E + r.cols * r.dx / 2.0)
            cn = round(r.ul_N - r.rows * r.dy / 2.0)
            set_scene_origin(scene, ce, cn, r.epsg, crs_keys=r.geokeys, crs_name=r.crs_name)
            origin = scene_origin(scene)
            if n_meshes_before and origin_mode != "RESET":      # 0.5.1: geometry without an origin - say so
                extra_notes.append(NO_ORIGIN_WARNING.format(n=n_meshes_before, e=ce, nn=cn))
        oe, on, scene_epsg = origin
        if r.epsg and scene_epsg and r.epsg != scene_epsg:
            self.report({"WARNING"}, f"Raster is EPSG:{r.epsg} but the scene origin is EPSG:{scene_epsg}: positions will not line up")
        # 0.6.3: user-defined CRS (GeoKeys) against the scene's CRS
        s_keys = scene_geokeys(scene)
        if r.geokeys and s_keys and r.geokeys != s_keys:
            extra_notes.append(f"the raster's user-defined CRS differs from the scene's ({scene.get(CRS_NAME, '') or 'user-defined'}): "
                               "positions may not line up; exports keep the scene's CRS")
        elif r.geokeys and scene_epsg:
            extra_notes.append(f"the raster has a user-defined CRS, the scene EPSG:{scene_epsg}: the CRS differs or cannot be "
                               "compared - check the positions; exports keep the scene's EPSG")
        elif r.epsg and s_keys:
            extra_notes.append(f"the raster is EPSG:{r.epsg}, the scene has a user-defined CRS: the CRS differs or cannot be "
                               "compared - check the positions; exports keep the scene's CRS")

        if self.vertex_mode == "CORNERS":              # 0.5.2: one vertex per cell corner, one face per cell
            verts, quads, info = meshing.grid_to_quads(meshing.corner_heights(r.z), r.ul_E - r.dx / 2.0, r.ul_N + r.dy / 2.0,
                                                       r.dx, r.dy, oe, on, self.nodata_mode)
        else:
            verts, quads, info = meshing.grid_to_quads(r.z, r.ul_E, r.ul_N, r.dx, r.dy, oe, on, self.nodata_mode)
        if info["n_quads"] == 0:
            self.report({"ERROR"}, "No cells with data in this raster window")
            return {"CANCELLED"}
        stem = os.path.splitext(os.path.basename(path))[0]
        me = meshing.mesh_from_quads(stem, verts, quads, smooth=smooth)
        ob = link_new_object(context, stem, me)
        store_facts(ob, r.source, r.ul_E, r.ul_N, r.dx, r.dy, r.rows, r.cols, r.epsg, origin, r.nodata,
                    self.nodata_mode, r.raster_type, step, list(r.notes) + extra_notes)
        ob["dl_gis_vertex_mode"] = "corners" if self.vertex_mode == "CORNERS" else "centres"
        scene[LAST_IMPORT_DIR] = os.path.dirname(os.path.abspath(path))
        for n in r.notes:
            self.report({"INFO"}, n)
        for n in extra_notes:
            self.report({"WARNING"} if "no scene origin" in n or "differ" in n or "moved" in n else {"INFO"}, n)
        self.report({"INFO"}, f"{r.source}: {info['n_verts']:,} vertices, {info['n_quads']:,} quads, "
                              f"cell {r.dx:g} m, EPSG:{r.epsg or '?'}, origin E {oe:.0f} N {on:.0f}")
        if self.frame_view:
            frame_view_on(context, ob)
        return {"FINISHED"}


# ------------------------------------------------------------------------------------------------ the reverse direction
def is_raster_grid(ob):
    """An object imported from a raster or sampled on a grid (cell size > 0), as against a vector layer."""
    return has_grid_facts(ob) and float(ob.get("dl_gis_cell_x", 0.0)) > 0.0


def terrain_in(objects):
    """The first raster grid among the objects (the imported terrain), or None."""
    for ob in objects:
        if is_raster_grid(ob):
            return ob
    return None


def extent_object(objects, name=""):
    """The object whose xy extent the grid takes: the named one (any mesh, e.g. a plane drawn over the area), else the
    first imported raster among the objects, else None."""
    t = bpy.data.objects.get(name) if name else None
    if t is not None and t.type in {"MESH", "CURVE", "SURFACE", "FONT", "META", "EMPTY"}:
        return t
    return terrain_in(objects)


def grid_facts_hold(ob, context):
    """0.4.4: the stored grid facts (corner, cell, rows, cols) describe an object's geometry only while it stays where
    it was imported or sampled. A test file of 2026-10-12: two imported cone grids moved 163 m east with the move
    applied into the vertices - the facts kept the old corner, the difference sampled an empty place and said the
    surfaces do not overlap. True when the object's world box lies inside the facts' extent (one cell of slack)."""
    if not has_grid_facts(ob):
        return False
    b = sampling.selection_box([ob], context.evaluated_depsgraph_get())
    if b is None:
        return False
    oe, on, _ = scene_origin(context.scene)
    dx = float(ob["dl_gis_cell_x"])
    dy = float(ob.get("dl_gis_cell_y", dx)) or dx
    e0, n1 = float(ob["dl_gis_ul_E"]), float(ob["dl_gis_ul_N"])
    e1, n0 = e0 + int(ob["dl_gis_cols"]) * dx, n1 - int(ob["dl_gis_rows"]) * dy
    return (b[0] + oe >= e0 - dx and b[1] + oe <= e1 + dx and b[2] + on >= n0 - dy and b[3] + on <= n1 + dy)


def grid_definition(context, cell, align, margin, use_active_grid, objects, crop_to_terrain=False, terrain_name=""):
    """(ul_E, ul_N, rows, cols, dx, dy, source_note) for the export / the new grid.

    crop_to_terrain (0.4.1 "Crop to terrain", 0.4.2 "Crop to extent"): the grid covers one object's xy extent only -
    an imported raster's cells, or the world bounding box of any other object (a plane drawn over the area) - at the
    chosen cell size; objects reaching beyond it are cut off.
    0.4.4: stored grid facts are used only while they still match the object's geometry (grid_facts_hold); a moved
    or edited grid falls back to its bounding box, and the note says so."""
    oe, on, _ = scene_origin(context.scene)
    stale = [o.name for o in objects if has_grid_facts(o) and not grid_facts_hold(o, context)]
    tail = f" ({', '.join(stale)}: stored grid no longer matches the geometry - moved? - bounding box used)" if stale else ""
    act = context.active_object
    if use_active_grid and has_grid_facts(act) and act in objects and act.name not in stale:
        return (float(act["dl_gis_ul_E"]), float(act["dl_gis_ul_N"]), int(act["dl_gis_rows"]), int(act["dl_gis_cols"]),
                float(act["dl_gis_cell_x"]), float(act["dl_gis_cell_y"]), f"grid of {act.name}" + tail)
    if crop_to_terrain:
        t = extent_object(objects, terrain_name)
        if t is not None and is_raster_grid(t) and grid_facts_hold(t, context):
            e0, n1 = float(t["dl_gis_ul_E"]), float(t["dl_gis_ul_N"])
            e1 = e0 + int(t["dl_gis_cols"]) * float(t["dl_gis_cell_x"])
            n0 = n1 - int(t["dl_gis_rows"]) * float(t["dl_gis_cell_y"])
            ul_E, ul_N, rows, cols = geotiff_io.grid_for_box(e0, e1, n0, n1, cell, align, margin)
            return ul_E, ul_N, rows, cols, cell, cell, f"extent of {t.name}" + tail
        if t is not None:
            b = sampling.selection_box([t], context.evaluated_depsgraph_get())
            if b is None or b[1] <= b[0] or b[3] <= b[2]:
                raise geotiff_io.GeoTiffError(f"{t.name} has no xy extent to crop to")
            ul_E, ul_N, rows, cols = geotiff_io.grid_for_box(b[0] + oe, b[1] + oe, b[2] + on, b[3] + on, cell, align, margin)
            return ul_E, ul_N, rows, cols, cell, cell, f"extent of {t.name}" + tail
    box = sampling.selection_box(objects, context.evaluated_depsgraph_get())
    if box is None:
        raise geotiff_io.GeoTiffError("Select the objects to sample first")
    ul_E, ul_N, rows, cols = geotiff_io.grid_for_box(box[0] + oe, box[1] + oe, box[2] + on, box[3] + on, cell, align, margin)
    return ul_E, ul_N, rows, cols, cell, cell, f"bounding box of {len(objects)} object(s)" + tail


def sampled_objects(objects, crop, extent_name, op):
    """A drawn Extent object (not an imported raster) defines the grid but is not sampled; say so once."""
    if not crop or not extent_name:
        return objects
    t = bpy.data.objects.get(extent_name)
    if t is None or is_raster_grid(t) or t not in objects:
        return objects
    rest = [o for o in objects if o != t]
    if rest:
        op.report({"INFO"}, f"{t.name} gives the extent and is not sampled")
        return rest
    return objects   # it is the only object: sample it after all


def sample_selection(context, objects, ul_E, ul_N, rows, cols, dx, dy, surface):
    oe, on, _ = scene_origin(context.scene)
    xs = (ul_E - oe) + (np.arange(cols, dtype=np.float64) + 0.5) * dx
    ys = (ul_N - on) - (np.arange(rows, dtype=np.float64) + 0.5) * dy
    deps = context.evaluated_depsgraph_get()
    return sampling.sample_heights(objects, deps, xs, ys, surface)


def _common_props(cls):
    cls.__annotations__["cell"] = FloatProperty(
        name="Cell size", description="Edge length of one raster cell in metres (1 = one height per metre)",
        default=1.0, min=0.001, soft_min=0.1, soft_max=100.0, unit="LENGTH")
    cls.__annotations__["surface"] = EnumProperty(
        name="Surface", description="Which surface a cell takes where objects overlap or a mesh has several layers",
        items=[("HIGHEST", "Highest", "The top surface: what a drop from the sky would meet first"),
               ("LOWEST", "Lowest", "The underside: what rises from below meets first")],
        default="HIGHEST")
    cls.__annotations__["crop_to_terrain"] = BoolProperty(
        name="Crop to extent",
        description="The grid covers only one object's xy extent: the imported terrain, or any object such as a plane you drew over the area; objects reaching beyond its edge are cut off",
        default=True)
    cls.__annotations__["terrain"] = StringProperty(
        name="Extent", description="The object whose xy extent the grid covers (the imported terrain is taken when left empty). A plane or other drawn object is not sampled, only an imported terrain is")
    cls.__annotations__["advanced"] = BoolProperty(name="Advanced", description="Show the settings below", default=False)
    cls.__annotations__["use_active_grid"] = BoolProperty(
        name="Reuse the active grid", description="If the active object is an imported or sampled grid, use its corner and cell size exactly (a round trip)",
        default=True)
    cls.__annotations__["align"] = BoolProperty(
        name="Align to whole cells", description="Snap the corner outward to multiples of the cell size, so the raster lines up with other rasters of that size",
        default=True)
    cls.__annotations__["margin"] = IntProperty(name="Margin (cells)", description="Extra cells around the bounding box", default=0, min=0, max=1000)
    return cls


def _draw_common(self, lay):
    lay.use_property_split = True
    lay.prop(self, "cell")
    lay.prop(self, "surface")
    lay.prop(self, "crop_to_terrain")
    if self.crop_to_terrain:
        lay.prop_search(self, "terrain", bpy.data, "objects")
    box = lay.box()
    box.prop(self, "advanced", toggle=True)
    col = box.column()
    col.enabled = self.advanced
    col.prop(self, "use_active_grid")
    col.prop(self, "align")
    col.prop(self, "margin")
    return col


@_common_props
class DLGIS_OT_grid_from_selection(bpy.types.Operator):
    """Make a quad mesh over the selection's bounding box at a chosen cell size, heights sampled from the selected objects"""
    bl_idname = "dlgis.grid_from_selection"
    bl_label = "Grid from selection"
    bl_options = {"REGISTER", "UNDO"}

    nodata_mode: EnumProperty(
        name="Nodata", description="Cells where no selected object lies underneath",
        items=[("HOLES", "Holes", "Left out of the mesh"), ("FLAT", "Flat floor", "Filled at the lowest sampled height")],
        default="HOLES")

    def invoke(self, context, event):
        act = context.active_object
        if has_grid_facts(act):
            self.cell = float(act["dl_gis_cell_x"])
        t = terrain_in(context.selected_objects)
        if t is not None and not self.terrain:
            self.terrain = t.name
            if is_raster_grid(act) is False and act is not None and not has_grid_facts(act):
                self.cell = float(t["dl_gis_cell_x"])
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        col = _draw_common(self, self.layout)
        self.layout.prop(self, "nodata_mode")

    def execute(self, context):
        if scene_origin(context.scene) is None:
            self.report({"ERROR"}, "No scene origin yet: import a GeoTIFF first or set one in the DL GIS panel")
            return {"CANCELLED"}
        objects = [o for o in context.selected_objects if o.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}]
        if not objects:
            self.report({"ERROR"}, "Select at least one mesh or curve object")
            return {"CANCELLED"}
        adv = self.advanced
        try:
            ul_E, ul_N, rows, cols, dx, dy, note = grid_definition(context, self.cell, self.align if adv else True,
                                                                   self.margin if adv else 0, self.use_active_grid if adv else True, objects,
                                                                   self.crop_to_terrain, self.terrain)
        except geotiff_io.GeoTiffError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        if self.crop_to_terrain and not note.startswith("extent of") and not note.startswith("grid of"):
            self.report({"INFO"}, "Crop to extent: no imported raster among the selected objects and no Extent chosen, so the selection's bounding box was used")
        objects = sampled_objects(objects, self.crop_to_terrain, self.terrain, self)
        if rows * cols > 16_000_000:
            self.report({"ERROR"}, f"{rows:,} x {cols:,} cells is too many; use a larger cell size")
            return {"CANCELLED"}
        t0 = time.time()
        z = sample_selection(context, objects, ul_E, ul_N, rows, cols, dx, dy, self.surface)
        origin = scene_origin(context.scene)
        verts, quads, info = meshing.grid_to_quads(z, ul_E, ul_N, dx, dy, origin[0], origin[1], self.nodata_mode)
        if info["n_quads"] == 0:
            self.report({"ERROR"}, "No cell of the grid lies over a selected object")
            return {"CANCELLED"}
        name = f"Grid {dx:g} m ({self.surface.lower()})"
        me = meshing.mesh_from_quads(name, verts, quads, smooth=True)
        ob = link_new_object(context, name, me)
        store_facts(ob, f"sampled: {note}", ul_E, ul_N, dx, dy, rows, cols, origin[2], origin, -9999.0,
                    self.nodata_mode, "area", 1, [f"surface {self.surface}", f"{int(np.isnan(z).sum())} cells without data"])
        self.report({"INFO"}, f"{rows} x {cols} cells at {dx:g} m sampled in {time.time()-t0:.1f} s: "
                              f"{info['n_verts']:,} vertices, {info['n_quads']:,} quads, {int(np.isnan(z).sum())} nodata")
        return {"FINISHED"}


@_common_props
class DLGIS_OT_export_geotiff(bpy.types.Operator, ExportHelper):
    """Write the selected objects as a GeoTIFF raster (float32 heights) that QGIS and SAGA GIS open in place"""
    bl_idname = "dlgis.export_geotiff"
    bl_label = "Export GeoTIFF"
    bl_options = {"REGISTER"}

    filename_ext = ".tif"
    filter_glob: StringProperty(default="*.tif;*.tiff", options={"HIDDEN"})
    nodata: FloatProperty(name="Nodata value", description="Written where no selected object lies underneath", default=-9999.0)
    compression: EnumProperty(name="Compression", items=[("deflate", "Deflate", "Smaller file (lossless)"), ("none", "None", "Uncompressed")], default="deflate")

    def invoke(self, context, event):
        act = context.active_object
        if has_grid_facts(act):
            self.cell = float(act["dl_gis_cell_x"])
        t = terrain_in(context.selected_objects)
        if t is not None and not self.terrain:
            self.terrain = t.name
            if not has_grid_facts(act):
                self.cell = float(t["dl_gis_cell_x"])
        self.filepath = default_export_path(context.scene, act)
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        col = _draw_common(self, self.layout)
        col.prop(self, "nodata")
        col.prop(self, "compression")

    def execute(self, context):
        origin = scene_origin(context.scene)
        if origin is None:
            self.report({"ERROR"}, "No scene origin yet: import a GeoTIFF first or set one in the DL GIS panel")
            return {"CANCELLED"}
        objects = [o for o in context.selected_objects if o.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}]
        if not objects:
            self.report({"ERROR"}, "Select at least one mesh or curve object")
            return {"CANCELLED"}
        adv = self.advanced
        try:
            ul_E, ul_N, rows, cols, dx, dy, note = grid_definition(context, self.cell, self.align if adv else True,
                                                                   self.margin if adv else 0, self.use_active_grid if adv else True, objects,
                                                                   self.crop_to_terrain, self.terrain)
        except geotiff_io.GeoTiffError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        if self.crop_to_terrain and not note.startswith("extent of") and not note.startswith("grid of"):
            self.report({"INFO"}, "Crop to extent: no imported raster among the selected objects and no Extent chosen, so the selection's bounding box was used")
        objects = sampled_objects(objects, self.crop_to_terrain, self.terrain, self)
        if rows * cols > 64_000_000:
            self.report({"ERROR"}, f"{rows:,} x {cols:,} cells is too many; use a larger cell size")
            return {"CANCELLED"}
        t0 = time.time()
        z = sample_selection(context, objects, ul_E, ul_N, rows, cols, dx, dy, self.surface)
        epsg = origin[2]
        keys = None if epsg else scene_geokeys(context.scene)        # 0.6.3: the source's user-defined CRS
        if not epsg and not keys:
            self.report({"WARNING"}, "The scene has no EPSG code: the raster is written with a user-defined CRS (set one in the DL GIS panel)")
        try:
            info = geotiff_io.write_geotiff(self.filepath, z, ul_E, ul_N, dx, dy, epsg,
                                            nodata=self.nodata if adv else -9999.0, compression=self.compression if adv else "deflate",
                                            geokeys=keys)
        except Exception as ex:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not write {os.path.basename(self.filepath)}: {type(ex).__name__}: {ex}")
            return {"CANCELLED"}
        context.scene[LAST_EXPORT_DIR] = os.path.dirname(os.path.abspath(self.filepath))
        self.report({"INFO"}, f"{os.path.basename(self.filepath)}: {rows} x {cols} cells at {dx:g} m, {info['crs']}, "
                              f"{info['n_nodata']} nodata, {info['bytes']/1024:.0f} KB, {note}, {time.time()-t0:.1f} s")
        return {"FINISHED"}


@_common_props
class DLGIS_OT_difference_raster(bpy.types.Operator, ExportHelper):
    """Write NEW minus EXISTING as a GeoTIFF (positive = fill, negative = cut) and report the cut and fill volumes"""
    bl_idname = "dlgis.difference_raster"
    bl_label = "Difference raster (cut / fill)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".tif"
    filter_glob: StringProperty(default="*.tif;*.tiff", options={"HIDDEN"})
    existing_surface: StringProperty(name="Existing", description="The surface to compare against, for example the imported terrain")
    new_surface: StringProperty(name="New", description="The changed surface: new minus existing, so above = fill (blue), below = cut (red)")
    make_mesh: BoolProperty(name="Show in Blender", default=True,
                            description="Also make a mesh of the difference, coloured red where the new surface cuts and blue where it fills")
    mesh_height: EnumProperty(
        name="Mesh height", description="What the difference mesh's height shows",
        items=[("DIFFERENCE", "Difference", "The mesh is flat at 0 and rises where the new surface fills, sinks where it cuts: the difference itself"),
               ("DRAPED", "On the existing surface", "The mesh follows the existing surface and shows the difference as colour only")],
        default="DIFFERENCE")
    exaggeration: FloatProperty(name="Exaggeration", description="Multiplies the mesh height (1 = metres as they are); the raster and the volumes are never exaggerated",
                                default=1.0, min=0.01, soft_max=20.0)
    make_solids: BoolProperty(name="Make solids", default=True,
                              description="Also make two closed solids between the surfaces: a positive one where the new surface fills (blue) and a negative one where it cuts (red), each with its volume")
    nodata: FloatProperty(name="Nodata value", description="Written where one of the two surfaces has no data", default=-9999.0)
    compression: EnumProperty(name="Compression", items=[("deflate", "Deflate", "Smaller file (lossless)"), ("none", "None", "Uncompressed")], default="deflate")

    def invoke(self, context, event):
        act = context.active_object
        others = [o for o in context.selected_objects if o != act and o.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}]
        if act is not None and not self.new_surface:
            self.new_surface = act.name
        if others and not self.existing_surface:
            self.existing_surface = others[0].name
        if has_grid_facts(act):
            self.cell = float(act["dl_gis_cell_x"])
        stem = f"{self.new_surface or 'new'} minus {self.existing_surface or 'existing'} - difference"
        self.filepath = os.path.join(os.path.dirname(default_export_path(context.scene, act)),
                                     "".join(c for c in stem if c not in '\\/:*?"<>|') + ".tif")
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.prop_search(self, "existing_surface", bpy.data, "objects")
        lay.prop_search(self, "new_surface", bpy.data, "objects")
        lay.prop(self, "cell")
        lay.prop(self, "make_mesh")
        if self.make_mesh:
            lay.prop(self, "mesh_height")
        lay.prop(self, "make_solids")
        box = lay.box()
        box.prop(self, "advanced", toggle=True)
        col = box.column()
        col.enabled = self.advanced
        col.prop(self, "exaggeration")
        col.prop(self, "use_active_grid")
        col.prop(self, "align")
        col.prop(self, "margin")
        col.prop(self, "nodata")
        col.prop(self, "compression")

    def execute(self, context):
        origin = scene_origin(context.scene)
        if origin is None:
            self.report({"ERROR"}, "No scene origin yet: import a GeoTIFF first or set one in the DL GIS panel")
            return {"CANCELLED"}
        ex = bpy.data.objects.get(self.existing_surface)
        nw = bpy.data.objects.get(self.new_surface)
        if ex is None or nw is None:
            self.report({"ERROR"}, "Choose both surfaces: Existing and New")
            return {"CANCELLED"}
        if ex == nw:
            self.report({"ERROR"}, "Existing and New are the same object")
            return {"CANCELLED"}
        adv = self.advanced
        try:
            ul_E, ul_N, rows, cols, dx, dy, note = grid_definition(context, self.cell, self.align if adv else True,
                                                                   self.margin if adv else 0, self.use_active_grid if adv else True, [ex, nw],
                                                                   self.crop_to_terrain, self.terrain or (ex.name if is_raster_grid(ex) else ""))
        except geotiff_io.GeoTiffError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if rows * cols > 64_000_000:
            self.report({"ERROR"}, f"{rows:,} x {cols:,} cells is too many; use a larger cell size")
            return {"CANCELLED"}
        t0 = time.time()
        z_ex = sample_selection(context, [ex], ul_E, ul_N, rows, cols, dx, dy, "HIGHEST")
        z_nw = sample_selection(context, [nw], ul_E, ul_N, rows, cols, dx, dy, "HIGHEST")
        d, stats = difference.difference(z_ex, z_nw, dx, dy)
        if stats["cells_compared"] == 0:
            self.report({"ERROR"}, f"The two surfaces do not overlap on this grid: {cols} x {rows} cells at {dx:g} m from "
                                   f"({ul_E:g}, {ul_N:g}), {note}; {ex.name} hits {int((~np.isnan(z_ex)).sum()):,} cells, "
                                   f"{nw.name} hits {int((~np.isnan(z_nw)).sum()):,}")
            return {"CANCELLED"}
        epsg = origin[2]
        try:
            info = geotiff_io.write_geotiff(self.filepath, d, ul_E, ul_N, dx, dy, epsg,
                                            nodata=self.nodata if adv else -9999.0, compression=self.compression if adv else "deflate",
                                            geokeys=None if epsg else scene_geokeys(context.scene))   # 0.6.3
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not write {os.path.basename(self.filepath)}: {type(e).__name__}: {e}")
            return {"CANCELLED"}
        context.scene[LAST_EXPORT_DIR] = os.path.dirname(os.path.abspath(self.filepath))
        record = dict(stats, existing=ex.name, new=nw.name, ul_E=ul_E, ul_N=ul_N, rows=rows, cols=cols, dx=dx, dy=dy,
                      epsg=info["epsg"], file=os.path.abspath(self.filepath))
        context.scene["dl_gis_last_difference"] = json.dumps(record)

        exag = self.exaggeration if adv else 1.0
        if self.make_mesh:
            if self.mesh_height == "DRAPED":
                z_mesh = np.where(np.isnan(d), np.nan, z_ex)
            else:
                z_mesh = d * exag                      # the difference itself: flat at 0, fill up, cut down
            verts, quads, minfo = meshing.grid_to_quads(z_mesh, ul_E, ul_N, dx, dy, origin[0], origin[1], "HOLES")
            if minfo["n_quads"]:
                name = f"Difference {nw.name} minus {ex.name}"
                me = meshing.mesh_from_quads(name, verts, quads, smooth=True)
                scale = difference.add_attributes(me, d.ravel(), minfo["kept"])
                me.materials.append(difference.cutfill_material())
                ob = link_new_object(context, name, me)
                n_views = difference.show_attribute_colours(context)
                if n_views:
                    self.report({"INFO"}, f"Viewport colour set to Attribute in {n_views} 3D view(s) so the cut / fill colours show")
                store_facts(ob, f"difference: {nw.name} minus {ex.name}", ul_E, ul_N, dx, dy, rows, cols, epsg, origin,
                            -9999.0, "HOLES", "area", 1, [f"cut {stats['cut_m3']:.1f} m3", f"fill {stats['fill_m3']:.1f} m3",
                                                          f"colour scale +-{scale:.2f} m"])
                ob["dl_gis_cut_m3"] = stats["cut_m3"]
                ob["dl_gis_fill_m3"] = stats["fill_m3"]
                ob["dl_gis_net_m3"] = stats["net_m3"]
                ob["dl_gis_area_m2"] = stats["area_m2"]
                ob["dl_gis_colour_scale_m"] = scale
                self.report({"INFO"}, f"Difference mesh: height = {'the difference' if self.mesh_height == 'DIFFERENCE' else 'the existing surface'}"
                                      f"{f' x {exag:g}' if exag != 1.0 else ''}; red = cut, blue = fill (material DL cut fill)")
        if self.make_solids:
            made = []
            for kind, colour in (("Fill", difference.FILL_RGB), ("Cut", difference.CUT_RGB)):
                sv, sf, sinfo = difference.solid_arrays(d, z_ex, z_nw, kind.lower(), ul_E, ul_N, dx, dy, origin[0], origin[1])
                if sinfo["n_quads"] == 0:
                    continue
                sname = f"{kind} solid {nw.name}"[:60]
                sme = vector_mesh.mesh_from_arrays(sname, sv, [], sf)
                vol, manifold = difference.solid_volume(sme)
                sme.materials.append(difference.plain_material(f"DL {kind.lower()} solid", colour))
                sob = link_new_object(context, sname, sme)
                sob["dl_gis_kind"] = "solid"
                sob["dl_gis_solid"] = kind.lower()
                sob["dl_gis_solid_m3"] = vol if kind == "Fill" else -vol
                sob["dl_gis_manifold"] = manifold
                sob["dl_gis_source"] = f"{kind.lower()} solid: {nw.name} minus {ex.name}"
                made.append(f"{kind} solid {vol:,.1f} m3 ({sinfo['n_quads']} cells{'' if manifold else ', NOT watertight'})")
            if made:
                self.report({"INFO"}, "; ".join(made) + f" - solids run between the vertices (cell centres) and ignore differences under {difference.SOLID_MIN_M * 1000:g} mm; the raster volumes count whole cells")
            else:
                self.report({"INFO"}, "No cut or fill: no solids made")
        self.report({"INFO"}, f"{nw.name} minus {ex.name}: cut {stats['cut_m3']:,.1f} m3, fill {stats['fill_m3']:,.1f} m3, "
                              f"net {stats['net_m3']:+,.1f} m3 over {stats['area_m2']:,.0f} m2 "
                              f"({stats['cells_compared']:,} cells at {dx:g} m, {stats['cells_nodata']:,} nodata); "
                              f"{os.path.basename(self.filepath)} written in {time.time()-t0:.1f} s")
        return {"FINISHED"}


_LAYER_ITEMS = {}   # (path, mtime) -> enum items; Blender needs the strings kept alive


def _layer_items(self, context):
    """The layers of the GeoPackage chosen in the file browser (one entry per table), plus 'All layers'."""
    path = self.filepath
    if path and path.lower().endswith(".gpkg") and os.path.isfile(path):
        key = (path, os.path.getmtime(path))
        if key not in _LAYER_ITEMS:
            try:
                ls = gpkg_io.list_layers(path)
            except Exception:  # noqa: BLE001
                ls = []
            items = [("ALL", f"All {len(ls)} layer(s)", "One object per layer")]
            items += [(t, f"{t}  ({g.lower()}, {n:,})", f"{t}: {n:,} features, {g.lower()}, srs {srs}") for t, g, srs, n in ls]
            _LAYER_ITEMS.clear()
            _LAYER_ITEMS[key] = items
        return _LAYER_ITEMS[key]
    return [("ALL", "All layers", "")]


class DLGIS_OT_import_shapefile(bpy.types.Operator, ImportHelper):
    """Import a vector file (.shp, .gpkg, .geojson) as one mesh per layer: points as vertices, lines as edges, polygons as faces, attributes kept"""
    bl_idname = "dlgis.import_shapefile"
    bl_label = "Import vector (.shp, .gpkg, .geojson)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(default="*.shp;*.gpkg;*.geojson;*.json", options={"HIDDEN"})
    layer: EnumProperty(name="Layer", description="Which layer of a GeoPackage to import (a GeoPackage can hold several)", items=_layer_items)
    place_on: StringProperty(
        name="Place on existing object",
        description="For a file that already holds this layer without a scene origin: pick the object the layer should lie on; "
                    "the scene origin is set so that the import lands on it (lower-left corners matched) (0.5.1)")
    faces_mode: EnumProperty(
        name="Faces", description="How polygons become faces (lines and points are not affected)",
        items=[("AUTO", "N-gons, triangles where holes", "One flat face per polygon; a polygon with a hole becomes triangles so the hole stays open (measured: exact in 4.5 and 5.2)"),
               ("TRIANGLES", "Triangles", "Every polygon as triangles (constrained Delaunay, holes open)"),
               ("OUTER", "Outer rings only", "One face per outer ring; holes are filled in")],
        default="AUTO")
    z_mode: EnumProperty(
        name="Height", description="Where the features sit in z (shapefiles are flat)",
        items=[("ZERO", "Flat at 0", "Everything at z = 0"),
               ("DRAPE", "Drape on terrain", "Every vertex dropped onto a terrain object below it: the features follow the ground"),
               ("FIELD", "From a field", "z from a numeric attribute of the file, for example a height column")],
        default="ZERO")
    drape_target: StringProperty(name="Terrain", description="The object the features are dropped onto")
    z_field: StringProperty(name="Field", description="The numeric attribute used as z")
    advanced: BoolProperty(name="Advanced", description="Show the settings below", default=False)
    z_scale: FloatProperty(name="Field scale", description="Multiplies the field value (1 = metres as they are)", default=1.0)
    keep_attributes: BoolProperty(name="Keep attributes", description="The attribute table's columns become mesh attributes", default=True)
    frame_view: BoolProperty(name="Frame the view", description="Centre the 3D view on the import and extend the view clip end so the whole layer is drawn (0.6.5)", default=True)
    origin_mode: EnumProperty(
        name="Scene origin", description="Where local (0, 0) lies in map coordinates",
        items=[("KEEP", "Keep", "Keep the scene origin set earlier, so layers line up"),
               ("RESET", "Reset to this file", "Put the scene origin at the centre of this file's extent")],
        default="KEEP")

    def invoke(self, context, event):
        act = context.active_object
        if act is not None and act.type == "MESH" and not self.drape_target and "dl_gis_ul_E" in act:
            self.drape_target = act.name
        return ImportHelper.invoke(self, context, event)

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        if self.filepath.lower().endswith(".gpkg"):
            lay.prop(self, "layer")
        if scene_origin(context.scene) is None or self.place_on:
            lay.prop_search(self, "place_on", bpy.data, "objects")
        lay.prop(self, "faces_mode")
        lay.prop(self, "z_mode")
        if self.z_mode == "DRAPE":
            lay.prop_search(self, "drape_target", bpy.data, "objects")
        elif self.z_mode == "FIELD":
            lay.prop(self, "z_field")
        box = lay.box()
        box.prop(self, "advanced", toggle=True)
        col = box.column()
        col.enabled = self.advanced
        col.prop(self, "keep_attributes")
        col.prop(self, "z_scale")
        col.prop(self, "origin_mode")
        col.prop(self, "frame_view")

    def read_layers(self, context, path):
        """[(object name, Layer)] for the file: a shapefile, a GeoJSON, or one / all layers of a GeoPackage (0.5.0)."""
        stem = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".gpkg":
            names = [t for t, _, _, _ in gpkg_io.list_layers(path)]
            if not names:
                raise vector_io.VectorError("The GeoPackage has no vector layers")
            chosen = names if self.layer in ("", "ALL") or self.layer not in names else [self.layer]
            return [(f"{stem} - {t}" if len(names) > 1 else stem, gpkg_io.read_gpkg(path, t)) for t in chosen]
        if ext in (".geojson", ".json"):
            o = scene_origin(context.scene)
            return [(stem, geojson_io.read_geojson(path, o[2] if o else None))]
        return [(stem, vector_io.read_shapefile(path))]

    def execute(self, context):
        path = self.filepath
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            layers = self.read_layers(context, path)
        except vector_io.VectorError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        except Exception as ex:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not read {os.path.basename(path)}: {type(ex).__name__}: {ex}")
            return {"CANCELLED"}
        layers = [(n, l) for n, l in layers if l.features]
        if not layers:
            self.report({"ERROR"}, "The file has no features")
            return {"CANCELLED"}
        scene = context.scene
        origin = scene_origin(scene)
        origin_mode = self.origin_mode if self.advanced else "KEEP"
        first = layers[0][1]
        extra_notes = []
        n_meshes_before = sum(1 for o in scene.objects if o.type == "MESH" and not is_startup_cube(o))   # 0.6.5: not the startup cube
        if self.place_on:                                  # 0.5.1: the origin that lands this layer on an existing object
            target = bpy.data.objects.get(self.place_on)
            if target is None:
                self.report({"ERROR"}, f"Object '{self.place_on}' to place the layer on was not found")
                return {"CANCELLED"}
            oe_new, on_new, b = origin_from_object(context, target, first.bbox[0], first.bbox[1])
            if oe_new is None:
                self.report({"ERROR"}, f"{target.name} has no extent to place the layer on")
                return {"CANCELLED"}
            if origin is not None and (abs(origin[0] - oe_new) > 1e-3 or abs(origin[1] - on_new) > 1e-3):
                extra_notes.append(f"scene origin moved from E {origin[0]:,.3f} N {origin[1]:,.3f} to E {oe_new:,.3f} N {on_new:,.3f}: "
                                   f"objects imported with the old origin keep their local place and no longer match their map position")
            set_scene_origin(scene, oe_new, on_new, first.epsg or (origin[2] if origin else None))
            origin = scene_origin(scene)
            extra_notes.append(f"scene origin set to E {oe_new:,.3f} N {on_new:,.3f} so that {first.source} lies on {target.name}")
        elif origin is None or origin_mode == "RESET":
            ce = round((first.bbox[0] + first.bbox[2]) / 2.0)
            cn = round((first.bbox[1] + first.bbox[3]) / 2.0)
            set_scene_origin(scene, ce, cn, first.epsg, crs_keys=None)   # 0.6.3: a vector origin carries no GeoKeys
            origin = scene_origin(scene)
            if n_meshes_before and origin_mode != "RESET":
                extra_notes.append(NO_ORIGIN_WARNING.format(n=n_meshes_before, e=ce, nn=cn))
        for n in extra_notes:
            self.report({"WARNING"} if "no scene origin" in n or "moved" in n else {"INFO"}, n)
        last = None
        for name, layer in layers:
            layer.notes = list(layer.notes) + extra_notes
            res = self.import_layer(context, path, name, layer, origin)
            if res != {"FINISHED"}:
                return res
            last = context.active_object
        if len(layers) > 1:
            self.report({"INFO"}, f"{len(layers)} layers imported from {os.path.basename(path)}")
        if last is not None:
            context.view_layer.objects.active = last
            if self.frame_view:
                frame_view_on(context, last)
        return {"FINISHED"}

    def import_layer(self, context, path, stem, layer, origin):
        scene = context.scene
        oe, on, scene_epsg = origin
        if layer.epsg and scene_epsg and layer.epsg != scene_epsg:
            self.report({"WARNING"}, f"{layer.source} is EPSG:{layer.epsg} but the scene origin is EPSG:{scene_epsg}: positions will not line up")

        t0 = time.time()
        z_scale = self.z_scale if self.advanced else 1.0
        z_mode = self.z_mode
        try:
            verts, edges, faces, v_fid, e_fid, f_fid = vector_mesh.build_layer(layer, oe, on, self.faces_mode, z_mode, self.z_field, z_scale)
        except Exception as ex:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not build the mesh: {type(ex).__name__}: {ex}")
            return {"CANCELLED"}
        notes = list(layer.notes)
        if z_mode == "DRAPE":
            target = bpy.data.objects.get(self.drape_target)
            if target is None:
                self.report({"WARNING"}, "No terrain chosen to drape onto: features left flat at 0")
            else:
                z = sampling.sample_points([target], context.evaluated_depsgraph_get(), verts[:, :2], "HIGHEST")
                miss = int(np.isnan(z).sum())
                verts[:, 2] = np.where(np.isnan(z), 0.0, z)
                if miss:
                    notes.append(f"{miss} vertices found no terrain below them and stay at z 0")
        me = vector_mesh.mesh_from_arrays(stem, verts, edges, faces)
        ob = link_new_object(context, stem, me)
        n_att = 0
        if self.keep_attributes if self.advanced else True:
            n_att = vector_mesh.add_attributes(me, layer, {"POINT": v_fid, "EDGE": e_fid, "FACE": f_fid})
            if n_att == 0:
                notes.append("attributes not written: the mesh was repaired and its element counts changed")
        store_facts(ob, layer.source, layer.bbox[0], layer.bbox[3], 0.0, 0.0, len(layer.features), layer.n_points, layer.epsg,
                    origin, None, "-", "vector", 1, notes)
        ob["dl_gis_kind"] = "vector"
        ob["dl_gis_geometry"] = layer.kind
        ob["dl_gis_crs_name"] = layer.crs_name
        ob["dl_gis_fields"] = ", ".join(f[0] for f in layer.fields)
        ob["dl_gis_encoding"] = layer.encoding
        scene[LAST_IMPORT_DIR] = os.path.dirname(os.path.abspath(path))
        for n in notes:
            self.report({"INFO"}, n)
        self.report({"INFO"}, f"{layer.source}: {len(layer.features):,} {layer.kind} features -> {len(me.vertices):,} vertices, "
                              f"{len(me.edges):,} edges, {len(me.polygons):,} faces, {n_att} attributes, EPSG:{layer.epsg or '?'}, {time.time()-t0:.1f} s")
        return {"FINISHED"}


class DLGIS_OT_set_origin(bpy.types.Operator):
    """Set where local (0, 0) lies in map coordinates, for scenes that did not start from an import"""
    bl_idname = "dlgis.set_origin"
    bl_label = "Set scene origin"
    bl_options = {"REGISTER", "UNDO"}

    # Text, not numbers: a FloatProperty is single precision and turns 7700000.75 into 7700001.0 (gate X7, 2026-10-09)
    east: StringProperty(name="East (E)", description="Map easting of the scene's (0, 0) in metres, for example 655100", default="0")
    north: StringProperty(name="North (N)", description="Map northing of the scene's (0, 0) in metres, for example 7738130", default="0")
    epsg: IntProperty(name="EPSG code", description="The projected coordinate reference system, for example 25833 for UTM zone 33N", default=25833, min=0, max=32766)

    def invoke(self, context, event):
        o = scene_origin(context.scene)
        if o:
            self.east, self.north, self.epsg = f"{o[0]:.3f}".rstrip("0").rstrip("."), f"{o[1]:.3f}".rstrip("0").rstrip("."), o[2] or 25833
        return context.window_manager.invoke_props_dialog(self, width=320)

    def execute(self, context):
        try:
            e = float(self.east.replace(",", ".").replace(" ", ""))
            n = float(self.north.replace(",", ".").replace(" ", ""))
        except ValueError:
            self.report({"ERROR"}, f"East and North must be numbers in metres (got {self.east!r}, {self.north!r})")
            return {"CANCELLED"}
        set_scene_origin(context.scene, e, n, self.epsg)
        self.report({"INFO"}, f"Scene origin E {e:.3f} N {n:.3f}, EPSG:{self.epsg}")
        return {"FINISHED"}


class DLGIS_OT_origin_from_file(bpy.types.Operator, ImportHelper):
    """Set the scene origin to the lower-left corner of a GIS file (a GeoTIFF's south-west corner, a vector file's extent), without typing numbers"""
    bl_idname = "dlgis.origin_from_file"
    bl_label = "Scene origin from a file's corner"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(default="*.tif;*.tiff;*.shp;*.gpkg;*.geojson;*.json", options={"HIDDEN"})

    def draw(self, context):
        lay = self.layout
        lay.label(text="The file's lower-left corner becomes local (0, 0).")
        lay.label(text="For example the outline or frame file of your site.")

    def execute(self, context):
        path = self.filepath
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            e, n, epsg, what, keys, kname = file_corner(path)
        except (geotiff_io.GeoTiffError, vector_io.VectorError) as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        except Exception as ex:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not read {os.path.basename(path)}: {type(ex).__name__}: {ex}")
            return {"CANCELLED"}
        old = scene_origin(context.scene)
        set_scene_origin(context.scene, e, n, epsg or (old[2] if old else None), crs_keys=keys if keys else "keep", crs_name=kname)
        if old is not None and (abs(old[0] - e) > 1e-3 or abs(old[1] - n) > 1e-3):
            n_imp = sum(1 for o in context.scene.objects if "dl_gis_source" in o)
            self.report({"WARNING"}, f"Scene origin moved from E {old[0]:,.3f} N {old[1]:,.3f}: {n_imp} object(s) imported with the old origin keep their local place")
        self.report({"INFO"}, f"Scene origin E {e:,.3f} N {n:,.3f}, EPSG:{epsg or old[2] if old else epsg or '?'} = {what}")
        return {"FINISHED"}


# ------------------------------------------------------------------------------------------------ UI
class DLGIS_OT_resample_grid(bpy.types.Operator):
    """Resample the active terrain grid to a new cell size (0.10-100 m), keeping its corner: finer cells by bilinear
    interpolation (no new measured detail), coarser cells by the area-weighted mean (0.7.0)"""
    bl_idname = "dlgis.resample_grid"
    bl_label = "Resample grid"
    bl_options = {"REGISTER", "UNDO"}

    cell: FloatProperty(name="New cell size (m)", description="Edge length of the new cells, 0.10 to 100 m", default=1.0,
                        min=0.1, max=100.0, soft_min=0.1, soft_max=100.0, step=10, precision=2)
    method: EnumProperty(
        name="Method",
        items=[("AUTO", "Auto", "Bilinear when the new cells are finer, area-weighted mean when they are coarser"),
               ("BILINEAR", "Bilinear", "Interpolate between the cell centres (for finer cells)"),
               ("AREA_MEAN", "Area mean", "Mean of the overlapped cells, weighted by area (for coarser cells; keeps the volume)")],
        default="AUTO")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH" and is_raster_grid(ob) and "dl_gis_cut_m3" not in ob

    def invoke(self, context, event):
        self.cell = float(context.active_object["dl_gis_cell_x"])
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        ob = context.active_object
        lay = self.layout
        lay.prop(self, "cell")
        lay.prop(self, "method")
        d0, rows, cols = float(ob["dl_gis_cell_x"]), int(ob["dl_gis_rows"]), int(ob["dl_gis_cols"])
        r1, c1, nested = resample.target_size(rows, cols, d0, self.cell)
        box = lay.box()
        box.label(text=f"source {cols:,} x {rows:,} cells of {d0:g} m")
        box.label(text=f"new {c1:,} x {r1:,} = {r1 * c1 / 1e6:,.2f} million cells of {self.cell:g} m"
                       + ("" if nested else " (not nested)"), icon="ERROR" if r1 * c1 > resample.MAX_CELLS else "NONE")
        if self.cell < d0 - 1e-12:
            box.label(text="finer cells are interpolated: no new measured detail", icon="INFO")

    def execute(self, context):
        ob = context.active_object
        if ob is None or not is_raster_grid(ob):
            self.report({"ERROR"}, "Select an imported terrain grid (or a grid from 'Grid from selection')")
            return {"CANCELLED"}
        d0 = float(ob["dl_gis_cell_x"])
        if abs(float(ob.get("dl_gis_cell_y", d0)) - d0) > 1e-9:
            self.report({"ERROR"}, "Only square cells can be resampled")
            return {"CANCELLED"}
        if not grid_facts_hold(ob, context):
            self.report({"ERROR"}, "This grid was moved since it was imported; use 'Grid from selection' to sample it onto a new grid")
            return {"CANCELLED"}
        rows, cols = int(ob["dl_gis_rows"]), int(ob["dl_gis_cols"])
        ul_E, ul_N = float(ob["dl_gis_ul_E"]), float(ob["dl_gis_ul_N"])
        corners = str(ob.get("dl_gis_vertex_mode", "centres")) == "corners"
        oe, on = float(ob["dl_gis_origin_E"]), float(ob["dl_gis_origin_N"])
        me = ob.data
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(ob.matrix_world)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        t0 = time.time()
        try:
            z0 = resample.cells_from_lattice(co[:, :2], co[:, 2], (oe, on), ul_E, ul_N, d0, rows, cols, corners)
            z1, used, r1, c1, nested = resample.resample(z0, d0, float(self.cell), self.method)
        except resample.ResampleError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}
        d1 = float(self.cell)
        if corners:
            verts, quads, info = meshing.grid_to_quads(meshing.corner_heights(z1.astype(np.float32)), ul_E - d1 / 2.0, ul_N + d1 / 2.0,
                                                       d1, d1, oe, on, "HOLES")
        else:
            verts, quads, info = meshing.grid_to_quads(z1.astype(np.float32), ul_E, ul_N, d1, d1, oe, on, "HOLES")
        if info["n_quads"] == 0:
            self.report({"ERROR"}, "The resampled grid has no cells with data")
            return {"CANCELLED"}
        notes = []
        if used == "BILINEAR":
            what = f"interpolated (bilinear) from {d0:g} m cells: no new measured detail"
        elif used == "AREA_MEAN":
            what = f"area-weighted mean of {d0:g} m cells"
        else:
            what = f"copied ({d0:g} m)"
        notes.append(what)
        if not nested:
            notes.append(f"{d1:g} m does not nest in {d0:g} m: the last column / row reaches past the source extent")
        v0 = float(np.nansum(z0)) * d0 * d0
        v1 = float(np.nansum(z1)) * d1 * d1
        notes.append(f"volume (heights x cell area) before {v0:,.1f} m3, after {v1:,.1f} m3"
                     + (f" ({(v1 - v0) / v0 * 100:+.4f} %)" if v0 else ""))
        name = f"{ob.name} - resampled {d1:g} m"
        new = link_new_object(context, name, meshing.mesh_from_quads(name, verts, quads, smooth=True))
        store_facts(new, f"{ob['dl_gis_source']} resampled to {d1:g} m", ul_E, ul_N, d1, d1, r1, c1,
                    int(ob.get("dl_gis_epsg", 0)) or None, (oe, on), float("nan"), "HOLES",
                    str(ob.get("dl_gis_raster_type", "area")), 1, notes)
        new["dl_gis_vertex_mode"] = "corners" if corners else "centres"
        new["dl_gis_resampled"] = what
        for n in notes:
            self.report({"INFO"}, n)
        self.report({"INFO"}, f"{name}: {c1:,} x {r1:,} cells, {info['n_verts']:,} vertices, {time.time() - t0:.1f} s")
        return {"FINISHED"}


class DLGIS_PT_sidebar(bpy.types.Panel):
    bl_label = "DL GIS Exchange"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "DL GIS"

    def draw(self, context):
        lay = self.layout
        col = lay.column(align=True)
        col.operator(DLGIS_OT_import_geotiff.bl_idname, text="Import raster (.tif .asc .xyz)", icon="IMPORT")
        col.operator(DLGIS_OT_import_shapefile.bl_idname, text="Import vector (.shp .gpkg .geojson)", icon="OUTLINER_DATA_CURVE")
        col.operator(DLGIS_OT_grid_from_selection.bl_idname, text="Grid from selection", icon="MESH_GRID")
        col.operator(DLGIS_OT_resample_grid.bl_idname, text="Resample grid (cell size)", icon="MOD_REMESH")
        col.operator(DLGIS_OT_export_geotiff.bl_idname, text="Export GeoTIFF", icon="EXPORT")
        col.operator(DLGIS_OT_difference_raster.bl_idname, text="Difference raster (cut / fill)", icon="MOD_BOOLEAN")
        o = scene_origin(context.scene)
        box = lay.box()
        row = box.row()
        row.label(text="Scene origin", icon="WORLD")
        row.operator(DLGIS_OT_origin_from_file.bl_idname, text="", icon="FILE_FOLDER")
        row.operator(DLGIS_OT_set_origin.bl_idname, text="", icon="GREASEPENCIL")
        if o is None:
            box.label(text="none yet: by the first import,")
            box.label(text="a file's corner (folder) or numbers (pencil)")
        else:
            box.label(text=f"E {o[0]:,.2f}   N {o[1]:,.2f}")
            box.label(text=f"EPSG:{o[2]}" if o[2] else (f"user-defined CRS {context.scene.get(CRS_NAME, '')}".strip()
                                                         if context.scene.get(CRS_KEYS) else "CRS unknown"))
        ob = context.active_object
        if ob is not None and "dl_gis_source" in ob:
            box = lay.box()
            if ob.get("dl_gis_kind") == "vector":
                box.label(text=ob["dl_gis_source"], icon="OUTLINER_DATA_CURVE")
                box.label(text=f"{ob['dl_gis_rows']:,} {ob.get('dl_gis_geometry', '')} features, {ob['dl_gis_cols']:,} points")
                box.label(text=f"EPSG:{ob['dl_gis_epsg'] or '?'}  {ob.get('dl_gis_crs_name', '')}")
                return
            if ob.get("dl_gis_kind") == "solid":          # 0.6.0: a solid has no grid facts (0.4.3-0.5.3 raised KeyError here)
                box.label(text=ob["dl_gis_source"], icon="MESH_CUBE")
                box.label(text=f"{ob.get('dl_gis_solid', '')} solid {float(ob.get('dl_gis_solid_m3', 0.0)):+,.1f} m3")
                box.label(text="watertight" if ob.get("dl_gis_manifold", True) else "NOT watertight - check before 3D printing")
                return
            box.label(text=ob["dl_gis_source"], icon="MESH_GRID")
            box.label(text=f"{ob['dl_gis_cols']} x {ob['dl_gis_rows']} cells, {ob['dl_gis_cell_x']:g} m")
            vm = ob.get("dl_gis_vertex_mode")
            if vm == "corners":
                box.label(text=f"vertices at cell corners: {ob['dl_gis_cols'] * ob['dl_gis_cell_x']:g} x {ob['dl_gis_rows'] * ob['dl_gis_cell_y']:g} m")
            elif vm == "centres":
                box.label(text=f"vertices at cell centres: {(ob['dl_gis_cols'] - 1) * ob['dl_gis_cell_x']:g} x {(ob['dl_gis_rows'] - 1) * ob['dl_gis_cell_y']:g} m")
            box.label(text=f"corner E {ob['dl_gis_ul_E']:,.1f}  N {ob['dl_gis_ul_N']:,.1f}")
            if "dl_gis_resampled" in ob:                     # 0.7.0
                box.label(text=str(ob["dl_gis_resampled"]), icon="INFO")
            if ob.get("dl_gis_step", 1) > 1:
                box.label(text=f"step {ob['dl_gis_step']} (every {ob['dl_gis_step']}. cell)")
            if "dl_gis_cut_m3" in ob:
                box.label(text=f"cut {ob['dl_gis_cut_m3']:,.1f} m3   fill {ob['dl_gis_fill_m3']:,.1f} m3")
                box.label(text=f"net {ob['dl_gis_net_m3']:+,.1f} m3 over {ob['dl_gis_area_m2']:,.0f} m2")


def menu_import(self, context):
    self.layout.operator(DLGIS_OT_import_geotiff.bl_idname, text="Terrain raster (.tif .asc .xyz) - DL GIS Exchange")
    op = self.layout.operator(DLGIS_OT_import_shapefile.bl_idname, text="Shapefile points, lines, polygons (.shp) - DL GIS Exchange")
    op.filter_glob = "*.shp"
    op = self.layout.operator(DLGIS_OT_import_shapefile.bl_idname, text="GeoPackage layers (.gpkg) - DL GIS Exchange")
    op.filter_glob = "*.gpkg"
    op = self.layout.operator(DLGIS_OT_import_shapefile.bl_idname, text="GeoJSON (.geojson) - DL GIS Exchange")
    op.filter_glob = "*.geojson;*.json"


def menu_export(self, context):
    self.layout.operator(DLGIS_OT_export_geotiff.bl_idname, text="GeoTIFF raster (.tif) - DL GIS Exchange")
    self.layout.operator(DLGIS_OT_difference_raster.bl_idname, text="GeoTIFF difference, cut / fill (.tif) - DL GIS Exchange")


CLASSES = (DLGIS_OT_import_geotiff, DLGIS_OT_import_shapefile, DLGIS_OT_origin_from_file, DLGIS_OT_grid_from_selection, DLGIS_OT_export_geotiff,
           DLGIS_OT_difference_raster, DLGIS_OT_set_origin, DLGIS_OT_resample_grid, DLGIS_PT_sidebar)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
