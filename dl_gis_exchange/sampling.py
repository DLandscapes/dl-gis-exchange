# DL GIS Exchange - sampling.py - Digital Landscapes, 2026. Licence: GPL-3.0-or-later.
# SPDX-FileCopyrightText: 2026 Marc Ihle (Digital Landscapes)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Heights of Blender geometry on a regular grid, by ray-casting (BVHTree) - the reverse direction.

A ray per cell centre, straight down from above the selection (mode HIGHEST: the first surface a drop would meet)
or straight up from below (mode LOWEST: the underside). Every selected object is cast separately in its own local
space (BVHTree.FromObject + the inverse world matrix) and the best hit across objects wins. A cell that no ray
hits is nodata (NaN). A ray that grazes an outer edge or a vertex can miss (DL-TerrainFill probe 135, 2026-10):
a miss is retried with four small offsets before it counts as nodata.
"""
from __future__ import annotations

import numpy as np

HELPER = "DL GIS realize (temporary)"


def instance_parents(objects, depsgraph):
    """0.5.3: the subset of objects whose evaluated result contains INSTANCES (Geometry Nodes Instance on Points,
    collection instances, particle / dupli instances). BVHTree.FromObject sees only an object's own evaluated mesh,
    which for such an object can be empty (a DL Terrain Select demo, 2026-10-17: the terrain replaced by
    213,624 cube instances -> every ray missed, "No cell of the grid lies over a selected object")."""
    wanted = {ob.original if hasattr(ob, "original") else ob for ob in objects}
    found = set()
    for inst in depsgraph.object_instances:
        if inst.is_instance and inst.parent is not None:
            p = inst.parent.original
            if p in wanted:
                found.add(p)
                if len(found) == len(wanted):
                    break
    return found


class Realized:
    """Context manager: for every object with instances, a temporary helper object whose Geometry Nodes modifier
    (Object Info, relative space, As Instance off -> Realize Instances) holds the object's evaluated geometry with
    all instances realized, in world space (the helper sits at the identity). Everything it creates is removed on
    exit, also after an error. Use: with Realized(objects, context) as R: R.caster(ob) -> (eval object, matrix)."""

    def __init__(self, objects, context=None):
        import bpy
        self.bpy = bpy
        self.context = context or bpy.context
        self.objects = [o for o in objects if o is not None]
        self.helpers = {}
        self.made = []

    def __enter__(self):
        bpy = self.bpy
        deps = self.context.evaluated_depsgraph_get()
        need = instance_parents(self.objects, deps)
        if not need:
            self.deps = deps
            return self
        scene = self.context.scene
        for ob in self.objects:
            if ob not in need:
                continue
            ng = bpy.data.node_groups.new(HELPER, "GeometryNodeTree")
            ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            info = ng.nodes.new("GeometryNodeObjectInfo")
            info.transform_space = "RELATIVE"
            info.inputs["Object"].default_value = ob
            if "As Instance" in info.inputs:
                info.inputs["As Instance"].default_value = False
            real = ng.nodes.new("GeometryNodeRealizeInstances")
            out = ng.nodes.new("NodeGroupOutput")
            ng.links.new(info.outputs["Geometry"], real.inputs["Geometry"])
            ng.links.new(real.outputs["Geometry"], out.inputs[0])
            me = bpy.data.meshes.new(HELPER)
            helper = bpy.data.objects.new(HELPER, me)
            scene.collection.objects.link(helper)
            mod = helper.modifiers.new("realize", "NODES")
            mod.node_group = ng
            helper.hide_render = True
            self.helpers[ob] = helper
            self.made.append((helper, me, ng))
        self.context.view_layer.update()
        self.deps = self.context.evaluated_depsgraph_get()
        return self

    def __exit__(self, *exc):
        bpy = self.bpy
        for helper, me, ng in self.made:
            try:
                bpy.data.objects.remove(helper, do_unlink=True)
            except (ReferenceError, RuntimeError):
                pass
            try:
                bpy.data.meshes.remove(me)
            except (ReferenceError, RuntimeError):
                pass
            try:
                bpy.data.node_groups.remove(ng)
            except (ReferenceError, RuntimeError):
                pass
        if self.made:
            self.context.view_layer.update()
        return False

    def caster(self, ob):
        """(object to cast against, its world matrix): the helper for an object with instances, else the object."""
        h = self.helpers.get(ob)
        return (h, h.matrix_world.copy()) if h is not None else (ob, ob.matrix_world.copy())

    @property
    def n_realized(self):
        return len(self.helpers)


def selection_box(objects, depsgraph, realized=None):
    """World-space (xmin, xmax, ymin, ymax, zmin, zmax) over the EVALUATED objects, or None. Objects with instances
    count with their realized geometry (0.5.3): without it their bound box can be empty (all zeros)."""
    from mathutils import Vector
    if realized is None and instance_parents(objects, depsgraph):
        with Realized(objects) as R:
            return selection_box(objects, R.deps, R)
    pts = []
    for ob in objects:
        if realized is not None and ob in realized.helpers:
            h = realized.helpers[ob].evaluated_get(depsgraph)
            pts.extend(h.matrix_world @ Vector(c) for c in h.bound_box)
            continue
        ev = ob.evaluated_get(depsgraph)
        if ob.type == "EMPTY":
            pts.append(ev.matrix_world.translation.copy())
        else:
            pts.extend(ev.matrix_world @ Vector(c) for c in ev.bound_box)
    if not pts:
        return None
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def sample_heights(objects, depsgraph, xs, ys, mode="HIGHEST", z_lo=None, z_hi=None, retry=1e-4):
    """Heights (rows, cols) of the objects at local-scene centres xs (cols) x ys (rows); NaN where nothing is hit.
    Objects that produce instances are cast against their realized geometry (0.5.3)."""
    with Realized(objects) as R:
        return _sample_heights(objects, R, xs, ys, mode, z_lo, z_hi, retry)


def _sample_heights(objects, R, xs, ys, mode, z_lo, z_hi, retry):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    depsgraph = R.deps
    casters = []
    for ob in objects:
        if ob.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and ob not in R.helpers:
            continue
        target, mw = R.caster(ob)
        tree = BVHTree.FromObject(target, depsgraph)
        casters.append((tree, mw.inverted(), mw))
    rows, cols = len(ys), len(xs)
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    if not casters:
        return out
    if z_lo is None or z_hi is None:
        box = selection_box(objects, depsgraph, R)
        z_lo = box[4] - 1.0 if z_lo is None else z_lo
        z_hi = box[5] + 1.0 if z_hi is None else z_hi
    down = mode == "HIGHEST"
    z0 = z_hi + 1.0 if down else z_lo - 1.0
    dir_world = Vector((0.0, 0.0, -1.0 if down else 1.0))
    span = abs(z_hi - z_lo) + 4.0
    # per caster: local ray direction (normalised) and the local distance that covers the span
    prepared = []
    for tree, inv, mw in casters:
        d_loc = (inv.to_3x3() @ dir_world)
        scale = d_loc.length
        prepared.append((tree, inv, mw, d_loc.normalized(), span * scale + 1.0))
    offsets = ((0.0, 0.0), (retry, retry), (-retry, retry), (retry, -retry), (-retry, -retry))

    def cast(x, y):
        best = None
        for tree, inv, mw, d, dist in prepared:
            o = inv @ Vector((x, y, z0))
            hit = tree.ray_cast(o, d, dist)
            if hit[0] is not None:
                z = (mw @ hit[0]).z
                if best is None or (z > best if down else z < best):
                    best = z
        return best

    for r, y in enumerate(ys):
        row = out[r]
        for c, x in enumerate(xs):
            best = None
            for ox, oy in offsets:
                best = cast(x + ox, y + oy)
                if best is not None:
                    break
            if best is not None:
                row[c] = best
    return out


def sample_points(objects, depsgraph, xy, mode="HIGHEST", retry=1e-4):
    """Heights of the objects under / over arbitrary local (x, y) points (n, 2); NaN where nothing is hit.
    Same casting as sample_heights; used to drape imported vector features onto a terrain (0.4.0); objects with
    instances are cast against their realized geometry (0.5.3)."""
    with Realized(objects) as R:
        return _sample_points(objects, R, xy, mode, retry)


def _sample_points(objects, R, xy, mode, retry):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    depsgraph = R.deps
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.full(len(xy), np.nan, dtype=np.float64)
    casters = []
    for ob in objects:
        if ob.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and ob not in R.helpers:
            continue
        target, mw = R.caster(ob)
        casters.append((BVHTree.FromObject(target, depsgraph), mw.inverted(), mw))
    if not casters or not len(xy):
        return out
    box = selection_box(objects, depsgraph, R)
    z_lo, z_hi = box[4] - 1.0, box[5] + 1.0
    down = mode == "HIGHEST"
    z0 = z_hi + 1.0 if down else z_lo - 1.0
    dir_world = Vector((0.0, 0.0, -1.0 if down else 1.0))
    span = abs(z_hi - z_lo) + 4.0
    prepared = []
    for tree, inv, mw in casters:
        d_loc = inv.to_3x3() @ dir_world
        prepared.append((tree, inv, mw, d_loc.normalized(), span * d_loc.length + 1.0))
    offsets = ((0.0, 0.0), (retry, retry), (-retry, retry), (retry, -retry), (-retry, -retry))
    for i, (x, y) in enumerate(xy):
        for ox, oy in offsets:
            best = None
            for tree, inv, mw, d, dist in prepared:
                hit = tree.ray_cast(inv @ Vector((x + ox, y + oy, z0)), d, dist)
                if hit[0] is not None:
                    z = (mw @ hit[0]).z
                    if best is None or (z > best if down else z < best):
                        best = z
            if best is not None:
                out[i] = best
                break
    return out
