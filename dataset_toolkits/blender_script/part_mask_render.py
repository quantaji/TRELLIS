import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import *

import bpy
import numpy as np
from mathutils import Matrix, Vector

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    "PNG": "png",
    "JPEG": "jpg",
    "OPEN_EXR": "exr",
    "TIFF": "tiff",
    "BMP": "bmp",
    "HDR": "hdr",
    "TARGA": "tga",
}


def init_render(engine="CYCLES", resolution=512, geo_mode=False) -> None:
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100

    # MOD: match blender_render_img_mask.set_color_output(): WEBP + quality=100 + RGBA + transparent film
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.quality = 100
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.use_compositing = True

    bpy.context.scene.cycles.device = "GPU"
    # MOD: match blender_render_img_mask RENDER_SAMPLES=64 (geo_mode keeps 1)
    bpy.context.scene.cycles.samples = 64 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = "BOX"
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"


def init_nodes(
    output_folder: str,
    save_depth: bool = False,
    save_normal: bool = False,
    save_albedo: bool = False,
    save_mist: bool = False,
    save_mask: bool = False,
) -> Tuple[dict, dict]:
    # MOD: include save_mask in the early-exit condition
    if not any([save_depth, save_normal, save_albedo, save_mist, save_mask]):
        return {}, {}
    outputs = {}
    spec_nodes = {}

    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = save_depth
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = save_normal
    bpy.context.scene.view_layers["ViewLayer"].use_pass_mist = save_mist
    bpy.context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers["ViewLayer"].use_pass_cryptomatte_object = save_mask

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)

    render_layers = nodes.new("CompositorNodeRLayers")

    if save_depth:
        depth_file_output = nodes.new("CompositorNodeOutputFile")
        depth_file_output.base_path = output_folder
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = "PNG"
        depth_file_output.format.color_depth = "16"
        depth_file_output.format.color_mode = "BW"
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10  # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)

        links.new(render_layers.outputs["Depth"], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])

        outputs["depth"] = depth_file_output
        spec_nodes["depth_map"] = map

    if save_normal:
        normal_file_output = nodes.new("CompositorNodeOutputFile")
        normal_file_output.base_path = output_folder
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = "OPEN_EXR"
        normal_file_output.format.color_mode = "RGB"
        normal_file_output.format.color_depth = "16"

        links.new(render_layers.outputs["Normal"], normal_file_output.inputs[0])

        outputs["normal"] = normal_file_output

    if save_albedo:
        albedo_file_output = nodes.new("CompositorNodeOutputFile")
        albedo_file_output.base_path = output_folder
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = "PNG"
        albedo_file_output.format.color_mode = "RGBA"
        albedo_file_output.format.color_depth = "8"

        alpha_albedo = nodes.new("CompositorNodeSetAlpha")

        links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
        links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])
        links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

        outputs["albedo"] = albedo_file_output

    if save_mist:
        bpy.data.worlds["World"].mist_settings.start = 0
        bpy.data.worlds["World"].mist_settings.depth = 10

        mist_file_output = nodes.new("CompositorNodeOutputFile")
        mist_file_output.base_path = output_folder
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = "PNG"
        mist_file_output.format.color_mode = "BW"
        mist_file_output.format.color_depth = "16"

        links.new(render_layers.outputs["Mist"], mist_file_output.inputs[0])

        outputs["mist"] = mist_file_output

    if save_mask:

        part_objs = sorted(
            [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name.startswith("part_")],
            key=lambda o: o.name,
        )

        # 逐个 part：matte -> (matte > 0) -> * part_id -> max 归约
        acc = None
        for idx, obj in enumerate(part_objs):
            part_id = float(idx + 1)  # 与原版 format_mask_output 的 i+1 对齐（背景0）:contentReference[oaicite:8]{index=8}

            crypto = nodes.new("CompositorNodeCryptomatteV2")
            crypto.matte_id = obj.name

            # 重要：给 Cryptomatte 一个 Image 输入（让它从渲染结果里读 crypto 元数据）
            links.new(render_layers.outputs["Image"], crypto.inputs["Image"])

            gt = nodes.new("CompositorNodeMath")
            gt.operation = "GREATER_THAN"
            gt.inputs[1].default_value = 0.0
            links.new(crypto.outputs["Matte"], gt.inputs[0])

            mul = nodes.new("CompositorNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = part_id
            links.new(gt.outputs[0], mul.inputs[0])

            if acc is None:
                acc = mul
            else:
                mx = nodes.new("CompositorNodeMath")
                mx.operation = "MAXIMUM"
                links.new(acc.outputs[0], mx.inputs[0])
                links.new(mul.outputs[0], mx.inputs[1])
                acc = mx

        # acc 输出是单通道 Value。为了“3通道 mask”，复制到 RGB。
        combine = nodes.new("CompositorNodeCombRGBA")
        combine.inputs["A"].default_value = 1.0

        if acc is None:
            # 没有任何 part：全 0
            combine.inputs["R"].default_value = 0.0
            combine.inputs["G"].default_value = 0.0
            combine.inputs["B"].default_value = 0.0
        else:
            links.new(acc.outputs[0], combine.inputs["R"])
            links.new(acc.outputs[0], combine.inputs["G"])
            links.new(acc.outputs[0], combine.inputs["B"])

        mask_file_output = nodes.new("CompositorNodeOutputFile")
        mask_file_output.base_path = output_folder
        mask_file_output.file_slots[0].use_node_format = True
        mask_file_output.format.file_format = "OPEN_EXR"
        mask_file_output.format.color_depth = "32"
        mask_file_output.format.color_mode = "RGB"  # 三通道

        links.new(combine.outputs["Image"], mask_file_output.inputs[0])
        outputs["mask"] = mask_file_output

    return outputs, spec_nodes


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera() -> None:
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_width = 36
    return cam


def set_global_light(env_light=0.5) -> None:
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes["Background"]
    back_node.inputs["Color"].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs["Strength"].default_value = 1.0


def init_lighting() -> dict:
    # MOD: instead of 3 explicit lights in render.py, match blender_render_img_mask (world background light)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    set_global_light(env_light=0.5)
    return {}


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    else:
        import_function(filepath=object_path)


# NEW (major change): import multiple glbs and assign pass_index per "part" (glb)
def load_objects_and_assign_pass_index(object_paths: List[str]) -> None:
    for part_idx, p in enumerate(object_paths):
        before = set(bpy.data.objects)
        load_object(p)
        after = set(bpy.data.objects)
        new_objs = list(after - before)

        new_mesh_objs = [o for o in (after - before) if o.type == "MESH"]
        if not new_mesh_objs:
            continue

        bpy.ops.object.select_all(action="DESELECT")
        for o in new_mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new_mesh_objs[0]
        bpy.ops.object.join()

        joined = bpy.context.view_layer.objects.active
        joined.name = f"part_{part_idx:04d}"
        joined.data.name = f"part_{part_idx:04d}_mesh"


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def split_mesh_normal() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def delete_custom_normals() -> None:
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()


def override_material() -> None:
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bpy.context.scene.view_layers["ViewLayer"].material_override = new_mat


def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)


def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")


def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def scene_bbox(objects=None, ignore_small_obj=False, ignore_matrix=False) -> Any:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False

    for obj in objects:
        # print(max(obj.dimensions*100))
        if max(obj.dimensions * 100) < 0.1 and ignore_small_obj:
            print("ignore_small_obj", obj.name, max(obj.dimensions * 100))
            continue
        found = True
        for coord in obj.bound_box:
            # print(coord[0], coord[1], coord[2])
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord

            bbox_min = Vector((min(bbox_min[i], coord[i]) for i in range(3)))
            bbox_max = Vector((max(bbox_max[i], coord[i]) for i in range(3)))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects() -> Any:
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def normalize_scene(normalization_range, objects) -> Any:
    bpy.ops.object.empty_add(type="PLAIN_AXES")
    root_object = bpy.context.object
    for obj in scene_root_objects():
        if obj != root_object:
            _matrix_world = obj.matrix_world.copy()
            obj.parent = root_object
            obj.matrix_world = _matrix_world
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objects)
    scale = normalization_range / max(bbox_max - bbox_min)
    root_object.scale *= scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(objects, True)
    mesh_offset = -(bbox_min + bbox_max) / 2
    root_object.matrix_local.translation = mesh_offset
    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action="DESELECT")
    return root_object, bbox_max - bbox_min, scale, mesh_offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


def main(arg) -> None:
    os.makedirs(arg.output_folder, exist_ok=True)

    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)

    init_scene()

    # MOD: load list of glbs, assign part_id via pass_index
    object_paths = json.loads(arg.objects)
    load_objects_and_assign_pass_index(object_paths)
    if arg.split_normal:
        split_mesh_normal()

    outputs, spec_nodes = init_nodes(
        output_folder=arg.output_folder,
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist,
        save_mask=arg.save_mask,  # MOD
    )
    for name, output in outputs.items():
        output.file_slots[0].path = f"tmp_{name}_"

    print("[INFO] Scene initialized.")

    # normalize scene
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.visible_get() is True and obj.hide_get() is False]
    normalization_range = 1.0
    root_object, bbox_size, scale, mesh_offset = normalize_scene(normalization_range, mesh_objects)

    bpy.context.view_layer.update()
    if arg.force_rotation_deg != 0.0:
        root_object.rotation_euler[2] = math.radians(arg.force_rotation_deg)

    # camera related
    default_camera_lens = 50.0
    default_camera_sensor_width = 36.0
    distance = (default_camera_lens / default_camera_sensor_width) * math.sqrt(bbox_size.x**2 + bbox_size.y**2 + bbox_size.z**2)
    print("[INFO] Scene normalized.")

    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print("[INFO] Camera and lighting initialized.")

    # Override material
    if arg.geo_mode:
        override_material()

    views = json.loads(arg.views)
    for i, view in enumerate(views):

        view["radius"] = float(distance)

        cam.location = (
            view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["pitch"]),
        )

        cam.data.sensor_width = default_camera_sensor_width
        cam.data.lens = (default_camera_sensor_width / 2.0) / math.tan(view["fov"] / 2.0)

        cam.rotation_euler = (Vector((0.0, 0.0, 0.0)) - Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()

        if arg.save_depth:
            spec_nodes["depth_map"].inputs[1].default_value = view["radius"] - 0.5 * np.sqrt(3)
            spec_nodes["depth_map"].inputs[2].default_value = view["radius"] + 0.5 * np.sqrt(3)

        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f"color_{i:04d}.png")

        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = list(Path(arg.output_folder).glob(f"tmp_{name}_*.{ext}"))[0]
            os.rename(path, os.path.join(arg.output_folder, f"{name}_{i:04d}.{ext}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renders given glb parts by rotating a camera around them.")
    parser.add_argument("--views", type=str, required=True, help="JSON string of views. list of {yaw, pitch, radius, fov}.")
    # MOD: list of glbs instead of single object
    parser.add_argument("--objects", type=str, required=True, help="JSON string list of GLB paths. Each GLB is one part id (pass_index = order+1).")
    parser.add_argument("--output_folder", type=str, default="/tmp", help="The path the output will be dumped to.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the images.")
    parser.add_argument("--engine", type=str, default="CYCLES", help="Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...")
    parser.add_argument("--geo_mode", action="store_true", help="Geometry mode for rendering.")
    parser.add_argument("--save_depth", action="store_true", help="Save the depth maps.")
    parser.add_argument("--save_normal", action="store_true", help="Save the normal maps.")
    parser.add_argument("--save_albedo", action="store_true", help="Save the albedo maps.")
    parser.add_argument("--save_mist", action="store_true", help="Save the mist distance maps.")
    parser.add_argument("--save_mask", action="store_true", help="Save the segmentation mask as IndexOB EXR.")
    parser.add_argument("--split_normal", action="store_true", help="Split the normals of the mesh.")
    parser.add_argument("--force_rotation_deg", type=float, default=0.0, help="Rotate normalized root object around +Z (degrees). Same role as old FORCE_ROTATION.")

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    main(args)
