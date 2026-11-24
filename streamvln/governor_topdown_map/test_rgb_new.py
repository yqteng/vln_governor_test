"""Compare two RGB renderings of one official HSGM + YOLOE geometry replay.

rgb_fused.png: RGB follows the author's point-cloud voxel means / history merges.
rgb_raw.png: overlay original measured RGB samples before any voxel averaging.
Both use identical official geometry, depth limits, semantic masks and contours.
The initial observation uses front/left/right/back views; every subsequent
recorded action fuses only the current first-person RGB-D observation.
Raw colors use top-surface selection (latest breaks exact height ties), never a
color mean, blur, nearest-neighbor hole fill or invented texture. A small tiled
display cache keeps original XYZ/RGB samples instead of retaining every frame.
Only the two latest global images per trajectory are saved by default; additional
floors have _floor_02 etc. Optional centered views are explicitly requested.

Requires the author's HSGM_public checkout and YOLOE-26L checkpoint.
Loads geometry, instance filtering/association and rendering from that checkout.
No Governor mapping, room partition, red-wall or opening/node algorithms are used.
A transport worker isolates YOLOE dependencies from the Habitat Python environment.
Compatibility-only fix: reconcile Open3D positions/colors before merging point clouds.
Recorded actions replace the paper's navigation policy; this is a mapping replay,
not a reproduction of the paper's end-to-end navigation metrics.
"""
import argparse
import base64
import contextlib
import io
import queue
import threading
import types
import ast
import copy
import gc
import hashlib
import heapq
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import numpy as np
from PIL import Image
ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / 'config/vln_r2r.yaml').is_file() and (p / 'streamvln').is_dir()), Path.cwd())
ACTION_KEYS = {0:'stop', 1:'move_forward', 2:'turn_left', 3:'turn_right', 4:'governor_move_backward'}
_BACKWARD_REGISTERED = False

def read_replay(args):
    path = Path(args.actions_file).expanduser().resolve()
    rows = []
    previous_step = None
    with path.open(encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError('%s line %d: invalid JSON' % (path, line_number)) from exc
            action = row.get('action') if isinstance(row, dict) else None
            if type(action) is not int or action not in ACTION_KEYS:
                raise ValueError('Line %d: action must be integer 0..4' % line_number)
            step = row.get('step_id', len(rows))
            if type(step) is not int or previous_step is not None and step <= previous_step:
                raise ValueError('Line %d: step_id must increase strictly' % line_number)
            previous_step = step
            rows.append(dict(row, step_id=step))
    if not rows:
        raise ValueError('Action file is empty')
    inferred = path.parent.name.rsplit('_', 1)
    scene = args.scene_id or (inferred[0] if len(inferred) == 2 else None)
    episode = args.episode_id if args.episode_id is not None else (inferred[1] if len(inferred) == 2 else None)
    if not scene or episode is None:
        raise ValueError('Set --scene_id and --episode_id, or use records/SCENE_EPISODE/actions.jsonl')
    return path, rows, str(scene), str(episode)


def scene_matches(scene_path, wanted):
    scene_path = str(scene_path).replace('\\', '/')
    return scene_path == wanted or wanted in scene_path.split('/') or Path(scene_path).stem == wanted


def add_backward(config, habitat):
    global _BACKWARD_REGISTERED
    import habitat_sim
    from habitat.config.default_structured_configs import ActionConfig
    from habitat.core.embodied_task import SimulatorTaskAction
    from habitat.core.registry import registry

    if not _BACKWARD_REGISTERED:
        @registry.register_task_action
        class MapReplayBackwardAction(SimulatorTaskAction):
            def step(self, *args, **kwargs):
                self._sim.get_agent(0).agent_config.action_space['governor_move_backward'] = habitat_sim.ActionSpec(
                    'move_backward', habitat_sim.ActuationSpec(amount=.25))
                return self._sim.step('governor_move_backward')
        _BACKWARD_REGISTERED = True

    with habitat.config.read_write(config):
        config.habitat.task.actions['governor_move_backward'] = ActionConfig(type='MapReplayBackwardAction')


def replay_environment(args, rows, scene, episode_id):
    # Only Habitat and its task/dataset registrations: never import streamvln_eval/model.
    sys.path[:0] = [str(ROOT / 'streamvln'), str(ROOT)]
    if importlib.util.find_spec('habitat') is None or importlib.util.find_spec('habitat_baselines') is None:
        source = Path(args.habitat_root)
        sys.path[:0] = [str(source / 'habitat-lab'), str(source / 'habitat-baselines')]
    import habitat
    from habitat_baselines.config.default import get_config
    # The project's custom VLN task/dataset/measurement registration package.
    importlib.import_module('habitat_extensions')
    importlib.import_module('habitat_extensions.measures')
    config = get_config(args.habitat_config_path)
    with habitat.config.read_write(config):
        cfg = config.habitat
        cfg.seed = args.seed
        cfg.dataset.split = args.eval_split
        if args.data_path:
            cfg.dataset.data_path = args.data_path
        if args.scenes_dir:
            cfg.dataset.scenes_dir = args.scenes_dir
        # Dataset filtering precedes Env construction, so no unrelated scene is rendered.
        cfg.dataset.content_scenes = ['*']
        cfg.task.measurements.clear()
        count = min(len(rows), args.max_steps) if args.max_steps else len(rows)
        cfg.environment.max_episode_steps = count + 1
        cfg.environment.max_episode_seconds = 0
        cfg.simulator.turn_angle = 15
        cfg.simulator.forward_step_size = .25
        cfg.simulator.habitat_sim_v0.gpu_device_id = 0
        cfg.simulator.habitat_sim_v0.enable_physics = False
        agent_key = cfg.simulator.agents_order[cfg.simulator.default_agent_id]
        agent = cfg.simulator.agents[agent_key]
        sensors = agent.sim_sensors
        if args.sensor_profile == 'official':
            import yaml
            official = yaml.safe_load((Path(args.hsgm_repo)/'config/vlnce_test.yaml').read_text())['camera']
            args.sensor_width = 1920//int(official['res_factor'])
            args.sensor_height = 1080//int(official['res_factor'])
            for name in ('rgb_sensor','depth_sensor'):
                sensor = sensors[name]
                # Habitat's structured sensor config declares hfov as int.
                # OmegaConf rejects even an integral float such as 135.0.
                hfov = float(official['fov'])
                if not hfov.is_integer():
                    raise ValueError('This Habitat sensor schema requires integer HFOV; official camera.fov='+str(hfov))
                sensor.hfov = int(hfov)
                sensor.position = [0.,float(official['height']),0.]
                sensor.orientation = [math.radians(float(official['pitch'])),0.,0.]
            sensors.depth_sensor.normalize_depth = False
            sensors.depth_sensor.min_depth = float(official.get('min_depth', .5))
            sensors.depth_sensor.max_depth = float(official.get('max_depth', 50.))
        if 'rgb_sensor' not in sensors or 'depth_sensor' not in sensors:
            raise ValueError('Habitat config must contain aligned rgb_sensor and depth_sensor')
        if args.sensor_width:
            for name in ('rgb_sensor', 'depth_sensor'):
                sensors[name].width = args.sensor_width
                sensors[name].height = args.sensor_height
        if abs(sensors.rgb_sensor.hfov - sensors.depth_sensor.hfov) > 1e-4:
            raise ValueError('RGB/depth HFOV must agree')
        if not np.allclose(sensors.rgb_sensor.position, sensors.depth_sensor.position) or not np.allclose(
                sensors.rgb_sensor.orientation, sensors.depth_sensor.orientation):
            raise ValueError('RGB/depth extrinsics must agree')
    if any(r['action'] == 4 for r in rows):
        add_backward(config, habitat)
    dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    matches = [ep for ep in dataset.episodes if str(ep.episode_id) == episode_id
               and scene_matches(ep.scene_id, scene)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one scene=%s episode=%s in split=%s, found %d. '
                         'Check --eval_split / --data_path.' % (scene, episode_id, args.eval_split, len(matches)))
    dataset.episodes = matches
    env = habitat.Env(config=config, dataset=dataset)
    env.seed(args.seed)
    env.current_episode = matches[0]
    return env, sensors, matches[0]


def ensure_unused_name(output, name):
    folder = output / name
    if folder.exists() and any(folder.glob('rgb_*.png')):
        raise ValueError('Maps already exist for %s; select a new --output_path' % name)


def batch_jobs(args):
    import re
    root = Path(args.records_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Records root does not exist: ' + str(root))
    # Natural ordering: numeric episode suffixes sort as 2, 10, 11 rather than 10, 11, 2.
    def key(path):
        relative = path.parent.relative_to(root).as_posix()
        return (tuple(int(s) if s.isdecimal() else s.lower()
                      for s in re.split(r'(\d+)', relative)), relative)
    paths = sorted((p.resolve() for p in root.rglob('actions.jsonl') if p.is_file()), key=key)
    if not paths:
        raise ValueError('No actions.jsonl files found under ' + str(root))
    if args.start_folder:
        supplied = Path(args.start_folder).expanduser()
        if supplied.is_absolute():
            wanted = supplied.resolve()
            found = [i for i, path in enumerate(paths) if path.parent == wanted]
        else:
            wanted = str(args.start_folder).replace('\\', '/').strip('/')
            found = [i for i, path in enumerate(paths)
                     if path.parent.relative_to(root).as_posix() == wanted]
            if not found:
                found = [i for i, path in enumerate(paths) if path.parent.name == wanted]
        if len(found) != 1:
            raise ValueError('--start_folder must match exactly one folder containing actions.jsonl; '
                             'matched %d: %s' % (len(found), args.start_folder))
        paths = paths[found[0]:]
    if args.max_episodes:
        paths = paths[:args.max_episodes]
    output = Path(args.output_path).expanduser().resolve() if args.output_path else ROOT / 'results/hsgm_geometry'
    jobs = []
    names = set()
    for path in paths:
        relative = path.parent.relative_to(root)
        name = path.parent.name
        if name in names:
            raise ValueError('Duplicate trajectory folder name cannot be flattened: ' + name)
        names.add(name)
        ensure_unused_name(output, name)
        jobs.append(dict(folder=relative.as_posix(), actions_file=str(path),
                         output_path=str(output), status='pending'))
    return root, output, jobs


GEOMETRY_METHODS = {
    '__init__', '_switch_to_floor', '_calculate_floor_height_difference',
    '_detect_and_update_floor', 'reset', '_identify_and_process_stairs',
    'is_on_platform', '_safe_clear_tpcd', 'release_vram', 'update',
    'update_multiview', 'release_all_vram', 'get_object_entities', 'associate_object_entities',
}


class RawRGBAtlas:
    """Display-only XY tiles holding actual source pixels, not mean colors.

    Half-output-pixel cells bound memory; each retains its original highest XYZ
    sample and uint8 RGB. This is sample selection, not voxel mean fusion. The
    official mapper never reads this cache. Negative coordinates use floor div.
    """
    TILE = 128

    def __init__(self, pixel_size):
        self.cell_size = float(pixel_size) / 2.
        self.layers = {}
        self.serial = 0

    def add(self, level, floor, ceiling, points, colors):
        points, colors = np.asarray(points), np.asarray(colors)
        if not len(points):
            return
        self.serial += 1
        finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        layer = self.layers.setdefault(level, {})
        for role, height_mask in (
                ('floor', (points[:, 2] > floor-1.) & (points[:, 2] < floor+.2)),
                ('obstacle', (points[:, 2] > floor+.2) & (points[:, 2] < ceiling))):
            valid = finite & height_mask
            xyz = np.asarray(points[valid], dtype=np.float32)
            rgb = np.asarray(colors[valid], dtype=np.uint8)
            if not len(xyz):
                continue
            cells = np.floor(xyz[:, :2] / self.cell_size).astype(np.int64)
            tile_xy = np.floor_divide(cells, self.TILE)
            local = np.remainder(cells, self.TILE)
            # Height precedes original pixel index: last/highest is displayed.
            slot = local[:, 1]*self.TILE + local[:, 0]
            order = np.lexsort((np.arange(len(xyz)), xyz[:, 2], slot,
                                tile_xy[:, 1], tile_xy[:, 0]))
            tile_xy, slot = tile_xy[order], slot[order]
            xyz, rgb = xyz[order], rgb[order]
            last = np.r_[(np.diff(tile_xy, axis=0) != 0).any(axis=1)
                         | (np.diff(slot) != 0), True]
            tile_xy, slot, xyz, rgb = tile_xy[last], slot[last], xyz[last], rgb[last]
            splits = np.r_[0, 1+np.flatnonzero((np.diff(tile_xy, axis=0) != 0).any(axis=1)), len(slot)]
            tiles = layer.setdefault(role, {})
            for first, end in zip(splits[:-1], splits[1:]):
                key = tuple(tile_xy[first].tolist())
                if key not in tiles:
                    size = self.TILE*self.TILE
                    tiles[key] = (np.full((size, 3), np.nan, np.float32),
                                  np.zeros((size, 3), np.uint8),
                                  np.zeros(size, np.int64))
                stored_xyz, stored_rgb, stamps = tiles[key]
                indices, new_xyz, new_rgb = slot[first:end], xyz[first:end], rgb[first:end]
                previous_z = stored_xyz[indices, 2]
                replace = ~np.isfinite(previous_z) | (new_xyz[:, 2] >= previous_z)
                indices = indices[replace]
                stored_xyz[indices], stored_rgb[indices] = new_xyz[replace], new_rgb[replace]
                stamps[indices] = self.serial

    def samples(self, level, role):
        points, colors, times = [], [], []
        roles = ('floor', 'obstacle') if role == 'surface' else (role,)
        for source_role in roles:
            for xyz, rgb, stamps in self.layers.get(level, {}).get(source_role, {}).values():
                valid = np.isfinite(xyz[:, 2])
                points.append(xyz[valid])
                colors.append(rgb[valid])
                times.append(stamps[valid])
        if not points:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), np.empty(0, np.int64)
        return np.concatenate(points), np.concatenate(colors), np.concatenate(times)


def capture_raw_rgb(mapper, world_points, camera_colors):
    """Observer inserted AFTER world projection and BEFORE the first voxel mean."""
    atlas = getattr(mapper, '_raw_rgb_atlas', None)
    if atlas is not None:
        atlas.add(mapper.floor_level, mapper.floor_height, mapper.ceiling_height,
                  world_points, camera_colors)


def paint_original_rgb(image, mask, mapper, transform, role):
    """Same masks/projection/z-buffer for both methods; only RGB source differs.

    No smoothing, neighbor color fill, or averaging is done here. Unsupported
    texture remains gray, including geometry supplied by ground interpolation.
    A sparse pixel-color projection is cached once per role/render, not per
    semantic object, to keep rendering costs independent of object count.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    cache = mapper._rgb_render_cache
    if role not in cache:
        if mapper._rgb_method == 'raw':
            points, colors, stamps = mapper._raw_rgb_atlas.samples(mapper._rgb_level, role)
        else:
            scene = getattr(mapper, 'scene_pcd', None)
            if scene is None or scene.is_empty() or 'colors' not in scene.point:
                points, colors = np.empty((0, 3)), np.empty((0, 3), np.uint8)
            else:
                points = scene.point.positions.cpu().numpy()
                rgb = scene.point.colors.cpu().numpy()
                valid = np.isfinite(points).all(axis=1) & np.isfinite(rgb).all(axis=1)
                points = points[valid]
                colors = np.rint(np.clip(rgb[valid], 0, 1)*255).astype(np.uint8)
            stamps = np.zeros(len(points), np.int64)
        lower, upper = mapper.floor_height, mapper.ceiling_height
        selected = np.isfinite(points).all(axis=1)
        if role == 'floor':
            selected &= (points[:, 2] > lower-1.) & (points[:, 2] < lower+.2)
        elif role == 'obstacle':
            selected &= (points[:, 2] > lower+.2) & (points[:, 2] < upper)
        else:
            # Semantic instances can be on the floor (e.g. carpets). They are
            # not all elevated obstacles; keep both kinds of measured surface.
            selected &= (points[:, 2] > lower-1.) & (points[:, 2] < upper)
        points, colors, stamps = points[selected], colors[selected], stamps[selected]
        h, w = mask.shape
        if len(points):
            pixels = np.asarray(transform(points), dtype=np.int64)
            valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            points, colors, stamps, pixels = points[valid], colors[valid], stamps[valid], pixels[valid]
            flat = pixels[:, 1]*w + pixels[:, 0]
            order = np.lexsort((stamps, points[:, 2], flat))
            flat = flat[order]
            last = np.r_[np.diff(flat) != 0, True] if len(flat) else np.empty(0, bool)
            keep = order[last]
            cache[role] = (pixels[keep], colors[keep, ::-1].copy())  # author's BGR canvas
        else:
            cache[role] = (np.empty((0, 2), np.int64), np.empty((0, 3), np.uint8))
    pixels, bgr = cache[role]
    if len(pixels):
        selected = mask[pixels[:, 1], pixels[:, 0]]
        image[pixels[selected, 1], pixels[selected, 0]] = bgr[selected]


def load_official_mapper(repo, depth_max, detector):
    """Use actual upstream algorithms, without importing segmentation/planners."""
    import open3d as o3d
    import quaternion
    import cv2
    import torch
    import matplotlib.pyplot as plt
    src = Path(repo).expanduser().resolve() / 'src'
    paths = [src/'mapping_utils/geometry.py', src/'mapping_utils/transform.py', src/'mapper.py']
    if any(not p.is_file() for p in paths):
        raise ValueError('Missing HSGM source. Clone HSGM_public and set --hsgm_repo.')
    namespace = dict(np=np, o3d=o3d, quaternion=quaternion, math=math, time=time,
                     gc=gc, heapq=heapq, Any=Any, torch=torch, cv2=cv2, plt=plt,
                     _capture_raw_rgb=capture_raw_rgb)
    selection = {
        'geometry.py': {'get_pointcloud_from_depth', 'translate_to_world',
                        'gpu_pointcloud_from_array', 'gpu_merge_pointcloud',
                        'get_pointcloud_from_depth_mask', 'gpu_cluster_filter', 'pointcloud_distance'},
        'transform.py': {'habitat_translation', 'habitat_rotation'},
    }
    fingerprints = {}
    for path in paths:
        text = path.read_text(encoding='utf-8-sig')
        fingerprints[str(path.relative_to(src))] = hashlib.sha256(text.encode()).hexdigest()
        tree = ast.parse(text, filename=str(path))
        if path.name == 'mapper.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Instruct_Mapper')
            methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in GEOMETRY_METHODS]
            missing = GEOMETRY_METHODS - {n.name for n in methods}
            if missing:
                raise ValueError('Upstream geometry interface changed: missing '+str(sorted(missing)))
            class EmptyVoxelGuard(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    if isinstance(node.func,ast.Attribute) and node.func.attr=='voxel_down_sample':
                        return ast.copy_location(ast.Call(func=ast.Name(id='_hsgm_safe_voxel',ctx=ast.Load()),
                            args=[node.func.value]+node.args,keywords=node.keywords),node)
                    return node
            class CaptureRawRGB(ast.NodeTransformer):
                count = 0

                def visit_Assign(self, node):
                    if (any(isinstance(t, ast.Name) and t.id == 'world_points' for t in node.targets)
                            and isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == 'translate_to_world'):
                        self.count += 1
                        callback = ast.Expr(value=ast.Call(
                            func=ast.Name(id='_capture_raw_rgb', ctx=ast.Load()),
                            args=[ast.Name(id=n, ctx=ast.Load())
                                  for n in ('self', 'world_points', 'camera_colors')], keywords=[]))
                        return [node, ast.copy_location(callback, node)]
                    return node

            capture = CaptureRawRGB()
            for method in methods:
                if method.name == 'update_multiview':
                    capture.visit(method)
            if capture.count != 1:
                raise ValueError('Expected one pre-voxel RGB capture site in official update_multiview')
            cls.body = [EmptyVoxelGuard().visit(method) for method in methods]
            body = [cls]
        else:
            body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selection[path.name]]
            if len(body) != len(selection[path.name]):
                raise ValueError('Upstream point projection interface changed: '+str(path))
        module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(module, str(path), 'exec'), namespace)
    # Depth is converted from Habitat normalized values before this function.
    # 0 removes the author's 4.9m software cap; invalid sensor values stay invalid.
    def preprocess(depth):
        depth = np.array(depth, dtype=np.float32, copy=True)
        good = np.isfinite(depth) & (depth >= .1)
        if depth_max > 0:
            good &= depth <= depth_max
        depth[~good] = 0
        return depth
    namespace['preprocess_depth'] = preprocess
    namespace['preprocess_image'] = lambda image: image
    namespace['instance_segmentation'] = detector.predict
    # Retain original projection/clustering/association. Normalize only tensor
    # storage fields to avoid Open3D _Map_base::at on normal-bearing stair clouds.
    original_merge = namespace['gpu_merge_pointcloud']
    def clean(cloud):
        result = o3d.t.geometry.PointCloud(cloud.device)
        if cloud.is_empty():
            # Open3D voxel_down_sample still looks up positions on empty
            # clouds; a bare TensorMap triggers _Map_base::at.
            result.point.positions = o3d.core.Tensor.zeros(
                (0,3), o3d.core.Dtype.Float32, cloud.device)
            result.point.colors = o3d.core.Tensor.zeros(
                (0,3), o3d.core.Dtype.Float32, cloud.device)
            return result
        result.point.positions = cloud.point.positions.clone()
        if 'colors' in cloud.point:
            result.point.colors = cloud.point.colors.clone()
        else:
            result.point.colors = o3d.core.Tensor.full(
                cloud.point.positions.shape, 100./255., o3d.core.Dtype.Float32, cloud.device)
        return result
    namespace['gpu_merge_pointcloud'] = lambda a,b: original_merge(clean(a), clean(b))
    namespace['_hsgm_safe_voxel'] = lambda cloud,*a,**k: clean(cloud) if cloud.is_empty() else cloud.voxel_down_sample(*a,**k)
    segment_file = src/'segmentation/instance_segmentation.py'
    segment_tree = ast.parse(segment_file.read_text(encoding='utf-8-sig'))
    color_fn = next(n for n in segment_tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_class_color')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[color_fn],type_ignores=[])),str(segment_file),'exec'),namespace)
    agent_file = src/'agent/agent.py'
    agent_tree = ast.parse(agent_file.read_text(encoding='utf-8-sig'))
    functions = [n for n in agent_tree.body if isinstance(n,ast.FunctionDef) and n.name in
                 {'draw_direction_markers','create_top_down_map_centered','create_top_down_map_global'}]
    if len(functions) != 3:
        raise ValueError('Expected official centered/global map renderers in agent/agent.py')
    namespace['_paint_original_rgb'] = paint_original_rgb
    for fn in functions:
        if fn.name in ('create_top_down_map_centered','create_top_down_map_global'):
            transform = 'transform_and_to_pixel' if fn.name.endswith('centered') else 'world_to_pixel'
            class RGBPaint(ast.NodeTransformer):
                def visit_Assign(self,node):
                    if len(node.targets) != 1:
                        return node
                    target = node.targets[0]
                    if not (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
                            and target.value.id=='top_down_map'):
                        return node
                    value = node.value
                    role = None
                    if isinstance(value,ast.Tuple) and all(isinstance(n,ast.Constant) for n in value.elts):
                        rgb = tuple(n.value for n in value.elts)
                        role = 'floor' if rgb==(200,200,200) else 'obstacle' if rgb==(50,50,50) else None
                    elif isinstance(value,ast.Name) and value.id in ('object_color','obj_color'):
                        role = 'surface'
                    if role is None:
                        return node
                    call = ast.Expr(value=ast.Call(func=ast.Name(id='_paint_original_rgb',ctx=ast.Load()),
                        args=[ast.Name(id='top_down_map',ctx=ast.Load()),target.slice,
                              ast.Name(id='mapper',ctx=ast.Load()),ast.Name(id=transform,ctx=ast.Load()),
                              ast.Constant(value=role)],keywords=[]))
                    # A semantic instance must not erase already-painted floor
                    # RGB with a flat class/dark color. Overlay measured RGB
                    # where available; otherwise retain the underlying map.
                    if role == 'surface':
                        return ast.copy_location(call,node)
                    return [node,ast.copy_location(call,node)]
            RGBPaint().visit(fn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(agent_file),'exec'),namespace)
    for file in (segment_file,agent_file,src/'segmentation/object_list.py'):
        fingerprints[str(file.relative_to(src))] = hashlib.sha256(file.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(['git', '-C', str(src.parent), 'rev-parse', 'HEAD'],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return namespace['Instruct_Mapper'], namespace, dict(repository='https://github.com/Teacher-Tom/HSGM_public',
                                              commit=commit, source_sha256=fingerprints)


def metric_depth(raw, cfg):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError('Depth must be HxW or HxWx1')
    depth = raw * (cfg.max_depth-cfg.min_depth) + cfg.min_depth if cfg.normalize_depth else raw.copy()
    # Saturated readings are not measured far surfaces. No invented far points.
    valid = np.isfinite(depth) & (depth > max(.001, cfg.min_depth)) & (depth < cfg.max_depth-1e-4)
    return np.where(valid, depth, 0).astype(np.float32)


def cloud_array(cloud):
    return np.empty((0, 3), np.float32) if cloud.is_empty() else cloud.point.positions.cpu().numpy()


class OfficialReplay:
    def __init__(self, env, sensors, args, mapper_class, renderers, output, name):
        self.env, self.sensors, self.args = env, sensors, args
        self.output, self.name, self.mapper_class = output, name, mapper_class
        self.renderers = renderers
        self.cfg = sensors.depth_sensor
        self.depth_key = getattr(self.cfg, 'uuid', 'depth')
        self.rgb_key = getattr(sensors.rgb_sensor, 'uuid', 'rgb')
        self.mapper = None
        self.layer_poses = {}
        self.layer_heights = {}
        # Display cache only; never used by original HSGM geometry/segmentation.
        self.raw_atlas = RawRGBAtlas(args.bev_pixel_size)
        self.step = 0

    def view(self, obs, state):
        depth = metric_depth(obs[self.depth_key], self.cfg)
        rgb = np.asarray(obs[self.rgb_key])[..., :3]
        if rgb.shape[:2] != depth.shape:
            raise ValueError('Aligned RGB and depth must have the same resolution')
        sensor = state.sensor_states[self.depth_key]
        h, w = depth.shape
        f = w/(2*math.tan(math.radians(self.cfg.hfov)/2))
        # Match the camera matrix supplied by the author's agent entry point.
        K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], np.float32)
        return dict(rgb=rgb, depth=depth, position=np.asarray(sensor.position),
                    rotation=sensor.rotation, camera_intrinsic=K)

    def fuse(self, obs):
        import quaternion
        sim = self.env.sim
        state = copy.deepcopy(sim.get_agent_state())
        primary = self.view(obs, state)
        initializing = self.mapper is None
        if self.mapper is None:
            offset = float(state.position[1]-primary['position'][1])
            self.mapper = self.mapper_class(primary['camera_intrinsic'],
                pcd_resolution=self.args.pcd_resolution, grid_resolution=self.args.grid_resolution,
                floor_height=offset, ceiling_height=offset+self.args.height_ceiling,
                resolution=primary['depth'].shape,
                device='cpu:0' if self.args.map_device == 'cpu' else self.args.map_device)
            self.mapper.reset(primary['position'], primary['rotation'])
            self.mapper._raw_rgb_atlas = self.raw_atlas
            instruction = getattr(self.env.current_episode, 'instruction', '')
            text = str(getattr(instruction, 'instruction_text', instruction)).lower()
            self.mapper.enable_stair_detection = any(word in text for word in ('stair','step'))
        # Four views only at episode reset. Later actions use precisely the
        # actual forward observation, including recorded turn actions.
        angles = [90., -90., 180.] if initializing else []
        views = [primary]
        previous = getattr(sim, '_prev_sim_obs', None)
        try:
            for angle in angles:
                rotation = state.rotation * quaternion.from_rotation_vector([0., math.radians(angle), 0.])
                extra = sim.get_observations_at(state.position, rotation, keep_agent_at_new_pose=True)
                views.append(self.view(extra, sim.get_agent_state()))
        finally:
            sim.get_agent(0).set_state(state, reset_sensors=False, infer_sensor_states=False)
            if hasattr(sim, '_prev_sim_obs'):
                sim._prev_sim_obs = previous
        # Official synchronized multiview update; primary is the recorded forward camera.
        self.mapper.update_multiview(views[1:]+views[:1], primary_index=len(views)-1, do_instance_segmentation=True)
        self.layer_heights[self.mapper.floor_level] = (self.mapper.floor_height,self.mapper.ceiling_height)
        self.layer_poses[self.mapper.floor_level] = (self.mapper.current_position.copy(), self.mapper.current_rotation.copy())

    def publish(self):
        import cv2
        mapper = self.mapper
        layers = dict(mapper.pcds_per_floor)
        layers[mapper.floor_level] = dict(scene_pcd=mapper.scene_pcd,navigable_pcd=mapper.navigable_pcd,
            obstacle_pcd=mapper.obstacle_pcd, object_entities=mapper.object_entities,
            trajectory_position=mapper.trajectory_position)
        folder = self.output / self.name
        folder.mkdir(parents=True, exist_ok=True)
        for index, (level, layer) in enumerate(sorted(layers.items()), 1):
            view = types.SimpleNamespace(**layer)
            view.floor_height, view.ceiling_height = self.layer_heights.get(level,(mapper.floor_height,mapper.ceiling_height))
            view.current_position, view.current_rotation = self.layer_poses.get(level,
                (mapper.current_position,mapper.current_rotation))
            modes = ('global','centered') if self.args.map_view == 'both' else (self.args.map_view,)
            for mode in modes:
                for method in ('fused', 'raw'):
                    view._rgb_method = method
                    view._rgb_level = level
                    view._raw_rgb_atlas = self.raw_atlas
                    view._rgb_render_cache = {}
                    if mode == 'centered':
                        image = self.renderers['create_top_down_map_centered'](view, self.cfg.hfov)
                    else:
                        image = self.renderers['create_top_down_map_global'](view,
                            map_scale=self.args.bev_pixel_size, fov_angle_deg=self.cfg.hfov,
                            enable_semantic=True, enable_navigation=False)
                    # Upstream global converts BGR->RGB before return; centered
                    # returns BGR. Avoid a second swap on the global RGB image.
                    if mode == 'centered':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    suffix = '_floor_%02d' % index if index > 1 else ''
                    if mode == 'centered':
                        suffix += '_centered'
                    target = folder / ('rgb_%s%s.png' % (method, suffix))
                    temporary = target.with_suffix('.tmp')
                    image.save(temporary, format='PNG', compress_level=1)
                    temporary.replace(target)
                    view._rgb_render_cache.clear()

    def run(self, rows):
        self.fuse(self.env.reset())
        self.publish()
        for self.step, row in enumerate(rows[:self.args.max_steps] if self.args.max_steps else rows, 1):
            if self.env.episode_over:
                break
            self.fuse(self.env.step({'action': ACTION_KEYS[row['action']]}))
            if self.step % self.args.render_every == 0 or row['action'] == 0:
                self.publish()
                print('[hsgm-official] %s step=%d floor=%s scene=%d nav=%d obstacles=%d' % (
                    self.name, self.step, self.mapper.floor_level, len(cloud_array(self.mapper.scene_pcd)),
                    len(cloud_array(self.mapper.navigable_pcd)), len(cloud_array(self.mapper.obstacle_pcd))), flush=True)
            if row['action'] == 0:
                break
        self.publish()


class YOLOEClient:
    """Run the author's detector function in the chosen Torch environment."""
    def __init__(self, args):
        import atexit
        checkpoint = Path(args.yoloe_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError('Missing official YOLOE weights: '+str(checkpoint))
        self.timeout = args.yoloe_timeout
        self.queue = queue.Queue()
        command = [args.yoloe_python, str(Path(__file__).resolve()), '--yoloe_worker',
                   '--hsgm_repo', args.hsgm_repo, '--checkpoint', str(checkpoint),
                   '--device', args.yoloe_device]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        atexit.register(self.close)
        def read():
            for line in self.process.stdout:
                try:
                    self.queue.put(json.loads(line))
                except json.JSONDecodeError:
                    print('[YOLOE] '+line.rstrip(), file=sys.stderr)
            self.queue.put(dict(ok=False,error='YOLOE worker exited'))
        threading.Thread(target=read,daemon=True).start()
        try:
            ready = self.receive()
            print('[hsgm-official] YOLOE ready; official categories=%d, conf=0.6, IoU=0.3' % ready['categories'],flush=True)
        except BaseException:
            self.close()
            raise

    def receive(self):
        try:
            result = self.queue.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError('YOLOE worker timed out; check model/text encoder files') from None
        if not result.get('ok'):
            raise RuntimeError('YOLOE: '+result.get('error','unknown error'))
        return result

    def predict(self, image):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(image,dtype=np.uint8)[...,:3]).save(buffer,format='PNG')
        self.process.stdin.write(json.dumps(dict(image=base64.b64encode(buffer.getvalue()).decode()))+'\n')
        self.process.stdin.flush()
        output = []
        for item in self.receive()['instances']:
            height,width = item.pop('shape')
            packed = base64.b64decode(item.pop('mask_bits'))
            item['mask'] = np.unpackbits(np.frombuffer(packed,np.uint8),count=height*width).reshape(height,width)*255
            output.append(item)
        return output

    def close(self):
        process = getattr(self,'process',None)
        if process is None:
            return
        self.process = None
        try:
            process.stdin.close()
            process.wait(timeout=5)
        except (OSError,subprocess.TimeoutExpired):
            process.kill()
            process.wait()


def yoloe_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloe_worker',action='store_true')
    parser.add_argument('--hsgm_repo',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    channel = sys.stdout
    def reply(value):
        channel.write(json.dumps(value)+'\n');channel.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import cv2
            import matplotlib.pyplot as plt
            from ultralytics import YOLOE
            src = Path(args.hsgm_repo).resolve()/'src/segmentation'
            objects = ast.parse((src/'object_list.py').read_text(encoding='utf-8-sig'))
            category_node = next(n for n in objects.body if isinstance(n,ast.Assign) and
                                 any(isinstance(t,ast.Name) and t.id=='categories' for t in n.targets))
            categories = ast.literal_eval(category_node.value)
            names = list({cat['id']:cat['name'] for cat in categories}.values())
            checkpoint = Path(args.checkpoint).resolve()
            # Ultralytics locates its local TorchScript text encoder in cwd.
            os.chdir(checkpoint.parent)
            model = YOLOE(str(checkpoint))
            model.to(args.device)
            model.set_classes(names, model.get_text_pe(names))
            tree = ast.parse((src/'instance_segmentation.py').read_text(encoding='utf-8-sig'))
            fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='instance_segmentation')
            namespace = dict(np=np,cv2=cv2,plt=plt,model=model)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                         str(src/'instance_segmentation.py'),'exec'),namespace)
        reply(dict(ok=True,categories=len(names)))
    except Exception as exc:
        reply(dict(ok=False,error=type(exc).__name__+': '+str(exc)))
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with Image.open(io.BytesIO(base64.b64decode(request['image']))) as image:
                array = np.asarray(image.convert('RGB')).copy()
            with contextlib.redirect_stdout(sys.stderr):
                instances = namespace['instance_segmentation'](array)
            encoded = []
            for item in instances:
                mask = np.asarray(item['mask'])>0
                encoded.append(dict(class_id=int(item['class_id']),class_name=item['class_name'],
                    shape=list(mask.shape),mask_bits=base64.b64encode(np.packbits(mask).tobytes()).decode()))
            reply(dict(ok=True,instances=encoded))
        except Exception as exc:
            reply(dict(ok=False,error=type(exc).__name__+': '+str(exc)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='HSGM-RGB-COMPARE-v1.2: four initial views, then first-person only; fused/raw RGB')
    parser.add_argument('--hsgm_repo', default=str(ROOT/'third_party/HSGM_public-main'))
    parser.add_argument('--actions_file')
    parser.add_argument('--records_root', default=str(ROOT/'results_base/val_unseen/streamvln_baseline_full/records'))
    parser.add_argument('--start_folder')
    parser.add_argument('--max_episodes', type=int, default=15)
    parser.add_argument('--max_steps', '--max_step', type=int, default=100)
    parser.add_argument('--scene_id')
    parser.add_argument('--episode_id')
    parser.add_argument('--output_path', default='results/test_rgb_new')
    parser.add_argument('--habitat_config_path', default=str(ROOT/'config/vln_r2r.yaml'))
    parser.add_argument('--habitat_root', default='/mnt/disk8t1/zwm/Progress-Aware-VLN/habitat-lab')
    parser.add_argument('--eval_split', default='val_unseen')
    parser.add_argument('--data_path')
    parser.add_argument('--scenes_dir')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sensor_width', type=int, default=0)
    parser.add_argument('--sensor_height', type=int, default=0)
    parser.add_argument('--initial_scan', choices=('four',), default='four',
                        help='initialization uses exactly front/left/right/back observations')
    parser.add_argument('--views', choices=('front',), default='front',
                        help='after initialization, use only the current first-person observation')
    parser.add_argument('--depth_max', type=float, default=4.9, help='4.9: original software cap; 0: unlimited valid sensor depth')
    parser.add_argument('--pcd_resolution', type=float, default=.025)
    parser.add_argument('--grid_resolution', type=float, default=.05)
    parser.add_argument('--height_ceiling', type=float, default=1.8, help='Obstacle inclusion ceiling above floor, not room ceiling estimation')
    parser.add_argument('--map_device', choices=('cpu','cuda:0'), default='cpu')
    parser.add_argument('--bev_pixel_size', type=float, default=.025)
    parser.add_argument('--bev_max_size', type=int, default=4096)
    parser.add_argument('--render_every', type=int, default=5)
    parser.add_argument('--yoloe_python', default='/home/ubuntun/miniconda3/envs/sam2/bin/python')
    parser.add_argument('--yoloe_checkpoint', default=None)
    parser.add_argument('--yoloe_device', default='cuda:0')
    parser.add_argument('--yoloe_timeout', type=float, default=600.)
    parser.add_argument('--sensor_profile', choices=('official','recorded'), default='official')
    parser.add_argument('--map_view', choices=('global','centered','both'), default='global')
    args = parser.parse_args()
    print('[hsgm-official] HSGM-RGB-COMPARE-v1.2: four initial views, then first-person only; fused/raw RGB', flush=True)
    if min(args.max_episodes,args.max_steps,args.sensor_width,args.sensor_height) < 0 or args.render_every < 1:
        parser.error('Counts must be nonnegative; render_every must be positive')
    if bool(args.sensor_width) != bool(args.sensor_height):
        parser.error('Set both sensor width and sensor height')
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.pcd_resolution,args.grid_resolution,args.height_ceiling,args.bev_pixel_size)) or args.bev_max_size < 128:
        parser.error('Resolutions/heights must be finite and positive; bev_max_size >=128')
    if not math.isfinite(args.depth_max) or args.depth_max < 0:
        parser.error('depth_max must be 0 or a positive finite distance')
    if args.actions_file and args.start_folder:
        parser.error('start_folder only applies to batch mode')
    if not args.actions_file and (args.scene_id is not None or args.episode_id is not None):
        parser.error('scene/episode overrides require actions_file')
    os.chdir(ROOT)
    args.hsgm_repo = str(Path(args.hsgm_repo).expanduser().resolve())
    if args.yoloe_checkpoint is None:
        args.yoloe_checkpoint = str(Path(args.hsgm_repo)/'ckpt/yoloe-26l-seg.pt')
    detector = YOLOEClient(args)
    try:
        mapper_class, renderers, provenance = load_official_mapper(args.hsgm_repo, args.depth_max, detector)
    except BaseException:
        detector.close()
        raise
    try:
        output = Path(args.output_path).expanduser().resolve()
        paths = [args.actions_file] if args.actions_file else [j['actions_file'] for j in batch_jobs(args)[2]]
        output.mkdir(parents=True, exist_ok=True)
        (output/'hsgm_run.json').write_text(json.dumps(dict(provenance=provenance, settings=vars(args),
            rgb_methods=dict(fused='Official voxel-mean scene RGB, projected after history fusion',
                             raw='Original RGB before voxel means; top sample per half-pixel XY cache cell',
                             shared='Same geometry/masks/z-buffer; no color smoothing/hole filling'),
            outputs=['TRAJECTORY/rgb_fused.png', 'TRAJECTORY/rgb_raw.png']),
                                                      ensure_ascii=False, indent=2), encoding='utf-8')
        for path in paths:
            args.actions_file = path
            source, rows, scene, episode = read_replay(args)
            name = source.parent.name
            ensure_unused_name(output, name)
            print('[hsgm-official] %s: saving fused/raw RGB in %s' % (name, output/name), flush=True)
            env, sensors, _ = replay_environment(args, rows, scene, episode)
            replay = None
            try:
                replay = OfficialReplay(env, sensors, args, mapper_class, renderers, output, name)
                replay.run(rows)
            finally:
                if replay is not None and replay.mapper is not None:
                    replay.mapper.release_all_vram()
                env.close()
                replay = None  # release raw display tiles before the next scene
    finally:
        detector.close()
    print('[hsgm-official] saved '+str(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(yoloe_worker() if '--yoloe_worker' in sys.argv else main())


