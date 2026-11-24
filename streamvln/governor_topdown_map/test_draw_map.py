"""RGB-D HSGM replay for StreamVLN Habitat action records.

Reference: HSGM Sec. 3.3 and the public implementation's geometric mapper.
https://github.com/Teacher-Tom/HSGM_public

This NumPy adaptation uses world-frame voxel centroids and the published
multiview ground/obstacle split. Overlapping depth views form the initial
panorama. Standing-to-ground segments are rasterized at output resolution.
It uses the replay rig's depth intrinsics and poses. Floor grouping uses stable
agent heights in place of HSGM's stair and landing detector. Each visited floor
has an RGB PNG, with neutral gray for ray-covered ground lacking surface color.
"""

import argparse
import copy
import heapq
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / 'config/vln_r2r.yaml').is_file() and (parent / 'streamvln').is_dir()),
    Path.cwd(),
)
ACTION_KEYS = {0: 'stop', 1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 4: 'governor_move_backward'}
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
        if 'rgb_sensor' not in sensors or 'depth_sensor' not in sensors:
            raise ValueError('Habitat config must contain aligned rgb_sensor and depth_sensor')
        if args.sensor_width:
            sensors.depth_sensor.width = args.sensor_width
            sensors.depth_sensor.height = args.sensor_height
        # Render both sensors on the same pixel grid without resampling depth.
        sensors.rgb_sensor.width = sensors.depth_sensor.width
        sensors.rgb_sensor.height = sensors.depth_sensor.height
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
    import re
    if output.exists():
        pattern = re.compile(re.escape(name) + r'_\d{2,}\.png')
        if any(pattern.fullmatch(p.name) for p in output.iterdir()):
            raise ValueError('Final floor images already exist for %s; select a new --output_path' % name)


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
    output = Path(args.output_path).expanduser().resolve() if args.output_path else ROOT / 'results/hsgm_geometry_rgb'
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


def project_depth(depth, rgb, cfg, position, rotation):
    """Return aligned world points, RGB samples, and upper-edge hits."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f'Expected H×W depth, got {depth.shape}')
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] not in (3, 4):
        raise ValueError(f'Expected RGB or RGBA aligned with depth {depth.shape}, got {rgb.shape}')
    if cfg.normalize_depth:
        depth = depth * (cfg.max_depth - cfg.min_depth) + cfg.min_depth
    valid = (np.isfinite(depth) & (depth >= max(.1, cfg.min_depth))
             & (depth <= 4.9) & (depth < cfg.max_depth - 1e-4))
    h, w = depth.shape
    focal = w / (2 * math.tan(math.radians(cfg.hfov) / 2))

    def project(rows, cols):
        meters = depth[rows, cols]
        camera = np.column_stack(((cols - (w - 1) / 2) * meters / focal,
                                  ((h - 1) / 2 - rows) * meters / focal, -meters))
        return (camera @ rotation.T + position).astype(np.float32)

    rows, cols = np.nonzero(valid)
    points = project(rows, cols)
    colors = rgb[rows, cols, :3]
    upper_edge = np.empty((0, 3), dtype=np.float32)
    if h >= 3:
        lower_row = min(h - 1, max(2, round(.04 * h)))
        middle_row = lower_row // 2
        cols = np.flatnonzero(valid[0] & valid[middle_row] & valid[lower_row])
        top = project(0, cols)
        middle = project(middle_row, cols)
        bottom = project(lower_row, cols)
        # Trace the surface up to the actual image boundary. A ceiling hit
        # has little world-Y change across these rows and fails this check.
        upper_delta = top - middle
        lower_delta = middle - bottom

        def rises(delta):
            rise = delta[:, 1]
            sideways = np.linalg.norm(delta[:, [0, 2]], axis=1)
            return (rise > .005) & (sideways <= .8 * rise + .02)

        upper_edge = top[rises(upper_delta) & rises(lower_delta)]
    return points, colors, upper_edge


def voxel_totals(points, size, colors=None):
    keys = np.floor(points / size).astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    values = points if colors is None else np.column_stack((points, colors))
    totals = np.column_stack([np.bincount(inverse.reshape(-1), weights=axis, minlength=len(unique))
                              for axis in values.T])
    return unique, totals, counts


class VoxelCloud:
    """Incremental position and RGB averages per occupied voxel."""

    def __init__(self, size=.025):
        self.size = size
        self.cells = {}

    def add(self, points, colors):
        if not len(points):
            return
        keys, totals, counts = voxel_totals(points, self.size, colors)
        for key, total, count in zip(map(tuple, keys.tolist()), totals.tolist(), counts.tolist()):
            entry = self.cells.get(key)
            if entry is None:
                self.cells[key] = [*total, count]
            else:
                for i, value in enumerate(total):
                    entry[i] += value
                entry[-1] += count

    def points_and_colors(self):
        if not self.cells:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        totals = np.asarray(list(self.cells.values()), dtype=np.float64)
        averages = totals[:, :-1] / totals[:, -1:]
        return averages[:, :3].astype(np.float32), np.rint(averages[:, 3:]).clip(0, 255).astype(np.uint8)


def spread_surface_colors(image, measured, allowed, steps):
    """Extend measured colors locally through pixels in the same map layer."""
    first, last, all_pixels = slice(None, -1), slice(1, None), slice(None)
    neighbors = (((last, all_pixels), (first, all_pixels)),
                 ((first, all_pixels), (last, all_pixels)),
                 ((all_pixels, last), (all_pixels, first)),
                 ((all_pixels, first), (all_pixels, last)))
    for _ in range(steps):
        remaining = allowed & ~measured
        if not remaining.any():
            break
        added = np.zeros_like(measured)
        for target, source in neighbors:
            fill = remaining[target] & measured[source]
            image[target][fill] = image[source][fill]
            remaining[target] &= ~fill
            added[target] |= fill
        if not added.any():
            break
        measured |= added


WALL_COLUMN_SIZE = .12
WALL_EXTENSION_METERS = .60
WALL_STROKE_GROWTH_METERS = .05
THICK_OBSTACLE_RADIUS = .075


def square_neighborhood(mask, radius, require_full):
    """Test a square window using two separable, linear-time pixel sums."""
    window = 2 * radius + 1
    for axis in (0, 1):
        rows = np.moveaxis(mask, axis, 0)
        padded = np.pad(rows, ((radius + 1, radius), (0, 0)))
        totals = np.cumsum(padded, axis=0, dtype=np.int32)
        counts = totals[window:] - totals[:-window]
        mask = np.moveaxis(counts == window if require_full else counts > 0, 0, axis)
    return mask


def extend_wall_mask(obstacle_mask, wall_seeds, pixel_size):
    """Trace thin obstacle contours, then widen the displayed stroke in place."""
    radius = max(2, math.ceil(THICK_OBSTACLE_RADIUS / pixel_size - 1e-9))
    block_cores = square_neighborhood(obstacle_mask, radius, require_full=True)
    blocked = square_neighborhood(block_cores, radius + 1, require_full=False)

    # Only edges of the original obstacle are eligible. Removing blobs from
    # the obstacle mask first would produce artificial edges at the cut.
    interior = np.zeros_like(obstacle_mask)
    interior[1:-1, 1:-1] = (
        obstacle_mask[:-2, 1:-1] & obstacle_mask[2:, 1:-1]
        & obstacle_mask[1:-1, :-2] & obstacle_mask[1:-1, 2:]
    )
    contour = obstacle_mask & ~interior & ~blocked
    walls = wall_seeds & contour
    height, width = walls.shape
    seeds = list(zip(*np.nonzero(walls)))
    distance = {point: 0. for point in seeds}
    frontier = [(0., row, col) for row, col in seeds]
    heapq.heapify(frontier)
    limit = WALL_EXTENSION_METERS / pixel_size
    neighbors = [(dy, dx, math.hypot(dy, dx)) for dy in (-1, 0, 1)
                 for dx in (-1, 0, 1) if dy or dx]
    while frontier:
        length, row, col = heapq.heappop(frontier)
        if length != distance[(row, col)]:
            continue
        walls[row, col] = True
        for dy, dx, cost in neighbors:
            y, x = row + dy, col + dx
            if not (0 <= y < height and 0 <= x < width and contour[y, x]):
                continue
            if dy and dx and not (obstacle_mask[row, x] or obstacle_mask[y, col]):
                continue  # Diagonally touching separate obstacles are not connected.
            next_length = length + cost
            if next_length <= limit + 1e-9 and next_length < distance.get((y, x), math.inf):
                distance[(y, x)] = next_length
                heapq.heappush(frontier, (next_length, y, x))

    # Grow only through connected black pixels, without entering thick blobs.
    paintable = obstacle_mask & ~blocked
    steps = max(2, math.ceil(WALL_STROKE_GROWTH_METERS / pixel_size - 1e-9))
    for _ in range(steps):
        wider = walls.copy()
        wider[1:] |= walls[:-1]
        wider[:-1] |= walls[1:]
        wider[:, 1:] |= walls[:, :-1]
        wider[:, :-1] |= walls[:, 1:]
        wider &= paintable
        if np.array_equal(wider, walls):
            break
        walls = wider
    return walls


class FloorMap:
    def __init__(self, height):
        self.height = float(height)
        self.scene = VoxelCloud()
        self.ground_rays = []
        self.clipped_wall_columns = set()
        self._wall_points = None

    def add_view(self, view, position):
        self.add_views((view,), position)

    def add_views(self, clouds, position):
        self._wall_points = None
        ground_samples = []
        for points, colors, upper_edge in clouds:
            relative = points[:, 1] - self.height
            # Retain ceiling-level measurements for the wall preview. Ordinary
            # obstacles still use the original lower height band at rendering.
            retained = (relative > -1.0) & (relative < 4.5)
            self.scene.add(points[retained], colors[retained])
            heights = upper_edge[:, 1] - self.height
            edge_xy = upper_edge[(heights >= .15) & (heights < 4.5)][:, [0, 2]]
            columns = np.floor(edge_xy / WALL_COLUMN_SIZE).astype(np.int32)
            self.clipped_wall_columns.update(map(tuple, columns.tolist()))
            ground = points[(relative > -.4) & (relative <= .2)]
            if len(ground):
                ground_samples.append(ground)
        if not ground_samples:
            return
        _, totals, counts = voxel_totals(np.concatenate(ground_samples), .05)
        endpoints = (totals[:, [0, 2]] / counts[:, None]).astype(np.float32)
        origin = np.asarray(position, dtype=np.float32)[[0, 2]]
        self.ground_rays.append((origin, endpoints))

    def estimate_ceiling(self, scene, relative):
        elevated = (relative >= 2.0) & (relative < 4.5)
        ceiling_candidates = relative[elevated]
        if len(ceiling_candidates) < 24:
            return None
        edges = np.linspace(2., 4.5, 26)
        counts, _ = np.histogram(ceiling_candidates, bins=edges)
        xy = scene[elevated][:, [0, 2]]
        for peak in np.argsort(counts)[::-1]:
            if counts[peak] < max(24, 2 * np.median(counts)):
                break
            band = (ceiling_candidates >= edges[peak]) & (ceiling_candidates < edges[peak + 1])
            height = float(np.median(ceiling_candidates[band]))
            cells = set(map(tuple, np.floor(xy[band] / WALL_COLUMN_SIZE).astype(np.int32).tolist()))
            # A strong height histogram bin can come from a long vertical wall.
            # Look for a measured patch spanning both horizontal axes.
            patches = sum((x + 1, z) in cells and (x, z + 1) in cells and (x + 1, z + 1) in cells
                          for x, z in cells)
            if len(cells) >= 18 and patches >= 6:
                return height
        return None

    def wall_footprint(self, scene=None):
        if self._wall_points is not None:
            return self._wall_points
        if scene is None:
            scene, _ = self.scene.points_and_colors()
        relative = scene[:, 1] - self.height
        ceiling = self.estimate_ceiling(scene, relative)
        upper = 4.5 if ceiling is None else ceiling + .12
        selected = (relative >= .15) & (relative <= upper)
        points, heights = scene[selected], relative[selected]
        xy = points[:, [0, 2]]
        columns = np.floor(xy / WALL_COLUMN_SIZE).astype(np.int32)
        keys, inverse = np.unique(columns, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        low = np.full(len(keys), np.inf)
        high = np.full(len(keys), -np.inf)
        np.minimum.at(low, inverse, heights)
        np.maximum.at(high, inverse, heights)
        bands = np.floor(heights / .15).astype(np.int32)
        observed = np.unique(np.column_stack((inverse, bands)), axis=0)
        bands_per_column = np.bincount(observed[:, 0], minlength=len(keys))
        span = np.floor(high / .15) - np.floor(low / .15) + 1
        supported = (low <= .55) & (bands_per_column >= .65 * span)
        # A visible ceiling elsewhere in the panorama does not resolve the
        # height of a vertical surface cut off by this view's upper edge.
        clipped = np.array([tuple(key) in self.clipped_wall_columns for key in keys], dtype=bool)
        tall = clipped & (bands_per_column >= 4)
        if ceiling is not None:
            tall |= high >= ceiling - .45
        self._wall_points = xy[(supported & tall)[inverse] & (heights > .2) & (heights < 1.8)]
        return self._wall_points

    def render(self, pixel_size, max_size, wall_overlay=False):
        scene, colors = self.scene.points_and_colors()
        relative = scene[:, 1] - self.height
        floor_hits = (relative > -.4) & (relative <= .2)
        obstacle_hits = (relative > .2) & (relative < 1.8)
        ground_points = scene[floor_hits][:, [0, 2]]
        obstacles = scene[obstacle_hits][:, [0, 2]]
        clouds = [endpoints for _, endpoints in self.ground_rays]
        if self.ground_rays:
            clouds.append(np.stack([origin for origin, _ in self.ground_rays]))
        if len(obstacles):
            clouds.append(obstacles)
        if len(ground_points):
            clouds.append(ground_points)
        if not clouds:
            return Image.new('RGB', (128, 128), 'white')

        lower = np.min([cloud.min(axis=0) for cloud in clouds], axis=0)
        upper = np.max([cloud.max(axis=0) for cloud in clouds], axis=0)
        padding = 20
        pixel_size = max(pixel_size, float((upper - lower).max()) / (max_size - 2 * padding - 1))
        width, height = np.minimum(max_size, np.ceil((upper - lower) / pixel_size).astype(int) + 2 * padding + 1)
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        def to_pixels(points):
            return np.floor((points - lower) / pixel_size).astype(int) + padding

        ground_mask = Image.new('L', (int(width), int(height)))
        draw = ImageDraw.Draw(ground_mask)
        for origin, endpoints in self.ground_rays:
            x, y = to_pixels(origin).tolist()
            for end_x, end_y in np.unique(to_pixels(endpoints), axis=0).tolist():
                draw.line((x, y, end_x, end_y), fill=255)
        ground = np.asarray(ground_mask) != 0
        # Match the one-pixel footprint of the obstacle samples.
        expanded = ground.copy()
        expanded[1:] |= ground[:-1]
        expanded[:-1] |= ground[1:]
        expanded[:, 1:] |= ground[:, :-1]
        expanded[:, :-1] |= ground[:, 1:]
        image[expanded] = 200

        def footprint(points):
            mask = np.zeros((height, width), dtype=bool)
            x, y = to_pixels(points).T
            for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                mask[y + dy, x + dx] = True
            return mask

        def paint_top_surface(points, heights, rgb, allowed):
            x, y = to_pixels(points).T
            indices = y * width + x
            order = np.lexsort((-heights, indices))
            _, first_hit = np.unique(indices[order], return_index=True)
            visible = order[first_hit]
            visible = visible[allowed[y[visible], x[visible]]]
            image[y[visible], x[visible]] = rgb[visible]
            measured = np.zeros_like(allowed)
            measured[y[visible], x[visible]] = True
            return measured

        occupied = footprint(obstacles)
        if len(ground_points):
            floor_area = (expanded | footprint(ground_points)) & ~occupied
            painted = paint_top_surface(ground_points, relative[floor_hits], colors[floor_hits], floor_area)
            spread_surface_colors(image, painted, floor_area, min(8, int(.10 / pixel_size)))
        if len(obstacles):
            painted = paint_top_surface(obstacles, relative[obstacle_hits], colors[obstacle_hits], occupied)
            spread_surface_colors(image, painted, occupied, 1)
        if wall_overlay:
            walls = self.wall_footprint(scene)
            if len(walls):
                red = extend_wall_mask(occupied, footprint(walls), pixel_size)
                image[red] = (220, 35, 35)
        return Image.fromarray(image)


class GeometryReplay:
    def __init__(self, env, sensors, args):
        self.env = env
        self.cfg = sensors.depth_sensor
        self.depth_key = getattr(self.cfg, 'uuid', 'depth')
        self.rgb_key = getattr(sensors.rgb_sensor, 'uuid', 'rgb')
        self.args = args
        self.floors = []
        self.active = None
        self.pending = []

    def _nearest_floor(self, height):
        return min(self.floors, key=lambda floor: abs(floor.height - height))

    def _flush_pending(self):
        for position, view in self.pending:
            self._nearest_floor(position[1]).add_view(view, position)
        self.pending.clear()

    def _project(self, raw, agent):
        import quaternion

        sensor = agent.sensor_states[self.depth_key]
        return project_depth(raw[self.depth_key], raw[self.rgb_key], self.cfg, np.asarray(sensor.position),
                             quaternion.as_rotation_matrix(sensor.rotation))

    def fuse(self, raw):
        agent = self.env.sim.get_agent_state()
        position = np.asarray(agent.position).copy()
        view = self._project(raw, agent)
        if self.active is None:
            self.active = FloorMap(position[1])
            self.floors.append(self.active)
        nearest = self._nearest_floor(position[1])
        if abs(nearest.height - position[1]) <= .22:
            self.active = nearest
            self._flush_pending()
            self.active.add_view(view, position)
            return

        self.pending.append((position, view))
        if len(self.pending) < 3:
            return
        heights = np.array([pose[1] for pose, _ in self.pending])
        estimate = float(np.median(heights))
        if np.ptp(heights) <= .10 and abs(self._nearest_floor(estimate).height - estimate) >= .65:
            self.active = FloorMap(estimate)
            self.floors.append(self.active)
            self._flush_pending()
        else:
            pose, cloud = self.pending.pop(0)
            self._nearest_floor(pose[1]).add_view(cloud, pose)

    def initialize(self, raw):
        import quaternion

        sim = self.env.sim
        state = copy.deepcopy(sim.get_agent_state())
        if self.args.initial_scan == 'none':
            self.fuse(raw)
            return

        yaw_spacing = min(15., self.cfg.hfov / 2.)
        view_count = math.ceil(360. / yaw_spacing)
        previous = getattr(sim, '_prev_sim_obs', None)
        floor = FloorMap(state.position[1])

        def panorama_clouds():
            yield self._project(raw, state)
            for index in range(1, view_count):
                yaw = -2 * math.pi * index / view_count
                rotation = state.rotation * quaternion.from_rotation_vector([0., yaw, 0.])
                observation = sim.get_observations_at(state.position, rotation, keep_agent_at_new_pose=True)
                yield self._project(observation, sim.get_agent_state())

        try:
            floor.add_views(panorama_clouds(), np.asarray(state.position))
        finally:
            sim.get_agent(0).set_state(state, reset_sensors=False, infer_sensor_states=False)
            if hasattr(sim, '_prev_sim_obs'):
                sim._prev_sim_obs = previous
        self.active = floor
        self.floors.append(floor)
        print(f'[hsgm-geometry] initial panorama: {view_count} views', flush=True)

    def run(self, rows):
        self.initialize(self.env.reset())

        for row in (rows[:self.args.max_steps] if self.args.max_steps else rows):
            if self.env.episode_over:
                break
            self.fuse(self.env.step({'action': ACTION_KEYS[row['action']]}))
            if row['action'] == 0:
                break
        self._flush_pending()


def main():
    parser = argparse.ArgumentParser(description='Replay Habitat actions and export RGB-D top-down PNGs')
    parser.add_argument('--actions_file')
    parser.add_argument('--scene_id')
    parser.add_argument('--episode_id')
    parser.add_argument('--records_root', default=str(ROOT / 'results_base/val_unseen/streamvln_baseline_full/records'))
    parser.add_argument('--start_folder')
    parser.add_argument('--max_episodes', '--max_trajectories', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--habitat_config_path', default=str(ROOT / 'config/vln_r2r.yaml'))
    parser.add_argument('--habitat_root', default='/mnt/disk8t1/zwm/Progress-Aware-VLN/habitat-lab')
    parser.add_argument('--eval_split', default='val_unseen')
    parser.add_argument('--data_path')
    parser.add_argument('--scenes_dir')
    parser.add_argument('--output_path', default='results/hsgm_geometry_rgb')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sensor_width', type=int, default=0)
    parser.add_argument('--sensor_height', type=int, default=0)
    parser.add_argument('--initial_scan', choices=('none', 'panorama'), default='panorama')
    parser.add_argument('--bev_pixel_size', type=float, default=.025)
    parser.add_argument('--bev_max_size', type=int, default=4096)
    parser.add_argument('--wall_overlay', action='store_true', help='Draw the original red wall overlay on top of RGB')
    args = parser.parse_args()
    if min(args.max_steps, args.max_episodes, args.sensor_width, args.sensor_height) < 0:
        parser.error('step/episode limits and sensor sizes must be nonnegative')
    if bool(args.sensor_width) != bool(args.sensor_height):
        parser.error('set both sensor width and height')
    if not math.isfinite(args.bev_pixel_size) or args.bev_pixel_size <= 0 or args.bev_max_size < 128:
        parser.error('BEV pixel size must be positive and maximum size must be at least 128')
    if args.actions_file and args.start_folder:
        parser.error('--start_folder is for batch mode')
    if not args.actions_file and (args.scene_id is not None or args.episode_id is not None):
        parser.error('scene/episode overrides require --actions_file')

    os.chdir(ROOT)
    output = Path(args.output_path).expanduser().resolve()
    if args.actions_file:
        paths = [args.actions_file]
    else:
        _, _, jobs = batch_jobs(args)
        paths = [job['actions_file'] for job in jobs]
    output.mkdir(parents=True, exist_ok=True)
    for actions_file in paths:
        args.actions_file = actions_file
        path, rows, scene, episode_id = read_replay(args)
        name = path.parent.name
        ensure_unused_name(output, name)
        env, sensors, _ = replay_environment(args, rows, scene, episode_id)
        try:
            replay = GeometryReplay(env, sensors, args)
            replay.run(rows)
            for index, floor in enumerate(sorted(replay.floors, key=lambda item: item.height), 1):
                target = output / f'{name}_{index:02}.png'
                temporary = target.with_suffix('.tmp')
                try:
                    floor.render(args.bev_pixel_size, args.bev_max_size, args.wall_overlay).save(temporary, format='PNG')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                print(f'[hsgm-geometry] saved {target}', flush=True)
        finally:
            env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
