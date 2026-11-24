"""HSGM-style geometry replay with wall-constrained region polygons.

Reference: HSGM Sec. 3.3 and the public implementation's geometric mapper.
https://github.com/Teacher-Tom/HSGM_public

This NumPy adaptation uses world-frame voxel centroids and the published
multiview ground/obstacle split. Overlapping depth views form the initial
panorama. Nearby measured ground points form the visible floor footprint.
It uses the replay rig's depth intrinsics and poses. Floor grouping uses stable
agent heights in place of HSGM's stair and landing detector. Output is a PNG
per visited floor, with pastel regions, gray ground, black obstacles, red walls
and a white background. The latest PNG is replaced after each map update.
Region geometry uses world coordinates in meters.
"""

import argparse
import colorsys
import copy
import heapq
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely import coverage_union_all, intersects_xy, polygons, prepare
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree


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
            if args.max_steps and len(rows) >= args.max_steps:
                break
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
        cfg.environment.max_episode_steps = len(rows) + 1
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


MIN_OPENING_WIDTH = .55
MAX_OPENING_WIDTH = 2.00
MIN_SIDE_HEIGHT = .95
MIN_DEPTH_JUMP = .75


class OpeningCandidate(NamedTuple):
    segments: tuple
    score: float
    alignment: float = 0.
    perpendicularity: float = .5
    travel_axis: tuple = ()


def same_opening(first, second):
    if np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)) > .08:
        return False
    direction = first[1] - first[0]
    other = second[1] - second[0]
    if abs(direction @ other) < math.cos(math.radians(8)) * np.linalg.norm(direction) * np.linalg.norm(other):
        return False
    return min(np.linalg.norm(first - second, axis=1).max(),
               np.linalg.norm(first - second[::-1], axis=1).max()) <= .08


def add_candidate(candidates, candidate):
    for i, existing in enumerate(candidates):
        if all(any(same_opening(mouth, old) for old in existing.segments) for mouth in candidate.segments):
            if len(candidate.segments) == len(existing.segments) and candidate.score > existing.score:
                candidates[i] = candidate
            return
    if len(candidate.segments) == 2:
        candidates[:] = [old for old in candidates if len(old.segments) == 2 or
                         not any(same_opening(old.segments[0], mouth) for mouth in candidate.segments)]
    candidates.append(candidate)


def opening_conflicts(segment, openings):
    line = LineString(segment)
    for other in openings:
        existing = LineString(other)
        if line.crosses(existing) or line.intersection(existing).length > .01:
            return True
    return False


def overlaps_opening(first, second):
    direction, other = first[1] - first[0], second[1] - second[0]
    width, other_width = np.linalg.norm(direction), np.linalg.norm(other)
    axis = direction / width
    if abs(axis @ other) < math.cos(math.radians(38)) * other_width:
        return False
    normal = np.array([-axis[1], axis[0]])
    if abs((second.mean(axis=0) - first.mean(axis=0)) @ normal) > .40:
        return False
    lo, hi = sorted((second - first[0]) @ axis)
    return min(width, hi) - max(0., lo) >= .65 * min(width, hi - lo)


def corridor_contains_opening(segment, pair):
    outer, inner = pair
    if np.linalg.norm(inner[0] - outer[0]) > np.linalg.norm(inner[1] - outer[0]):
        inner = inner[::-1]
    axis = inner.mean(axis=0) - outer.mean(axis=0)
    fraction = float((segment.mean(axis=0) - outer.mean(axis=0)) @ axis / (axis @ axis))
    return 0. < fraction < 1. and overlaps_opening(outer + fraction * (inner - outer), segment)


def boundary_perpendicularity(candidate, boundary):
    """Prefer perpendicular mouths where the endpoint has a clear local edge."""
    if boundary is None or not boundary.n:
        return .5
    scores = []
    for segment in candidate.segments:
        direction = segment[1] - segment[0]
        direction = direction / np.linalg.norm(direction)
        for endpoint in segment:
            distance, nearest = boundary.query(endpoint)
            score = .5
            if distance > .13:
                scores.append(score)
                continue
            samples = boundary.data[boundary.query_ball_point(boundary.data[nearest], .24)]
            if len(samples) >= 5:
                centered = samples - samples.mean(axis=0)
                values, vectors = np.linalg.eigh(centered.T @ centered / len(samples))
                if values[1] >= .045 ** 2 and values[0] <= .30 * values[1]:
                    score = 1. - abs(direction @ vectors[:, 1])
            scores.append(score)
    return float(np.mean(scores))


def refine_corridor_mouths(candidate, boundary):
    if len(candidate.segments) != 2 or boundary is None or not boundary.n:
        return candidate
    outer, inner = candidate.segments
    if np.linalg.norm(inner[0] - outer[0]) > np.linalg.norm(inner[1] - outer[0]):
        inner = inner[::-1]
    delta = inner.mean(axis=0) - outer.mean(axis=0)
    length = np.linalg.norm(delta)
    along = np.asarray(candidate.travel_axis) if candidate.travel_axis else delta / length
    if along @ delta < 0:
        along = -along
    across = np.array([-along[1], along[0]])

    def extension(mouth, direction, limit):
        if limit < .12:
            return 0.
        extents = []
        for endpoint in mouth:
            center = endpoint + .5 * limit * direction
            nearby = boundary.data[boundary.query_ball_point(center, .5 * limit + .18)] - endpoint
            offsets = nearby @ direction
            supported = np.sort(offsets[(np.abs(nearby @ across) <= .09) & (offsets >= -.12) & (offsets <= limit)])
            if not len(supported) or supported[0] > .08:
                extents.append(0.)
                continue
            gaps = np.flatnonzero(np.diff(supported) > .12)
            last = supported[gaps[0] if len(gaps) else -1]
            if not len(gaps) and last >= limit - .10:
                return None  # The wall still continues. Defer the pair.
            extents.append(max(0., last))
        distance = min(extents)
        if abs(extents[0] - extents[1]) > .22:
            return None
        return float(distance) if distance >= .12 else 0.

    front = extension(outer, -along, min(.95, 4.0 - length))
    if front is None:
        return None
    back = extension(inner, along, min(.95, 4.0 - length - front))
    if back is None:
        return None
    return candidate._replace(segments=(outer - front * along, inner + back * along), travel_axis=tuple(along))


def measured_passage_axis(segment, observed):
    """Estimate the longitudinal axis from narrow, measured floor sections."""
    if observed is None or observed.is_empty:
        return ()
    across = segment[1] - segment[0]
    width = np.linalg.norm(across)
    across = across / width
    along = np.array([-across[1], across[0]])
    center = segment.mean(axis=0)
    stations = np.arange(.22, 1.01, .17)
    lateral = np.arange(-width - .50, width + .54, .04)
    estimates = []
    for sign in (-1, 1):
        slices = center + sign * stations[:, None, None] * along + lateral[None, :, None] * across
        visible = intersects_xy(observed, slices[:, :, 0], slices[:, :, 1])
        centers, widths = [], []
        previous = 0.
        for offsets in visible:
            edges = np.diff(np.r_[False, offsets, False].astype(np.int8))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            options = [(lateral[a], lateral[b - 1]) for a, b in zip(starts, ends) if a and b < len(lateral)
                       and .52 * width <= lateral[b - 1] - lateral[a] <= min(1.70 * width, width + .65)]
            if not options:
                if centers:
                    break
                continue
            low, high = min(options, key=lambda item: abs((item[0] + item[1]) / 2 - previous))
            midpoint = (low + high) / 2
            if abs(midpoint - previous) > (.35 if centers else .40):
                break
            centers.append(midpoint)
            widths.append(high - low)
            previous = midpoint
        if len(centers) < 3 or np.ptp(widths) > .40 * width:
            continue
        offsets = np.arange(len(centers)) * .17
        slope, intercept = np.polyfit(offsets, centers, 1)
        if np.max(np.abs(np.asarray(centers) - (slope * offsets + intercept))) > .085:
            continue
        axis = along + sign * slope * across
        axis = axis / np.linalg.norm(axis)
        if axis @ along >= .66:
            estimates.append((float(np.median(widths)), axis))
    if not estimates:
        return ()
    estimates.sort(key=lambda item: item[0])
    if len(estimates) == 2 and estimates[0][1] @ estimates[1][1] < .90 and estimates[1][0] - estimates[0][0] < .20:
        return ()
    return tuple(estimates[0][1])


def snap_opening_mouth(segment, boundary, observed, radius, travel_axis=(), outward=None):
    """Use nearby supported wall tips or corners as the two mouth anchors."""
    direction = segment[1] - segment[0]
    width = np.linalg.norm(direction)
    axis = direction / width
    normal = np.array([-axis[1], axis[0]])
    anchors = []
    for endpoint in segment:
        options = []
        nearby = boundary.data[boundary.query_ball_point(endpoint, radius)]
        # Candidate anchors closer than this are nearly indistinguishable;
        # their support still uses the original dense boundary samples.
        cells = np.floor((nearby - endpoint) / .035).astype(int)
        _, representative = np.unique(cells, axis=0, return_index=True)
        nearby = nearby[representative]
        supports = boundary.query_ball_point(nearby, .21)
        forward_supports = boundary.query_ball_point(nearby + .20 * outward, .18) if outward is not None else None
        for index, point in enumerate(nearby):
            if outward is not None:
                forward = boundary.data[forward_supports[index]] - point
                lateral = np.array([-outward[1], outward[0]])
                continued = forward[(forward @ outward >= .12) & (np.abs(forward @ lateral) <= .065)]
                if len(continued) >= 3 and np.ptp(continued @ outward) >= .08:
                    continue
            support = boundary.data[supports[index]] - point
            distances = np.linalg.norm(support, axis=1)
            ring = (distances >= .065) & (distances <= .21)
            if np.count_nonzero(ring) < 5 or distances.max() < .14:
                continue
            # A wall tip or corner has predominantly one-sided support.
            corner = np.linalg.norm((support[ring] / distances[ring, None]).mean(axis=0))
            if corner < .28:
                continue
            shift = np.linalg.norm(point - endpoint)
            options.append((.85 * corner - 1.25 * shift, point, support[ring]))
        distinct = []
        for score, point, support in sorted(options, key=lambda row: row[0], reverse=True):
            if all(np.linalg.norm(point - old) >= .055 for _, old, _ in distinct):
                distinct.append((score, point, support))
            if len(distinct) == 10:
                break
        if not distinct:
            return None
        anchors.append(distinct)

    def wall_continuation(support, outward):
        along = support @ outward
        across = np.abs(support @ np.array([-outward[1], outward[0]]))
        aligned = along[(along >= .07) & (across <= .018 + .12 * along)]
        return min(1., float(aligned.max() - .07) / .12) if len(aligned) >= 3 else 0.

    def side_direction(support):
        values, vectors = np.linalg.eigh(support.T @ support)
        if values[0] > .24 * values[1]:
            return None
        tangent = vectors[:, 1]
        return tangent if tangent @ normal >= 0 else -tangent

    best = None
    for left_score, left, left_support in anchors[0]:
        for right_score, right, right_support in anchors[1]:
            gap = right - left
            length = np.linalg.norm(gap)
            if not MIN_OPENING_WIDTH <= length <= MAX_OPENING_WIDTH or gap @ axis < .72 * length:
                continue
            inside = np.linspace(.12 / length, 1. - .12 / length, max(9, math.ceil(length / .03)))
            samples = left + inside[:, None] * gap
            if np.any(boundary.query(samples)[0] < .027):
                continue
            if observed is not None and intersects_xy(observed, samples[:, 0], samples[:, 1]).mean() < .60:
                continue
            across_opening = gap / length
            longitudinal = np.asarray(travel_axis) if len(travel_axis) else None
            if longitudinal is None:
                one, two = side_direction(left_support), side_direction(right_support)
                if one is not None and two is not None and one @ two >= .90 and min(one @ normal, two @ normal) >= .73:
                    longitudinal = (one + two) / np.linalg.norm(one + two)
            angle_error = abs(across_opening @ longitudinal) if longitudinal is not None else abs(across_opening @ normal)
            if len(travel_axis) and angle_error > .14:
                continue
            continuity = sorted((wall_continuation(left_support, -across_opening),
                                 wall_continuation(right_support, across_opening)))
            score = ((left_score + right_score) / 2 + .95 * continuity[1] + .75 * continuity[0]
                     - .35 * max(0., length - width) - 1.4 * angle_error)
            if best is None or score > best[1]:
                best = np.stack((left, right)), score
    return best


def fit_opening(candidate, boundary, observed):
    candidate = refine_corridor_mouths(candidate, boundary)
    if candidate is None or boundary is None or not boundary.n:
        return None
    radius = .23 if len(candidate.segments) == 2 else .42
    travel = candidate.travel_axis or measured_passage_axis(candidate.segments[0], observed)
    outward = (-np.asarray(travel), np.asarray(travel)) if len(candidate.segments) == 2 else (None,)
    fitted = [snap_opening_mouth(segment, boundary, observed, radius, travel, direction)
              for segment, direction in zip(candidate.segments, outward)]
    if any(result is None for result in fitted):
        return None
    segments, scores = zip(*fitted)
    if len(segments) == 2:
        old_axis = candidate.segments[1].mean(axis=0) - candidate.segments[0].mean(axis=0)
        new_axis = segments[1].mean(axis=0) - segments[0].mean(axis=0)
        if new_axis @ old_axis < .45 * np.linalg.norm(old_axis):
            return None
    fitted = candidate._replace(segments=segments, alignment=float(np.mean(scores)), travel_axis=travel)
    return fitted._replace(perpendicularity=boundary_perpendicularity(fitted, boundary))


def suppress_openings(fitted):
    """Keep the best-supported mouth and accept corridor pairs atomically."""
    ranked = sorted((candidate for candidate in fitted if candidate is not None), key=lambda candidate: (
        -len(candidate.segments), -candidate.alignment, -candidate.perpendicularity, -candidate.score,
        sum(np.linalg.norm(segment[1] - segment[0]) for segment in candidate.segments)))
    kept, corridors, selected = [], [], []
    for candidate in ranked:
        segments = candidate.segments
        if len(segments) == 1 and any(corridor_contains_opening(segments[0], pair) for pair in corridors):
            continue
        if any(any(overlaps_opening(segment, old) for old in kept) or
               opening_conflicts(segment, (*kept, *segments[:i])) for i, segment in enumerate(segments)):
            continue
        kept.extend(segments)
        selected.append(candidate)
        if len(segments) == 2:
            corridors.append(segments)
    return selected


def update_opening_cache(candidates, cache, view_samples, occupied, origin, pixel_size, observed):
    """Refit new mouths and those near depth samples from the latest views."""
    active = {id(candidate): candidate for candidate in candidates}
    refresh = {key for key in active if key not in cache}
    if view_samples and len(active) > len(refresh):
        support = cKDTree(np.concatenate(view_samples))
        for key, candidate in active.items():
            if key in refresh:
                continue
            fitted = cache[key][1]
            segments = candidate.segments if fitted is None else candidate.segments + fitted.segments
            probes = [point for segment in segments for point in (*segment, segment.mean(axis=0))]
            reach = 1.05 if len(candidate.segments) == 2 else .50
            if np.any(support.query(probes, distance_upper_bound=reach)[0] < reach):
                refresh.add(key)
    if not refresh and len(active) == len(cache):
        return None

    boundary = None
    if refresh:
        prepare(observed)
    if refresh and occupied.any():
        edges = occupied.copy()
        edges[1:-1, 1:-1] &= ~(occupied[:-2, 1:-1] & occupied[2:, 1:-1]
                                & occupied[1:-1, :-2] & occupied[1:-1, 2:])
        z, x = np.nonzero(edges)
        boundary = cKDTree((np.column_stack((x, z)) + origin + .5) * pixel_size)
    for key in cache.keys() - active.keys():
        del cache[key]
    for key, candidate in active.items():
        if key in refresh:
            # Keep the original object alongside the fit so its identity cannot be reused.
            cache[key] = candidate, fit_opening(candidate, boundary, observed)
    return suppress_openings(cache[key][1] for key in active)


def depth_openings(depth, visible, project, floor_height):
    """Detect doors and distinguish paired corridor mouths in the same view."""
    height, width = depth.shape
    candidates = []
    edges = np.diff(np.where(visible, depth, np.nan), axis=1)
    row_stride = max(1, height // 48)
    support_cache = {}

    def tall_side(column, outward, near):
        key = (column, outward, round(float(near) / .05))
        if key in support_cache:
            return support_cache[key]
        tolerance = .13 + .03 * near
        for col in (column, column + outward, column + 2 * outward):
            if not 0 <= col < width:
                continue
            rows = np.flatnonzero(visible[:, col] & (np.abs(depth[:, col] - near) <= tolerance))
            splits = np.flatnonzero(np.diff(rows) > max(2, height // 55)) + 1
            for run in np.split(rows, splits):
                if len(run) < 3:
                    continue
                endpoints = np.array([run[0], run[-1]])
                levels = project(endpoints, np.full(2, col))[:, 1] - floor_height
                if np.ptp(levels) >= MIN_SIDE_HEIGHT and levels.min() <= .95 and levels.max() >= 1.15:
                    support_cache[key] = True
                    return True
        support_cache[key] = False
        return False

    def track_side(row, inner, outward):
        columns = [inner]
        profile = depth[row]
        for column in range(inner + outward, width if outward > 0 else -1, outward):
            if not visible[row, column]:
                return None
            previous = profile[columns[-1]]
            change = profile[column] - previous
            if abs(change) > max(.26, .10 * previous):
                break
            columns.append(column)
            if len(columns) >= 12 and np.ptp(profile[columns[-7:]]) <= .06:
                columns = columns[:-6]
                break
            if profile[inner] - profile[column] > 4.0:
                return None
        else:
            return None  # The outer end is still outside the image.

        if len(columns) < 6 or profile[inner] - profile[columns[-1]] < .45:
            return None
        rises = np.maximum(np.diff(profile[columns]), 0.).sum()
        if rises > .15 + .20 * (profile[inner] - profile[columns[-1]]):
            return None
        # A planar side produces a steady slope in inverse depth in image
        # coordinates. Raw depth does not vary linearly under perspective.
        inverse = 1. / profile[columns].astype(float)
        trend = np.linspace(inverse[0], inverse[-1], len(inverse))
        if np.quantile(np.abs(inverse - trend), .85) > max(.008, .12 * (inverse[-1] - inverse[0])):
            return None
        outer = columns[-1]
        if not tall_side(outer, -outward, profile[outer]):
            return None
        points = project(np.full(len(columns), row), np.asarray(columns))[:, [0, 2]]
        side = points[-1] - points[0]
        length = np.linalg.norm(side)
        if not .65 <= length <= 4.0:
            return None
        direction = side / length
        normal = np.array([-direction[1], direction[0]])
        if np.quantile(np.abs((points - points[0]) @ normal), .85) > .15:
            return None
        return points[-1], direction, length

    def corridor_pair(row, left, right, inner):
        left_side, right_side = track_side(row, left, -1), track_side(row, right, 1)
        if left_side is None or right_side is None:
            return None
        left_outer, left_direction, left_length = left_side
        right_outer, right_direction, right_length = right_side
        if left_direction @ right_direction < .94 or abs(left_length - right_length) > .42:
            return None
        outer = np.stack((left_outer, right_outer))
        outer_width = np.linalg.norm(outer[1] - outer[0])
        inner_width = np.linalg.norm(inner[1] - inner[0])
        if not MIN_OPENING_WIDTH <= outer_width <= MAX_OPENING_WIDTH:
            return None
        if abs(outer_width - inner_width) > max(.22, .25 * inner_width):
            return None
        inner_axis = (inner[1] - inner[0]) / inner_width
        outer_axis = (outer[1] - outer[0]) / outer_width
        if inner_axis @ outer_axis < .94 or abs(inner_axis @ left_direction) > .32:
            return None
        travel = -(left_direction + right_direction)
        travel /= np.linalg.norm(travel)
        return (outer, inner), tuple(travel)

    sample_slots = np.arange(13)
    for row in range(row_stride // 2, height, row_stride):
        rises = np.flatnonzero(edges[row] >= MIN_DEPTH_JUMP)
        falls = np.flatnonzero(edges[row] <= -MIN_DEPTH_JUMP) + 1
        if not len(rises) or not len(falls):
            continue
        left_points = project(np.full(len(rises), row), rises)
        right_points = project(np.full(len(falls), row), falls)
        right_levels = right_points[:, 1] - floor_height
        right_usable = (right_levels >= .25) & (right_levels <= 1.75)
        profile = depth[row]
        for left, left_point in zip(rises, left_points):
            if not .25 <= left_point[1] - floor_height <= 1.75:
                continue
            near_left = profile[left]
            widths = np.linalg.norm(right_points[:, [0, 2]] - left_point[[0, 2]], axis=1)
            farthest_near = np.maximum(near_left, profile[falls])
            threshold = farthest_near + MIN_DEPTH_JUMP
            eligible = (right_usable & (falls >= left + 5) & (widths >= MIN_OPENING_WIDTH)
                        & (widths <= MAX_OPENING_WIDTH) & (profile[left + 1] >= threshold)
                        & (profile[falls - 1] >= threshold))
            choices = np.flatnonzero(eligible)
            if not len(choices):
                continue
            spans = falls[choices] - left - 1
            counts = np.minimum(len(sample_slots), spans)
            used = sample_slots < counts[:, None]
            offsets = np.minimum(sample_slots, counts[:, None] - 1) * (spans[:, None] - 1)
            interior = left + 1 + offsets // (counts[:, None] - 1)
            distant = used & visible[row, interior] & (profile[interior] >= threshold[choices, None])
            shares = distant.sum(axis=1) / counts
            supported = np.flatnonzero(shares >= .80)
            if not len(supported) or not tall_side(left, -1, near_left):
                continue
            for i in supported:
                choice = choices[i]
                right = falls[choice]
                near_right = profile[right]
                if not tall_side(right, 1, near_right):
                    continue
                segment = np.stack((left_point[[0, 2]], right_points[choice, [0, 2]]))
                pair = corridor_pair(row, left, right, segment)
                depth_support = np.quantile(profile[interior[i, distant[i]]], .20) - farthest_near[choice]
                jump = min(edges[row, left], -edges[row, right - 1], depth_support, 2.5)
                score = float(jump + .35 * shares[i] - .20 * abs(near_left - near_right))
                segments, travel = pair if pair is not None else ((segment,), ())
                add_candidate(candidates, OpeningCandidate(segments, score, travel_axis=travel))
                break
    return candidates


def project_depth(depth, cfg, position, rotation, floor_height):
    """Return world depth points, upper-edge wall evidence, and opening candidates."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f'Expected H×W depth, got {depth.shape}')
    # The sensor's saturated value is not a measured surface.
    unsaturated = depth < (1.0 if cfg.normalize_depth else cfg.max_depth)
    if cfg.normalize_depth:
        depth = depth * (cfg.max_depth - cfg.min_depth) + cfg.min_depth
    valid = np.isfinite(depth) & unsaturated & (depth >= max(.1, cfg.min_depth))
    h, w = depth.shape
    focal = w / (2 * math.tan(math.radians(cfg.hfov) / 2))

    def project(rows, cols):
        meters = depth[rows, cols]
        camera = np.column_stack(((cols - (w - 1) / 2) * meters / focal,
                                  ((h - 1) / 2 - rows) * meters / focal, -meters))
        return (camera @ rotation.T + position).astype(np.float32)

    points = project(*np.nonzero(valid))
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
    return points, upper_edge, depth_openings(depth, valid, project, floor_height)


def voxel_totals(points, size):
    keys = np.floor(points / size).astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    totals = np.column_stack([np.bincount(inverse.reshape(-1), weights=axis, minlength=len(unique))
                              for axis in points.T])
    return unique, totals, counts


class VoxelCloud:
    """Incremental weighted centroids, with storage proportional to occupied voxels."""

    def __init__(self, size=.025):
        self.size = size
        self.cells = {}

    def add(self, points):
        if not len(points):
            return
        keys, totals, counts = voxel_totals(points, self.size)
        for key, total, count in zip(map(tuple, keys.tolist()), totals.tolist(), counts.tolist()):
            entry = self.cells.get(key)
            if entry is None:
                self.cells[key] = [*total, count]
            else:
                entry[0] += total[0]
                entry[1] += total[1]
                entry[2] += total[2]
                entry[3] += count

    def points(self):
        if not self.cells:
            return np.empty((0, 3), dtype=np.float32)
        totals = np.asarray(list(self.cells.values()), dtype=np.float64)
        return (totals[:, :3] / totals[:, 3:]).astype(np.float32)


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


def thick_obstacle_mask(obstacle_mask, pixel_size):
    radius = max(2, math.ceil(THICK_OBSTACLE_RADIUS / pixel_size - 1e-9))
    block_cores = square_neighborhood(obstacle_mask, radius, require_full=True)
    return square_neighborhood(block_cores, radius + 1, require_full=False)


def extend_wall_mask(obstacle_mask, wall_seeds, pixel_size, blocked):
    """Trace thin obstacle contours, then widen the displayed stroke in place."""
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


REGION_COLORS = ((185, 218, 244), (246, 212, 232), (191, 234, 213),
                 (248, 230, 182), (220, 207, 245), (247, 206, 183))


def wall_segments(points):
    """Fit local lines to height-supported wall points, retaining measured ends."""
    _, totals, counts = voxel_totals(points, .06)
    points = totals / counts[:, None]
    if len(points) < 5:
        return []
    tree = cKDTree(points)
    used = np.zeros(len(points), dtype=bool)
    segments = []
    for index, origin in enumerate(points):
        if used[index]:
            continue
        local = np.asarray(tree.query_ball_point(origin, 2.8), dtype=int)
        offsets = points[local] - origin
        distances = np.linalg.norm(offsets, axis=1)
        directions = offsets[(distances >= .22) & (distances <= .55)]
        if not len(directions):
            continue
        angles = np.arctan2(directions[:, 1], directions[:, 0]) % math.pi
        _, unique = np.unique(np.floor(angles / math.radians(5)), return_index=True)
        best = np.empty(0, dtype=int)
        for angle in angles[unique]:
            axis = np.array([math.cos(angle), math.sin(angle)])
            normal = np.array([-axis[1], axis[0]])
            close = np.flatnonzero(np.abs(offsets @ normal) <= .055)
            along = offsets[close] @ axis
            order = np.argsort(along)
            runs = np.split(close[order], np.flatnonzero(np.diff(along[order]) > .14) + 1)
            support = next(run for run in runs if index in local[run])
            if len(support) > len(best):
                best = support
        if len(best) < 5:
            continue
        samples = points[local[best]]
        center = samples.mean(axis=0)
        _, _, axes = np.linalg.svd(samples - center, full_matrices=False)
        axis = axes[0]
        along = (samples - center) @ axis
        if np.ptp(along) < .30:
            continue
        used[local[best]] = True
        start, end = center + along.min() * axis, center + along.max() * axis
        for i, (a, b, direction) in enumerate(segments):
            normal = np.array([-direction[1], direction[0]])
            if abs(axis @ direction) < math.cos(math.radians(8)):
                continue
            if max(abs((start - a) @ normal), abs((end - a) @ normal)) > .06:
                continue
            low, high = sorted(((start - a) @ direction, (end - a) @ direction))
            if min(high, (b - a) @ direction) - max(low, 0.) <= .04:
                continue
            segments[i] = (a + min(0., low) * direction,
                           a + max((b - a) @ direction, high) * direction, direction)
            break
        else:
            segments.append((start, end, axis))
    return segments


def observed_floor(ground_frames):
    """Build local surfaces from measured floor samples, preserving view gaps."""
    patches = []
    for frame in ground_frames:
        points = np.unique(frame[np.isfinite(frame).all(axis=1)], axis=0)
        if not len(points):
            continue
        supported = np.zeros(len(points), dtype=bool)
        if len(points) >= 3:
            try:
                triangles = Delaunay(points).simplices
            except QhullError:
                triangles = np.empty((0, 3), dtype=int)
            corners = points[triangles].astype(np.float64)
            edges = corners - np.roll(corners, -1, axis=1)
            short = np.all(np.sum(edges * edges, axis=2) <= .14 ** 2, axis=1)
            area_twice = np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
            triangles = triangles[short & (area_twice > 1e-10)]
            if len(triangles):
                tiles = polygons(points[triangles])
                try:
                    surface = coverage_union_all(tiles)
                except GEOSException:
                    surface = unary_union(tiles)
                else:
                    if not surface.is_valid:
                        surface = unary_union(tiles)
                patches.append(surface)
                supported[triangles.ravel()] = True
        if not supported.all():
            patches.append(MultiPoint(points[~supported]).buffer(.025, quad_segs=3))
    return unary_union(patches)


def polygon_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def wall_region_polygons(wall_points, observed, candidates, obstacles=None):
    """Partition measured floor using fitted walls and depth-detected mouths."""
    segments = wall_segments(wall_points) if len(wall_points) else []
    wall_barrier = unary_union([LineString([a, b]).buffer(.065) for a, b, _ in segments])
    obstacle_barrier = GeometryCollection()
    if candidates and obstacles is not None and len(obstacles):
        # Include measured black obstacles near each mouth in connectivity.
        obstacle_index = cKDTree(obstacles)
        nearby = set()
        for candidate in candidates:
            for segment in candidate.segments:
                half_width = np.linalg.norm(segment[1] - segment[0]) / 2
                nearby.update(obstacle_index.query_ball_point(segment.mean(axis=0), half_width + 1.40))
        if nearby:
            local = obstacles[np.fromiter(nearby, dtype=int)]
            obstacle_barrier = observed_floor([local]).buffer(.022, quad_segs=3)
    structural = unary_union([wall_barrier, obstacle_barrier])
    free = observed.difference(structural)
    free_boundary = free.boundary

    def cut_anchor(endpoint, direction):
        # The fitted yellow endpoint is on the measured obstacle outline.
        # Its partition boundary can still have a small gap to the wall fit.
        reach = LineString([endpoint, endpoint + .30 * direction])
        contacts = reach.intersection(free_boundary)
        if not contacts.is_empty:
            contact = nearest_points(Point(endpoint), contacts)[1]
            return np.asarray(contact.coords[0])
        return endpoint + .08 * direction

    proposals = []
    for candidate in candidates:
        lines = [LineString(segment) for segment in candidate.segments]
        if any(line.intersection(free).length < .35 * line.length for line in lines):
            continue
        centers = [segment.mean(axis=0) for segment in candidate.segments]
        if len(lines) == 2:
            axis = centers[1] - centers[0]
        elif candidate.travel_axis:
            axis = np.asarray(candidate.travel_axis)
        else:
            along = candidate.segments[0][1] - candidate.segments[0][0]
            axis = np.array([-along[1], along[0]])
        axis /= np.linalg.norm(axis)
        cuts = []
        for segment in candidate.segments:
            along = segment[1] - segment[0]
            along /= np.linalg.norm(along)
            start = cut_anchor(segment[0], -along)
            end = cut_anchor(segment[1], along)
            cuts.append(LineString([start, end]).buffer(.045))
        proposals.append((lines, centers, axis, cuts))

    def probe(index, faces, center, direction):
        lateral = np.array([-direction[1], direction[0]])
        for offset in (.18, .30, .48):
            base = center + offset * direction
            for shift in (0., -.13, .13, -.24, .24):
                found = index.query(Point(base + shift * lateral), predicate='covered_by')
                for i in found:
                    if faces[int(i)].area >= .65:
                        return int(i)
        return None

    accepted = proposals
    while True:
        cuts = [cut for _, _, _, group in accepted for cut in group]
        domain = free.difference(unary_union(cuts)) if cuts else free
        final_faces = list(polygon_parts(domain))
        trial = STRtree(final_faces)
        valid = []
        for lines, centers, axis, group in accepted:
            first = probe(trial, final_faces, centers[0], -axis)
            last = probe(trial, final_faces, centers[-1], axis)
            if first is None or last is None:
                continue
            if len(lines) == 2:
                middle = trial.query(Point((centers[0] + centers[1]) / 2), predicate='covered_by')
                if not len(middle) or int(middle[0]) in (first, last) or final_faces[int(middle[0])].area < .28:
                    continue
            elif first == last:
                continue
            valid.append((lines, centers, axis, group))
        if len(valid) == len(accepted):
            break
        accepted = valid

    index = STRtree(final_faces)
    region_ids = {max(range(len(final_faces)), key=lambda i: final_faces[i].area)} if final_faces else set()
    for lines, centers, axis, _ in accepted:
        region_ids.add(probe(index, final_faces, centers[0], -axis))
        region_ids.add(probe(index, final_faces, centers[-1], axis))
        if len(lines) == 2:
            middle = Point((centers[0] + centers[1]) / 2)
            region_ids.update(map(int, index.query(middle, predicate='covered_by')))
    # A measured wall may itself separate two spaces. An unknown frontier alone
    # cannot create a new colored room, even if it disconnects observed floor.
    for a, b, axis in segments:
        normal = np.array([-axis[1], axis[0]])
        for fraction in (.25, .5, .75):
            center = a + fraction * (b - a)
            sides = [index.query(Point(center + sign * .20 * normal), predicate='covered_by')
                     for sign in (-1, 1)]
            if all(len(side) and final_faces[int(side[0])].area >= .65 for side in sides):
                if sides[0][0] != sides[1][0]:
                    region_ids.update(int(side[0]) for side in sides)

    ordered = sorted(region_ids, key=lambda i: (-final_faces[i].area, final_faces[i].bounds))
    regions = [[final_faces[i]] for i in ordered]
    barriers = unary_union([structural] + [cut for _, _, _, group in accepted for cut in group])
    for i, fragment in enumerate(final_faces):
        if i in region_ids or fragment.area < .025:
            continue
        nearby = sorted((fragment.distance(final_faces[j]), rank) for rank, j in enumerate(ordered))
        for distance, rank in nearby:
            if distance > .65:
                break
            start, end = nearest_points(fragment, regions[rank][0])
            if distance > 0 and barriers.intersects(LineString([start, end])):
                continue
            regions[rank].append(fragment)
            break
    return regions, sum(len(lines) == 1 for lines, _, _, _ in accepted), sum(len(lines) == 2 for lines, _, _, _ in accepted), domain, barriers


def preserve_region_splits(previous, groups, free):
    """Apply new boundaries without combining previously separate regions."""
    if not previous:
        return groups
    old_shapes = list(previous.values())
    index = STRtree(old_shapes)
    proposed = [unary_union(group) for group in groups]
    result = []
    for shape in proposed:
        pieces = [shape.intersection(old_shapes[i])
                  for i in index.query(shape, predicate='intersects')]
        held = [piece for piece in pieces if piece.area > 1e-6]
        if len(held) < 2:
            result.append([shape])
            continue
        separated = [[part] for part in held]
        for fragment in polygon_parts(shape.difference(unary_union(held))):
            neighbors = [i for i, part in enumerate(held) if fragment.distance(part) <= .01]
            if len(neighbors) == 1:
                separated[neighbors[0]].append(fragment)
        result.extend(separated)

    covered = unary_union(proposed)
    for old in old_shapes:
        if old.intersection(covered).area < .25 * old.area:
            remaining = old.intersection(free).difference(covered)
            if remaining.area >= .12:
                result.append([remaining])
    return result


TINY_REGION_AREA = .18
THIN_REGION_AREA = .60
REGION_CLEANUP_REACH = .28


def merge_tiny_regions(groups, domain):
    """Relabel tiny and thin fragments within their local observed floor."""
    result = [[] for _ in groups]
    cores, owners, fragments = [], [], []
    for owner, group in enumerate(groups):
        for part in polygon_parts(unary_union(group)):
            tiny = part.area < TINY_REGION_AREA
            thin = part.area <= THIN_REGION_AREA and part.buffer(-.10).is_empty
            if tiny or thin:
                fragments.append(part)
            else:
                result[owner].append(part)
                cores.append(part)
                owners.append(owner)
    if not cores or not fragments:
        return groups

    index = STRtree(cores)
    for fragment in fragments:
        neighborhood = fragment.buffer(REGION_CLEANUP_REACH)
        nearby = index.query(neighborhood, predicate='intersects')
        if not len(nearby):
            continue
        # The local domain already excludes both walls and accepted yellow cuts.
        # Gray measured floor can connect two colored pieces without filling
        # unknown space or borrowing a room label through an obstacle.
        connected = [part for part in polygon_parts(domain.intersection(neighborhood))
                     if part.intersection(fragment).area > 1e-6]
        choices = []
        for j in nearby:
            core = cores[j]
            if not any(part.intersection(core).area > 1e-5 for part in connected):
                continue
            shared = fragment.boundary.intersection(core.boundary).length
            choices.append((-shared, fragment.distance(core), -core.area, owners[j]))
        if choices:
            result[min(choices)[-1]].append(fragment)
    return [group for group in result if group]


def track_regions(previous, groups, next_id):
    """Keep a one-to-one region ID across boundary edits and splits."""
    shapes = [unary_union(group) for group in groups]
    old_ids = list(previous)
    old_shapes = list(previous.values())
    index = STRtree(old_shapes)
    matches = []
    for current, shape in enumerate(shapes):
        for old in index.query(shape, predicate='intersects'):
            overlap = shape.intersection(old_shapes[old]).area
            if overlap > .02 and overlap >= .25 * min(shape.area, old_shapes[old].area):
                matches.append((-overlap, old_ids[old], current))
    ids = {}
    matched = set()
    for _, old_id, current in sorted(matches):
        if old_id not in matched and current not in ids:
            ids[current] = old_id
            matched.add(old_id)
    for current in range(len(shapes)):
        if current not in ids:
            ids[current] = next_id
            next_id += 1
    return {ids[i]: shape for i, shape in enumerate(shapes)}, next_id


class FloorMap:
    def __init__(self, height):
        self.height = float(height)
        self.scene = VoxelCloud()
        self.ground_frames = []
        self.clipped_wall_columns = set()
        self._wall_points = None
        self.revision = 0
        self.observed = GeometryCollection()
        self._floor_frame_count = 0
        self._region_revision = -1
        self.regions = {}
        self.next_region_id = 1
        self.opening_candidates = []
        self.selected_openings = []
        self.opening_cache = {}
        self.opening_view_samples = []
        self.doors = 0
        self.corridors = 0

    def add_view(self, view):
        self.add_views((view,))

    def add_views(self, clouds):
        self._wall_points = None
        self.revision += 1
        ground_samples = []
        for points, upper_edge, openings in clouds:
            for candidate in openings:
                add_candidate(self.opening_candidates, candidate)
            relative = points[:, 1] - self.height
            nearby_samples = points[(relative > -.4) & (relative < 1.8)][::8, [0, 2]]
            if len(nearby_samples):
                self.opening_view_samples.append(nearby_samples)
            # Retain ceiling-level measurements for the wall preview. Ordinary
            # obstacles still use the original lower height band at rendering.
            self.scene.add(points[(relative > -1.0) & (relative < 4.5)])
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
        self.ground_frames.append(endpoints)

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
            scene = self.scene.points()
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

    def update_observed_floor(self):
        new_frames = self.ground_frames[self._floor_frame_count:]
        if new_frames:
            self.observed = self.observed.union(observed_floor(new_frames))
            self._floor_frame_count = len(self.ground_frames)

    def update_regions(self, walls, obstacles):
        if self._region_revision == self.revision:
            return
        groups, self.doors, self.corridors, domain, _ = wall_region_polygons(
            walls, self.observed, self.selected_openings, obstacles)
        groups = preserve_region_splits(self.regions, groups, domain)
        groups = merge_tiny_regions(groups, domain)
        self.regions, self.next_region_id = track_regions(self.regions, groups, self.next_region_id)
        self._region_revision = self.revision

    def render(self, pixel_size, max_size):
        scene = self.scene.points()
        relative = scene[:, 1] - self.height
        obstacles = scene[(relative > .2) & (relative < 1.8)][:, [0, 2]]
        walls = self.wall_footprint(scene)
        self.update_observed_floor()
        clouds = []
        if not self.observed.is_empty:
            clouds.append(np.asarray(self.observed.bounds).reshape(2, 2))
        if len(obstacles):
            clouds.append(obstacles)
        if self.opening_candidates:
            clouds.append(np.concatenate([segment for candidate in self.opening_candidates for segment in candidate.segments]))
        if not clouds:
            return Image.new('RGB', (128, 128), 'white')

        lower = np.min([cloud.min(axis=0) for cloud in clouds], axis=0)
        upper = np.max([cloud.max(axis=0) for cloud in clouds], axis=0)
        padding = 20
        pixel_size = max(pixel_size, float((upper - lower).max()) / (max_size - 2 * padding - 2))
        image_origin = np.floor(lower / pixel_size).astype(int) - padding
        width, height = np.floor(upper / pixel_size).astype(int) - image_origin + padding + 1
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        def to_pixels(points):
            return np.floor(points / pixel_size).astype(int) - image_origin

        def polygon_mask(shape):
            mask = Image.new('L', (int(width), int(height)))
            painter = ImageDraw.Draw(mask)
            for part in polygon_parts(shape):
                exterior = to_pixels(np.asarray(part.exterior.coords))
                painter.polygon(list(map(tuple, exterior)), fill=1)
                for hole in part.interiors:
                    interior = to_pixels(np.asarray(hole.coords))
                    painter.polygon(list(map(tuple, interior)), fill=0)
            return np.asarray(mask) != 0

        def footprint(points):
            mask = np.zeros((height, width), dtype=bool)
            x, y = to_pixels(points).T
            for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                mask[y + dy, x + dx] = True
            return mask

        occupied = footprint(obstacles)
        selected = update_opening_cache(self.opening_candidates, self.opening_cache, self.opening_view_samples,
                                        occupied, image_origin, pixel_size, self.observed)
        self.opening_view_samples.clear()
        if selected is not None:
            self.selected_openings = selected
        self.update_regions(walls, obstacles)

        ground = polygon_mask(self.observed)
        image[ground] = 200

        for region_id, shape in self.regions.items():
            index = region_id - 1
            if index < len(REGION_COLORS):
                color = REGION_COLORS[index]
            else:
                rgb = colorsys.hsv_to_rgb(index * .618034 % 1, .27, .96)
                color = tuple(round(channel * 255) for channel in rgb)
            image[polygon_mask(shape) & ground] = color

        image[occupied] = (0, 0, 0)
        if len(walls):
            blocked = thick_obstacle_mask(occupied, pixel_size)
            red = extend_wall_mask(occupied, footprint(walls), pixel_size, blocked)
            image[red] = (220, 35, 35)
        canvas = Image.fromarray(image)
        painter = ImageDraw.Draw(canvas)
        for candidate in self.selected_openings:
            for segment in candidate.segments:
                painter.line(list(map(tuple, to_pixels(segment))), fill=(255, 206, 0),
                             width=max(2, round(.055 / pixel_size)))
        return canvas


class GeometryReplay:
    def __init__(self, env, sensors, args):
        self.env = env
        self.cfg = sensors.depth_sensor
        self.depth_key = getattr(self.cfg, 'uuid', 'depth')
        self.args = args
        self.floors = []
        self.active = None
        self.pending = []
        self.position = None

    def _nearest_floor(self, height):
        return min(self.floors, key=lambda floor: abs(floor.height - height))

    def _flush_pending(self):
        for position, view in self.pending:
            self._nearest_floor(position[1]).add_view(view)
        self.pending.clear()

    def _project(self, raw, agent):
        import quaternion

        sensor = agent.sensor_states[self.depth_key]
        return project_depth(raw[self.depth_key], self.cfg, np.asarray(sensor.position),
                             quaternion.as_rotation_matrix(sensor.rotation), agent.position[1])

    def fuse(self, raw):
        agent = self.env.sim.get_agent_state()
        position = np.asarray(agent.position).copy()
        self.position = position
        view = self._project(raw, agent)
        if self.active is None:
            self.active = FloorMap(position[1])
            self.floors.append(self.active)
        nearest = self._nearest_floor(position[1])
        if abs(nearest.height - position[1]) <= .22:
            self.active = nearest
            self._flush_pending()
            self.active.add_view(view)
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
            self._nearest_floor(pose[1]).add_view(cloud)

    def initialize(self, raw):
        import quaternion

        sim = self.env.sim
        state = copy.deepcopy(sim.get_agent_state())
        self.position = np.asarray(state.position).copy()
        if self.args.initial_scan == 'none':
            self.fuse(raw)
            self.active.wall_footprint()
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
            floor.add_views(panorama_clouds())
        finally:
            sim.get_agent(0).set_state(state, reset_sensors=False, infer_sensor_states=False)
            if hasattr(sim, '_prev_sim_obs'):
                sim._prev_sim_obs = previous
        self.active = floor
        self.floors.append(floor)
        floor.wall_footprint()
        print(f'[hsgm-geometry] initial panorama: {view_count} views', flush=True)

    def run(self, rows, publish):
        self.initialize(self.env.reset())
        publish(0)
        for step, row in enumerate(rows, 1):
            if self.env.episode_over:
                break
            self.fuse(self.env.step({'action': ACTION_KEYS[row['action']]}))
            if row['action'] == 0 or self.env.episode_over or step == len(rows):
                self._flush_pending()
            publish(step)
            if row['action'] == 0:
                break


def replay_latest_maps(replay, rows, output, name):
    """Publish each floor's latest frame at its fixed path while replay runs."""
    published = {}
    last_region = None

    def publish(step):
        nonlocal last_region
        for index, floor in enumerate(replay.floors, 1):
            if published.get(floor) == floor.revision:
                continue
            previous = set(floor.regions)
            target = output / f'{name}_{index:02}.png'
            temporary = target.with_suffix('.tmp')
            try:
                floor.render(replay.args.bev_pixel_size, replay.args.bev_max_size).save(temporary, format='PNG', compress_level=1)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            published[floor] = floor.revision
            new_ids = ','.join(f'R{i}' for i in sorted(set(floor.regions) - previous)) or '-'
            print(f'[wall-regions] step={step} floor={index:02} regions={len(floor.regions)} '
                  f'new={new_ids} yellow={sum(len(item.segments) for item in floor.selected_openings)} '
                  f'doors={floor.doors} short_corridors={floor.corridors} '
                  f'map={target}', flush=True)

        floor = replay.active
        index = replay.floors.index(floor) + 1
        current = None
        if abs(floor.height - replay.position[1]) <= .22:
            point = Point(replay.position[[0, 2]])
            current = next((i for i, shape in floor.regions.items() if shape.covers(point)), None)
        if last_region != (floor, current):
            last_region = floor, current
            current_name = f'R{current}' if current is not None else 'unknown'
            print(f'[wall-regions] step={step} floor={index:02} current={current_name}', flush=True)

    replay.run(rows, publish)


def main():
    parser = argparse.ArgumentParser(description='Replay Habitat actions and export wall-constrained region PNGs')
    parser.add_argument('--actions_file')
    parser.add_argument('--scene_id')
    parser.add_argument('--episode_id')
    parser.add_argument('--records_root', default=str(ROOT / 'results_base/val_unseen/streamvln_baseline_full/records'))
    parser.add_argument('--start_folder')
    parser.add_argument('--max_episodes', '--max_trajectories', type=int, default=1)
    parser.add_argument('--max_step', '--max_steps', dest='max_steps', type=int, default=100,
                        help='Maximum actions read per trajectory (default: 100, 0: unlimited)')
    parser.add_argument('--habitat_config_path', default=str(ROOT / 'config/vln_r2r.yaml'))
    parser.add_argument('--habitat_root', default='/mnt/disk8t1/zwm/Progress-Aware-VLN/habitat-lab')
    parser.add_argument('--eval_split', default='val_unseen')
    parser.add_argument('--data_path')
    parser.add_argument('--scenes_dir')
    parser.add_argument('--output_path', default='results/test_region_35')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sensor_width', type=int, default=0)
    parser.add_argument('--sensor_height', type=int, default=0)
    parser.add_argument('--initial_scan', choices=('none', 'panorama'), default='panorama')
    parser.add_argument('--bev_pixel_size', type=float, default=.025)
    parser.add_argument('--bev_max_size', type=int, default=4096)
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
            replay_latest_maps(replay, rows, output, name)
        finally:
            env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
