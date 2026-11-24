"""RGB-D geometry, portal fitting, and object fusion for the Governor."""

import heapq
import math
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label as label_components
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely import box, coverage_union_all, intersects, intersects_xy, linestrings, polygons, prepare
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union


MIN_OPENING_WIDTH = .55
MAX_OPENING_WIDTH = 2.00
MIN_SIDE_HEIGHT = .95
MIN_DEPTH_JUMP = .75
MAX_MOUTH_SKEW = math.sin(math.radians(8))


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


def corridor_footprint(pair):
    outer, inner = pair
    if np.linalg.norm(inner[0] - outer[0]) > np.linalg.norm(inner[1] - outer[0]):
        inner = inner[::-1]
    footprint = Polygon((outer[0], outer[1], inner[1], inner[0]))
    return footprint if footprint.is_valid else GeometryCollection()


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
                if one is not None and two is not None and one @ two >= .90 and min(one @ normal, two @ normal) >= .60:
                    longitudinal = (one + two) / np.linalg.norm(one + two)
            angle_error = abs(across_opening @ longitudinal) if longitudinal is not None else abs(across_opening @ normal)
            if longitudinal is not None and angle_error > MAX_MOUTH_SKEW:
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
        for segment in segments:
            across = segment[1] - segment[0]
            if abs(across @ new_axis) > MAX_MOUTH_SKEW * np.linalg.norm(across) * np.linalg.norm(new_axis):
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
        lines = [LineString(segment) for segment in segments]
        # Once both mouths are known, no additional line should cut through
        # the passage between them, including a diagonal joining its corners.
        if any(line.intersection(interior).length > .18 * line.length
               for line in lines for interior in corridors):
            continue
        if any(any(overlaps_opening(segment, old) for old in kept) or
               opening_conflicts(segment, (*kept, *segments[:i])) for i, segment in enumerate(segments)):
            continue
        kept.extend(segments)
        selected.append(candidate)
        if len(segments) == 2:
            interior = corridor_footprint(segments).buffer(-.06)
            if not interior.is_empty:
                corridors.append(interior)
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


class ObjectObservation(NamedTuple):
    label: str
    score: float
    bounds: np.ndarray
    samples: np.ndarray


def nearby_object_boxes(bounds, neighbors, same_wall=False):
    """Find touching or locally aligned boxes in floor coordinates."""
    overlap = np.minimum(bounds[2:], neighbors[:, 2:]) - np.maximum(bounds[:2], neighbors[:, :2])
    gap = np.linalg.norm(np.maximum(0., -overlap), axis=1)
    if same_wall:
        return bool((gap <= .45).any())
    min_span = np.minimum(bounds[2:] - bounds[:2], neighbors[:, 2:] - neighbors[:, :2])
    aligned = (overlap >= np.minimum(.10, .30 * min_span)).any(axis=1)
    return bool(((gap <= .06) | ((gap <= .25) & aligned)).any())


def enclose_objects(first, second):
    bounds = np.r_[np.minimum(first.bounds[:2], second.bounds[:2]),
                   np.maximum(first.bounds[2:], second.bounds[2:])]
    return first._replace(bounds=bounds, score=max(first.score, second.score))


class ObjectTrack:
    def __init__(self, observation):
        self.fragments = observation.bounds[None, :]
        self.samples = [observation.samples]
        self.estimate = observation

    def merge(self, other):
        self.fragments = np.concatenate((self.fragments, other.fragments))
        self.samples.extend(other.samples)
        self.estimate = enclose_objects(self.estimate, other.estimate)


def fuse_wall_objects(tracks):
    """Merge local fragments on a wall without merging every object along it."""
    groups = []
    for track in tracks:
        item, fragments = track.estimate, track.fragments
        separate = []
        for old, old_fragments in groups:
            if any(nearby_object_boxes(bounds, old_fragments, same_wall=True) for bounds in track.fragments):
                item = enclose_objects(item, old)
                fragments = np.concatenate((fragments, old_fragments))
            else:
                separate.append((old, old_fragments))
        groups = separate + [(item, fragments)]
    return [item for item, _ in groups]


def supporting_obstacle(samples, components, to_pixels, pixel_size):
    """Associate a detection with the obstacle supported by its projected pixels."""
    pixels = np.unique(to_pixels(np.concatenate(samples)), axis=0)
    if len(pixels) < 3:
        return 0
    reach = .10 / pixel_size
    margin = math.ceil(reach)
    left, top = np.maximum(pixels.min(axis=0) - margin, 0)
    right, bottom = np.minimum(pixels.max(axis=0) + margin + 1, components.shape[::-1])
    crop = components[top:bottom, left:right]
    if not crop.any():
        return 0
    x, y = (pixels - (left, top)).T
    hits = crop[y, x].copy()
    missing = hits == 0
    if missing.any():
        distances, nearest = distance_transform_edt(crop == 0, return_indices=True)
        close = missing & (distances[y, x] <= reach)
        hits[close] = crop[tuple(nearest[:, y[close], x[close]])]
    ids, counts = np.unique(hits[hits != 0], return_counts=True)
    if not len(ids):
        return 0
    best = int(np.argmax(counts))
    if counts[best] >= max(3, .40 * len(pixels)) and counts[best] > counts.sum() / 2:
        return int(ids[best])
    return 0


def project_objects(detections, depth, valid, project, floor_height):
    objects = []
    height, width = depth.shape
    for detection in detections:
        mask = detection['mask']
        if mask.shape != depth.shape:
            mask = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
        rows, cols = np.nonzero(mask & valid)
        if len(rows) < 12:
            continue
        # Keep the mask's main depth band, so a few background pixels can't
        # move the object across the map.
        stride = math.ceil(len(rows) / 6000)
        rows, cols = rows[::stride], cols[::stride]
        distances = depth[rows, cols]
        median = float(np.median(distances))
        inliers = np.abs(distances - median) <= max(.30, .18 * median)
        points = project(rows[inliers], cols[inliers])
        heights = points[:, 1] - floor_height
        points = points[(heights >= -.25) & (heights <= 3.5)]
        if len(points) < 12:
            continue
        xy = points[:, [0, 2]]
        bounds = np.quantile(xy, [.04, .96], axis=0).ravel()
        samples = xy[np.all((xy >= bounds[:2]) & (xy <= bounds[2:]), axis=1)]
        samples = samples[::max(1, math.ceil(len(samples) / 160))]
        objects.append(ObjectObservation(detection['label'], detection['score'], bounds, samples))
    return objects


def project_depth(depth, cfg, position, rotation, floor_height, detections=(), rgb=None):
    """Project scene geometry, pixel colors, and mask-backed objects."""
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

    rows, cols = np.nonzero(valid)
    points = project(rows, cols)
    point_colors = None
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f'Expected H×W RGB or RGBA, got {rgb.shape}')
        rgb_height, rgb_width = rgb.shape[:2]
        # Sensors share a pose and horizontal field of view. Scale both
        # pixel axes by the ratio of horizontal focal lengths.
        scale = rgb_width / w
        rgb_rows = np.floor((rows + .5 - h / 2) * scale + rgb_height / 2).astype(int)
        rgb_cols = np.floor((cols + .5 - w / 2) * scale + rgb_width / 2).astype(int)
        visible = (rgb_rows >= 0) & (rgb_rows < rgb_height)
        point_colors = np.zeros((len(points), 4), dtype=np.uint8)
        point_colors[visible, :3] = rgb[rgb_rows[visible], rgb_cols[visible], :3]
        point_colors[visible, 3] = 1
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
    objects = project_objects(detections, depth, valid, project, floor_height)
    return points, upper_edge, depth_openings(depth, valid, project, floor_height), objects, point_colors


def voxel_totals(points, size, point_colors=None):
    keys = np.floor(points / size).astype(np.int32)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    channels = points.T if point_colors is None else (*points.T, *point_colors.T)
    totals = np.column_stack([np.bincount(inverse.reshape(-1), weights=axis, minlength=len(unique))
                              for axis in channels])
    return unique, totals, counts


class VoxelCloud:
    """Incremental position and RGB averages for the occupied voxels."""

    def __init__(self, size=.025):
        self.size = size
        self.cells = {}

    def add(self, points, point_colors=None):
        if not len(points):
            return
        keys, totals, counts = voxel_totals(points, self.size, point_colors)
        if point_colors is None:
            totals = np.pad(totals, ((0, 0), (0, 4)))
        for key, total, count in zip(map(tuple, keys.tolist()), totals.tolist(), counts.tolist()):
            entry = self.cells.get(key)
            if entry is None:
                self.cells[key] = [*total, count]
            else:
                for index, value in enumerate(total):
                    entry[index] += value
                entry[7] += count

    def points(self, with_rgb=False):
        totals = np.asarray(list(self.cells.values()), dtype=np.float64).reshape(-1, 8)
        xyz = (totals[:, :3] / totals[:, 7:8]).astype(np.float32)
        if not with_rgb:
            return xyz
        rgb = np.full((len(totals), 3), np.nan, dtype=np.float32)
        np.divide(totals[:, 3:6], totals[:, 6:7], out=rgb, where=totals[:, 6:7] != 0)
        return xyz, rgb


def paint_projected_rgb(image, mask, pixels, colors, pixel_size, heights=None):
    """Paint measured colors into a ground or obstacle footprint."""
    height, width = mask.shape
    x, y = pixels.T
    valid = np.isfinite(colors[:, 0]) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    indices = np.flatnonzero(valid)
    indices = indices[mask[y[indices], x[indices]]]
    painted = np.zeros_like(mask)
    if not len(indices):
        return painted

    cells, inverse = np.unique(y[indices] * width + x[indices], return_inverse=True)
    inverse = inverse.reshape(-1)
    if heights is not None:
        levels = heights[indices]
        tops = np.full(len(cells), -np.inf)
        np.maximum.at(tops, inverse, levels)
        visible = levels >= tops[inverse] - .05
        inverse, indices = inverse[visible], indices[visible]

    counts = np.bincount(inverse, minlength=len(cells))
    means = np.column_stack([np.bincount(inverse, weights=channel, minlength=len(cells))
                             for channel in colors[indices].T]) / counts[:, None]
    source = np.column_stack(np.divmod(cells, width))
    rows, cols = source.T
    image[rows, cols] = np.rint(means).astype(np.uint8)
    painted[rows, cols] = True

    # Fill tiny gaps in the displayed surface only. Unknown space remains blank.
    missing = np.column_stack(np.nonzero(mask & ~painted))
    if len(missing):
        distance, nearest = cKDTree(source).query(missing, distance_upper_bound=.075 / pixel_size)
        close = np.isfinite(distance)
        rows, cols = missing[close].T
        image[rows, cols] = image[tuple(source[nearest[close]].T)]
        painted[rows, cols] = True
    return painted


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


NODE_GATE_DISTANCE = .75
NODE_VISIBILITY_SPACING = .025
TURN_SCAN_STEPS = 5
TURN_LOOK_DISTANCE = .45
TURN_GATE_DISTANCE = .42
MIN_TURN_DEGREES = 40.


class OpeningNode(NamedTuple):
    step: int
    point: np.ndarray
    gate: np.ndarray


class TurnNode(NamedTuple):
    step: int
    point: np.ndarray
    degrees: float


def route_turn_nodes(legs):
    """Compare the travel direction before and after each route position."""
    turns = []
    for steps, positions in legs:
        moved = np.r_[True, np.linalg.norm(np.diff(positions, axis=0), axis=1) > .012]
        steps, positions = steps[moved], positions[moved]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        pivots = np.flatnonzero((arc >= TURN_LOOK_DISTANCE - 1e-9) &
                                (arc[-1] - arc >= TURN_LOOK_DISTANCE - 1e-9))
        if not len(pivots):
            continue
        before = np.column_stack([np.interp(arc[pivots] - TURN_LOOK_DISTANCE, arc, axis) for axis in positions.T])
        after = np.column_stack([np.interp(arc[pivots] + TURN_LOOK_DISTANCE, arc, axis) for axis in positions.T])
        incoming, outgoing = positions[pivots] - before, after - positions[pivots]
        lengths = np.linalg.norm(incoming, axis=1), np.linalg.norm(outgoing, axis=1)
        supported = (lengths[0] >= .32) & (lengths[1] >= .32)
        cosines = np.ones(len(pivots))
        cosines[supported] = np.sum(incoming[supported] * outgoing[supported], axis=1) / (lengths[0][supported] * lengths[1][supported])
        angles = np.degrees(np.arccos(np.clip(cosines, -1., 1.)))
        kept = []
        for i in np.argsort(-angles, kind='stable'):
            if angles[i] < MIN_TURN_DEGREES:
                break
            pivot = pivots[i]
            if any(abs(arc[pivot] - arc[old]) < TURN_LOOK_DISTANCE for old in kept):
                continue
            kept.append(pivot)
            turns.append(TurnNode(int(steps[pivot]), positions[pivot], float(angles[i])))
    return sorted(turns, key=lambda turn: turn.step)


def opening_node_blockers(occupied, image_origin, pixel_size, gates):
    """Use the same obstacle footprint as the map around each yellow line."""
    strips = []
    margin = math.ceil(NODE_GATE_DISTANCE / pixel_size) + 1
    height, width = occupied.shape
    for gate in gates:
        pixels = np.floor(gate / pixel_size).astype(int) - image_origin
        left, top = np.maximum(pixels.min(axis=0) - margin, 0)
        right, bottom = np.minimum(pixels.max(axis=0) + margin + 1, (width, height))
        crop = occupied[top:bottom, left:right]
        changes = np.diff(np.pad(crop.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        rows, starts = np.nonzero(changes == 1)
        _, ends = np.nonzero(changes == -1)
        if len(rows):
            x = left + image_origin[0]
            z = top + rows + image_origin[1]
            strips.extend(box((x + starts) * pixel_size, z * pixel_size,
                              (x + ends) * pixel_size, (z + 1) * pixel_size))
    blockers = unary_union(strips)
    prepare(blockers)
    return blockers


def clear_view_to_gate(position, gate, blockers):
    """Find a clear ray to a point on the finite line within the node radius."""
    if intersects_xy(blockers, position[0], position[1]):
        return False
    direction = gate[1] - gate[0]
    length = np.linalg.norm(direction)
    axis = direction / length
    relative = position - gate[0]
    projection = relative @ axis
    across = np.linalg.norm(relative - projection * axis)
    if across > NODE_GATE_DISTANCE + 1e-9:
        return False
    reach = math.sqrt(max(0., NODE_GATE_DISTANCE ** 2 - across ** 2))
    start, end = max(0., projection - reach), min(length, projection + reach)
    if start > end + 1e-9:
        return False
    offsets = np.r_[np.clip(projection, start, end),
                    np.linspace(start, end, max(1, math.ceil((end - start) / NODE_VISIBILITY_SPACING) + 1))]
    targets = gate[0] + offsets[:, None] * axis
    if np.any(np.linalg.norm(targets - position, axis=1) <= 1e-9):
        return True
    rays = linestrings(np.stack((np.broadcast_to(position, targets.shape), targets), axis=1))
    return bool(np.any(~intersects(blockers, rays)))


def opening_approach_node(legs, gate, blockers, other_gates):
    """Choose a checkpoint in the last gate-bounded stretch of the approach."""
    direction = gate[1] - gate[0]
    width = np.linalg.norm(direction)
    if width <= 1e-9:
        return None
    axis = direction / width
    normal = np.array([-axis[1], axis[0]])
    visibility_blockers = blockers
    if not other_gates.is_empty:
        visibility_blockers = GeometryCollection([blockers, other_gates])
        prepare(visibility_blockers)
    for steps, positions in legs:
        relative = positions - gate[0]
        along = relative @ axis
        signed = relative @ normal
        gaps = np.hypot(signed, along - np.clip(along, 0., width))
        near = gaps <= NODE_GATE_DISTANCE + 1e-9
        edges = np.diff(np.r_[False, near, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            # Detect the actual passage across the finite opening. A change of
            # side beyond a jamb does not count as crossing the doorway.
            edges_before = np.arange(max(0, start - 1), min(stop, len(steps) - 1))
            delta = signed[edges_before] - signed[edges_before + 1]
            crosses = (signed[edges_before] * signed[edges_before + 1] <= 0.) & (np.abs(delta) > 1e-9)
            candidates = edges_before[crosses]
            fraction = signed[candidates] / (signed[candidates] - signed[candidates + 1])
            crossing = along[candidates] + fraction * (along[candidates + 1] - along[candidates])
            passage = np.flatnonzero((crossing >= 0.) & (crossing <= width))
            if len(passage):
                before = int(candidates[passage[0]])
                last = min(before, stop - 1)
                arrival = signed[before]
            else:
                last = start + int(np.argmin(gaps[start:stop]))
                arrival = signed[start]
            if abs(arrival) <= 1e-6 and start:
                arrival = signed[start - 1]
            if abs(arrival) <= 1e-6:
                continue
            first_eligible = start
            if last >= start and not other_gates.is_empty:
                approach = positions[start:last + 1]
                if len(passage):
                    contact = gate[0] + crossing[passage[0]] * axis
                    approach = np.vstack((approach, contact))
                moving = np.flatnonzero(np.linalg.norm(np.diff(approach, axis=0), axis=1) > 1e-9)
                segments = linestrings(np.stack((approach[moving], approach[moving + 1]), axis=1))
                crossed = moving[intersects(other_gates, segments)]
                if len(crossed):
                    first_eligible += int(crossed[-1]) + 1
            eligible = np.arange(first_eligible, last + 1)
            eligible = eligible[signed[eligible] * np.sign(arrival) >= .025]
            for i in eligible[np.argsort(-gaps[eligible], kind='stable')]:
                if visibility_blockers.is_empty or clear_view_to_gate(positions[i], gate, visibility_blockers):
                    return OpeningNode(int(steps[i]), positions[i], gate)
    return None


def route_opening_nodes(legs, openings, blockers):
    """Keep one pre-entry checkpoint for each finite yellow line."""
    boundaries = [LineString(gate) for gate in openings]
    nodes = [opening_approach_node(legs, gate, blockers,
                                   GeometryCollection(boundaries[:i] + boundaries[i + 1:]))
             for i, gate in enumerate(openings)]
    return sorted((node for node in nodes if node is not None), key=lambda node: node.step)


def match_opening_nodes(nodes, history):
    """Reuse IDs one-to-one when an opening is refitted or reappears."""
    matches = []
    for index, node in enumerate(nodes):
        for node_id, old in history.items():
            if overlaps_opening(node.gate, old.gate) and overlaps_opening(old.gate, node.gate):
                delta = np.linalg.norm(node.gate.mean(axis=0) - old.gate.mean(axis=0))
                matches.append((delta, node_id, index))
    assigned, used = {}, set()
    for _, node_id, index in sorted(matches):
        if index not in assigned and node_id not in used:
            assigned[index] = node_id
            used.add(node_id)
    next_id = max(history, default=0) + 1
    for index in range(len(nodes)):
        if index not in assigned:
            assigned[index] = next_id
            next_id += 1
    active = {assigned[index]: node for index, node in enumerate(nodes)}
    history.update(active)
    return active


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
        self.opening_candidates = []
        self.selected_openings = []
        self.route = []
        self.nodes = {}
        self.node_history = {}
        self.turn_history = {}
        self.turn_nodes = {}
        self.turn_scan_route_count = 0
        self.opening_cache = {}
        self.opening_view_samples = []
        self._object_tracks = {}

    @property
    def objects(self):
        return [track.estimate for tracks in self._object_tracks.values() for track in tracks]

    def add_objects(self, observations):
        for item in observations:
            track = ObjectTrack(item)
            separate = []
            for old in self._object_tracks.get(item.label, ()):
                # A new fragment may bridge several existing clusters. Test
                # against their measured boxes, never the larger drawn outline.
                if nearby_object_boxes(item.bounds, old.fragments):
                    track.merge(old)
                else:
                    separate.append(old)
            separate.append(track)
            self._object_tracks[item.label] = separate

    def objects_on_obstacles(self, occupied, walls, to_pixels, pixel_size):
        if not any(len(tracks) > 1 for tracks in self._object_tracks.values()):
            return self.objects
        solid = occupied if walls is None else occupied & ~walls
        # Use the solid interior to avoid linking furniture through a wall
        # stroke, diagonal contact, or a one-pixel bridge in the preview.
        core = square_neighborhood(solid, max(1, math.ceil(.03 / pixel_size)), require_full=True)
        components, count = label_components(core)
        wall_components, wall_count = label_components(occupied & walls) if walls is not None else (None, 0)
        if not count and not wall_count:
            return self.objects

        objects = []
        for tracks in self._object_tracks.values():
            if len(tracks) == 1:
                objects.append(tracks[0].estimate)
                continue
            fused = {}
            wall_tracks = {}
            for track in tracks:
                item = track.estimate
                wall = supporting_obstacle(track.samples, wall_components, to_pixels, pixel_size) if wall_count else 0
                if wall:
                    wall_tracks.setdefault(wall, []).append(track)
                    continue
                component = supporting_obstacle(track.samples, components, to_pixels, pixel_size) if count else 0
                if not component:
                    objects.append(item)
                elif component in fused:
                    fused[component] = enclose_objects(fused[component], item)
                else:
                    fused[component] = item
            objects.extend(fused.values())
            for attached in wall_tracks.values():
                objects.extend(fuse_wall_objects(attached))
        return objects

    def add_view(self, view):
        self.add_views((view,))

    def add_views(self, clouds):
        self._wall_points = None
        self.revision += 1
        ground_samples = []
        for points, upper_edge, openings, objects, point_colors in clouds:
            self.add_objects(objects)
            for candidate in openings:
                add_candidate(self.opening_candidates, candidate)
            relative = points[:, 1] - self.height
            nearby_samples = points[(relative > -.4) & (relative < 1.8)][::8, [0, 2]]
            if len(nearby_samples):
                self.opening_view_samples.append(nearby_samples)
            # Keep high surfaces for wall-aware object association.
            retained = (relative > -1.0) & (relative < 4.5)
            self.scene.add(points[retained], None if point_colors is None else point_colors[retained])
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

    def render(self, pixel_size, max_size, scan_turns=False):
        """Return the geometry image, its coordinate transform, and fused proposals."""
        scene, scene_rgb = self.scene.points(with_rgb=True)
        relative = scene[:, 1] - self.height
        obstacle_points = (relative > .2) & (relative < 1.8)
        obstacles = scene[obstacle_points][:, [0, 2]]
        self.update_observed_floor()
        clouds = []
        if not self.observed.is_empty:
            clouds.append(np.asarray(self.observed.bounds).reshape(2, 2))
        if len(obstacles):
            clouds.append(obstacles)
        legs = []
        if self.route:
            steps = np.asarray([step for step, _ in self.route])
            points = np.stack([point for _, point in self.route])
            breaks = np.flatnonzero((np.diff(steps) != 1) | (np.linalg.norm(np.diff(points, axis=0), axis=1) > .55)) + 1
            legs = list(zip(np.split(steps, breaks), np.split(points, breaks)))
            clouds.append(points)
        if self.opening_candidates:
            clouds.append(np.concatenate([segment for candidate in self.opening_candidates for segment in candidate.segments]))
        objects = self.objects
        if objects:
            clouds.append(np.stack([item.bounds.reshape(2, 2) for item in objects]).reshape(-1, 2))
        if not clouds:
            clouds.append(np.zeros((1, 2)))

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

        def footprint(points, dilate=True):
            mask = np.zeros((height, width), dtype=bool)
            x, y = to_pixels(points).T
            mask[y, x] = True
            if dilate:
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    mask[y + dy, x + dx] = True
            return mask

        occupied = footprint(obstacles)
        selected = update_opening_cache(self.opening_candidates, self.opening_cache, self.opening_view_samples,
                                        occupied, image_origin, pixel_size, self.observed)
        self.opening_view_samples.clear()
        if selected is not None:
            self.selected_openings = selected
        ground = polygon_mask(self.observed)
        image[ground] = 200
        ground_points = (relative > -.4) & (relative <= .2)
        paint_projected_rgb(image, ground, to_pixels(scene[ground_points][:, [0, 2]]),
                            scene_rgb[ground_points], pixel_size)

        image[occupied] = (0, 0, 0)
        paint_projected_rgb(image, occupied, to_pixels(obstacles), scene_rgb[obstacle_points],
                            pixel_size, heights=relative[obstacle_points])
        wall_mask = None
        walls = self.wall_footprint(scene)
        if len(walls):
            blocked = thick_obstacle_mask(occupied, pixel_size)
            wall_mask = extend_wall_mask(occupied, footprint(walls), pixel_size, blocked)
            image[wall_mask] = (220, 35, 35)
        canvas = Image.fromarray(image)
        painter = ImageDraw.Draw(canvas)
        for _, points in legs:
            if len(points) > 1:
                painter.line(list(map(tuple, to_pixels(points))), fill=(37, 92, 205),
                             width=max(2, round(.040 / pixel_size)))
        gates = [segment for candidate in self.selected_openings for segment in candidate.segments]
        blockers = (opening_node_blockers(footprint(obstacles, dilate=False), image_origin, pixel_size, gates)
                    if legs and gates else GeometryCollection())
        self.nodes = match_opening_nodes(route_opening_nodes(legs, gates, blockers), self.node_history)
        gate_nodes = [event.point for event in self.nodes.values()]
        if scan_turns and len(self.route) != self.turn_scan_route_count:
            for turn in route_turn_nodes(legs):
                if any(np.linalg.norm(turn.point - point) <= TURN_GATE_DISTANCE for point in gate_nodes):
                    continue
                if any(np.linalg.norm(turn.point - old.point) <= .26 for old in self.turn_history.values()):
                    continue
                self.turn_history[max(self.turn_history, default=0) + 1] = turn
            self.turn_scan_route_count = len(self.route)
        self.turn_nodes = {index: turn for index, turn in self.turn_history.items()
                           if all(np.linalg.norm(turn.point - point) > TURN_GATE_DISTANCE for point in gate_nodes)}
        for candidate in self.selected_openings:
            for segment in candidate.segments:
                painter.line(list(map(tuple, to_pixels(segment))), fill=(255, 206, 0),
                             width=max(2, round(.055 / pixel_size)))
        if objects:
            objects = self.objects_on_obstacles(occupied, wall_mask, to_pixels, pixel_size)
        return canvas, to_pixels, objects
