"""Run the StreamVLN Governor with an unpartitioned RGB-D map.

Keep this file, governor_topdown_map_nodes.py, rgb_mapping.py and room_similarity.py together.
"""

import copy
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import ImageDraw

import governor_topdown_map_nodes as base
import rgb_mapping as rgb
from room_similarity import PEClient, TrajectoryRooms, atomic_json

REVIEW_INTERVAL = 16
_ROOM_CLIENT = None

# User-provided category vocabulary. These are candidate TYPE names, not detected
# rooms, instruction target IDs, or proof that a target area has been entered.
ROOM_TYPE_VOCABULARY = (
    'aisle', 'alcove', 'antechamber', 'art room', 'balcony', 'ballroom', 'bar',
    'basement', 'basketball court', 'bathroom', 'bedroom', 'breezeway', 'bridal room',
    'coat room', 'conference room', 'corridor', 'deck', 'den', 'dining room',
    'display room', 'elevator', 'entryway', 'exercise room', 'exit', 'floor', 'fork',
    'foyer', 'garage', 'garden room', 'great room', 'hall', 'intersection', 'kitchen',
    'landing', 'laundry room', 'library', 'living room', 'lobby', 'locker room',
    'massage room', 'office', 'pantry', 'path', 'patio', 'recreation room', 'salon',
    'sanctuary', 'sauna', 'shop', 'staircase', 'step', 'suite', 'theater room',
    'wine room', 'work room', 'workspace',
)

RGB_REVIEW = """Use the original instruction, the latest RGB bird's-eye map, the supplied region
target descriptions, the numbered object proposals and the retained trajectory. First review each
proposed object from its RGB appearance, size, shape and location. Use confirmed for supported
identity and relevant modifiers, rejected for a clear mismatch, and uncertain for inadequate
visibility or identity. Use only objects you confirm in this response as evidence for the verdicts.
The map has no room
polygons or room labels. Infer the current area, if possible, from RGB surfaces, furniture layout,
walls, yellow openings and the actual route. Red wall marks are thin; magenta traces are withdrawn
or rollback motion. Blue is the retained route and current heading. White outside the map may be
unobserved. Green boxes carry numeric object IDs whose identities are supplied separately.
Region IDs are instruction target IDs, not map polygons. Use null when the current area is
unidentifiable or does not correspond to a listed area.
room_type_vocabulary contains possible generic area TYPES, not observed facts. Return
current_region_type as one exact supplied vocabulary entry, or null if map evidence is inadequate.
Never infer a type merely because it appears in the list. A type is distinct from an instruction's
specific region target: multiple places may share one type. Match an instruction-specific region
by checking its modifiers against map-visible spatial relations, walls/openings and currently
confirmed numbered objects. Use the retained trajectory to check actual entry and visit order;
early visibility of a place or object is not entry. A generic type alone cannot establish target
identity, route correctness or permission to stop. Unlisted or ambiguous areas may remain null.
Keep modified region descriptions in the instruction target list only. No extra map label,
polygon, instruction-specific region annotation, or precomputed modifier match is provided or
required. Interpret those descriptions during this review using the map and confirmed DINO/SAM
object positions. Instruction target IDs identify task references only, never physical room labels.
Do not require the map to contain the full instruction phrase as a label before it can match.
Conversely, a matching category or object name alone is insufficient: verify the relevant spatial
relation and the robot's actual location from the map. If a required relation is invisible or
unresolved, keep the match uncertain instead of inventing a label or inferring it from route intent.
Only ten room_type_vocabulary entries were selected from the original instruction. Route segments
are numbered S1, S2, ... on the map; route_segments lists their start/end nodes and steps, and the
PE argmax over mean similarities of four uniformly sampled camera frames to those ten types.
This is a CLOSED candidate set: PE always has a largest score, even for an unlisted space.
Cosine similarity is not a probability or proof of category. Check margin, per-frame disagreement,
mixed_views and provisional fewer-than-four status against the map and verified object layout.
Segment types are generic hypotheses, not instruction-specific region labels or guaranteed facts.
Infer temporal order from the numbered retained route and geometry. Raw sampled camera images
are used ONLY by PE and are not supplied to you; do not claim to have seen them. Do not use any
action history. Modifiers and specific region identity still require map-grounded reasoning.
Explain the physical evidence and any
uncertainty. Seeing a destination is not entering it. Use the order of actual spatial visits and
substantial turns, not heading alone. Declare off_route only for a supported spatial or order
violation. Otherwise use normal or uncertain. Give a route verdict only if Route review is requested.
Give a stop verdict only if STOP review is requested. An unrequested verdict is uncertain.
For STOP, decide whether stopping here now would satisfy the required earlier route and endpoint,
even if a separate route verdict was not requested. A request
to wait here can be fulfilled by stopping now. Use continue for an established unmet requirement
and uncertain when the map cannot decide.
Recovery may select only a supplied retained checkpoint. Prefer the most recent suitable one
whose semantic route position is supported. Use needed=false and node_id=null when unsupported.
If travel must continue and a supplied P waypoint is useful, include its ID in recovery.point_id.
Use null otherwise. Never invent a checkpoint or waypoint.
Evidence must cite existing region target IDs, verified numeric object IDs, and/or physical
route_steps. A listed name alone proves nothing. Cite route steps for an off-route judgment or
a STOP decision. Return one JSON object of this form:
{"objects":[{"id":1,"status":"confirmed|rejected|uncertain","evidence":"specific visual reason"}],
"route_state":"normal|off_route|uncertain","current_region_id":null,"current_region_type":null,
"region_evidence":"RGB or spatial evidence",
"position_review":{"stage":"...","completed":[],"completion_status":"...","location":"...",
"history_match":"...","reason":"..."},
"stop_proposal":{"decision":"stop|continue|uncertain","timing":"early|late|on_time|uncertain","reason":"..."},
"map_evidence":{"route":{"sufficient":false,"region_ids":[],"object_ids":[],"route_steps":[],"relation":"..."},
"stop":{"sufficient":false,"region_ids":[],"object_ids":[],"route_steps":[],"relation":"..."}},
"recovery":{"needed":false,"node_id":null,"point_id":null,"current_node_suitable":false,"reason":"..."}}.
Return an empty objects list when no proposals were supplied."""

RGB_ORIENT = """Choose the initial direction for the original instruction using the supplied
24-sector panorama and the verified RGB map. Sectors are clockwise in 15-degree steps:
0=front, 6=right, 12=back, 18=left. Choose a heading for the first required traversal.
Return {"sector":0,"reason":"...","initial_description":{"area":"...","front":"...",
"right":"...","back":"...","left":"..."}}. Use sector=null if the direction is uncertain.
The description is an initial observation snapshot and does not establish route progress."""


class GovernorRGBMap(base._MetricMap):
    def __init__(self, *args):
        super().__init__(*args)
        self.rgb_floors = {}
        self.rgb_views = {}
        self.object_targets = {}
        self.object_records = {}
        self.live_objects = {}
        self.graph = None
        self.room_tracker = None

    def candidates(self, grid):
        # The legacy motion controller still uses its safety grid for A*.
        grid['regions'] = {}
        return grid['regions']

    def update(self, obs, detections=()):
        if (obs.step, obs.view) in self.seen:
            return
        super().update(obs)
        if self.active is None or self.transition:
            return
        floor = self.active
        if floor not in self.rgb_floors:
            self.rgb_floors[floor] = rgb.FloorMap(self.floors[floor])
        layer = self.rgb_floors[floor]
        layer.height = self.floors[floor]
        camera = SimpleNamespace(hfov=obs.hfov, min_depth=obs.min_depth, max_depth=obs.max_depth,
                                 normalize_depth=obs.normalized)
        detections = [dict(item, label=item['target_id']) for item in detections
                      if item.get('target_id') in self.object_targets]
        layer.add_view(rgb.project_depth(obs.depth, camera, obs.camera_position, obs.camera_rotation,
                                         layer.height, detections, rgb=obs.rgb))

    def truncate_route(self, physical_step):
        super().truncate_route(physical_step)
        if self.room_tracker is not None:
            self.room_tracker.truncate(physical_step)
        for layer in self.rgb_floors.values():
            layer.turn_history = {i: event for i, event in layer.turn_history.items()
                                  if event.step <= physical_step}
            layer.turn_scan_route_count = 0

    def _sync_objects(self, floor, projected):
        old_ids = set(self.live_objects.get(floor, ()))
        assignments = []
        for index, item in enumerate(projected):
            for oid in old_ids:
                previous = self.object_records[oid]
                bounds = np.asarray(previous['bounds'])
                if previous['target_id'] == item.label and rgb.nearby_object_boxes(item.bounds, bounds[None, :]):
                    delta = np.linalg.norm((item.bounds[:2] + item.bounds[2:] - bounds[:2] - bounds[2:]) / 2)
                    assignments.append((previous['status'] == 'rejected', delta, oid, index))
        matched, used = {}, set()
        for _, _, oid, index in sorted(assignments):
            if index not in matched and oid not in used:
                matched[index] = oid
                used.add(oid)
        current = []
        for index, item in enumerate(projected):
            oid = matched.get(index)
            if oid is None:
                oid = self.next_object
                self.next_object += 1
            target = self.object_targets[item.label]
            previous = self.object_records.get(oid)
            status = 'candidate' if previous is None else previous['status']
            if status == 'confirmed' and np.max(np.abs(item.bounds - previous['bounds'])) > .15:
                status = 'candidate'
            x, z = (item.bounds[:2] + item.bounds[2:]) / 2
            self.object_records[oid] = dict(id=oid, target_id=item.label, label=target['name'],
                                            floor=floor, bounds=item.bounds.tolist(),
                                            center=[float(x), self.floors[floor], float(z)],
                                            score=round(item.score, 3), status=status)
            current.append(oid)
        self.live_objects[floor] = current
        self._sync_verified()

    def _sync_verified(self):
        self.objects = {oid: self.object_records[oid] for ids in self.live_objects.values() for oid in ids
                        if self.object_records[oid]['status'] == 'confirmed'}

    def floor_view(self):
        floor = self.active
        cached = self.rgb_views.get(floor)
        if cached is not None and cached[0] == self.version:
            return cached[1:]
        if floor not in self.rgb_floors:
            self.rgb_floors[floor] = rgb.FloorMap(self.floors[floor])
        layer = self.rgb_floors[floor]
        layer.route = [(row['step'], np.asarray(row['position'])[[0, 2]])
                       for row in self.route if row['floor'] == floor]
        if not layer.route:
            layer.route = [(0, self.position[[0, 2]])]
        scan_turns = layer.route[-1][0] % rgb.TURN_SCAN_STEPS == 0
        image, transform, objects = layer.render(.025, 1600, scan_turns)
        self._sync_objects(floor, objects)
        self.rgb_views[floor] = (self.version, image, transform, layer)
        return image, transform, layer

    def object_list(self, include_candidates=False):
        return [self.object_records[oid] for oid in self.live_objects.get(self.active, ())
                if self.object_records[oid]['status'] == 'confirmed' or
                (include_candidates and self.object_records[oid]['status'] != 'rejected')]

    def apply_object_review(self, reviews):
        changed = False
        for oid, status in reviews.items():
            item = self.object_records[oid]
            changed |= status != item['status']
            item['status'] = status
        self._sync_verified()
        if changed:
            self._saved_snapshot = None

    def metadata(self, grid):
        _, _, layer = self.floor_view()
        nodes = [] if self.graph is None else [node for node in self.graph.nodes.values()
                                                 if node['floor'] == self.active and node.get('retained', True)]
        return dict(version=self.version, floor=self.active, position=self.position.tolist(),
                    forward=self.forward.tolist(), on_stairs_or_unconfirmed_level=self.transition,
                    retained_route=self.route, red_routes=self.error_routes, decision_nodes=nodes,
                    route_segments=[] if self.room_tracker is None else self.room_tracker.segments,
                    opening_segments_xz=[segment.tolist() for item in layer.selected_openings
                                         for segment in item.segments],
                    object_list=self.object_list())

    def render(self, grid=None, points=(), target=None, show_agent=True, *, include_candidates=False):
        image, to_pixels, _ = self.floor_view()
        image = image.copy()
        draw = ImageDraw.Draw(image)

        def pixel(position):
            return tuple(to_pixels(np.asarray(position)[[0, 2]]))

        def marker(position, label, color):
            x, y = pixel(position)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline='white')
            draw.text((x + 5, y - 10), label, fill=color, stroke_width=1, stroke_fill='white')

        for route in self.error_routes:
            for first, second in zip(route['points'], route['points'][1:]):
                if first['floor'] == second['floor'] == self.active:
                    draw.line([pixel(first['position']), pixel(second['position'])], fill='#d33586', width=2)
        if self.graph is not None:
            for nid, node in self.graph.nodes.items():
                if node['floor'] == self.active and node.get('retained', True):
                    marker(node['position'], nid, '#6a25a0')
        if self.room_tracker is not None:
            for segment in self.room_tracker.segments:
                if segment['floor'] == self.active:
                    x, y = pixel(segment['label_position'])
                    draw.text((x + 7, y + 7), 'S%d' % segment['id'], fill='#1751e5',
                              stroke_width=2, stroke_fill='white')
        for item in self.object_list(include_candidates):
            lower, upper = to_pixels(np.asarray(item['bounds']).reshape(2, 2))
            label = str(item['id'])
            left, top, right, bottom = draw.textbbox((0, 0), label, stroke_width=1)
            center = (lower + upper) // 2
            half_size = np.array([right - left + 4, bottom - top + 4]) // 2
            lower, upper = np.minimum(lower - 2, center - half_size), np.maximum(upper + 2, center + half_size)
            draw.rectangle((*lower, *upper), outline='#00754e', width=2)
            draw.text(tuple(center - np.array([(right + left) // 2, (bottom + top) // 2])),
                      label, fill='#003f2a', stroke_width=1, stroke_fill='white')
        for point in points:
            marker(point['world'], point['id'], '#0b863b')
        if target is not None:
            marker(target, '', '#0b863b')
        if show_agent and not self.transition:
            origin = pixel(self.position)
            end = pixel(self.position + .35 * self.forward)
            draw.line([origin, end], fill='#1751e5', width=3)
            marker(self.position, '', '#1751e5')
        return image

    def save(self, folder, ordered=False, force=False):
        self.floor_view()
        super().save(folder, ordered=ordered, force=force)
        if self.room_tracker is not None:
            self.room_tracker.save(folder)

    def save_final(self, folder, termination=''):
        if self.room_tracker is not None:
            self.room_tracker.refresh(self.route, [] if self.graph is None else self.graph.nodes.values())
            self.room_tracker.save(folder)
        super().save_final(folder, termination)
        if self.graph is not None:
            final = Path(folder) / 'final_map.json'
            value = json.loads(final.read_text(encoding='utf-8'))
            value.update(decision_nodes=list(self.graph.nodes.values()),
                         decision_edges=sorted(self.graph.edges))
            temporary = final.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, final)
        path = Path(folder) / 'object_inventory.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(list(self.objects.values()), ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)


class RGBDecisionNodes(base.DecisionNodes):
    def __init__(self):
        super().__init__()
        self.geometry_nodes = {}

    def observe(self, episode):
        mapper = episode.map
        if not episode.pose.valid or mapper.transition:
            return None
        _, _, layer = mapper.floor_view()
        states = {frame.physical_step: i for i, frame in enumerate(episode.history)}
        route = {row['step']: row['position'] for row in mapper.route if row['floor'] == mapper.active}
        proposals = [('opening', i, event) for i, event in layer.nodes.items()]
        proposals.extend(('turn', i, event) for i, event in layer.turn_nodes.items())
        current_keys = {(mapper.active, kind, index) for kind, index, _ in proposals}
        for key, nid in self.geometry_nodes.items():
            if key[0] == mapper.active and key not in current_keys:
                self.nodes[nid]['retained'] = False
        created = None
        for kind, index, event in sorted(proposals, key=lambda proposal: proposal[2].step):
            step = event.step
            if step not in states or step not in route or step not in episode.map_poses:
                continue
            key = (mapper.active, kind, index)
            known_id = self.geometry_nodes.get(key)
            previous = self.nodes.get(known_id)
            if previous and previous['retained']:
                nid = known_id
                if kind == 'opening':
                    previous['opening_segment_xz'] = event.gate.tolist()
                if previous['step'] == step:
                    continue
            else:
                nid = f'N{len(self.nodes) + 1}'
            node = dict(id=nid, floor=mapper.active, kind=kind, branches=0, retained=True,
                        route_state='uncertain', state=states[step], step=step, position=route[step])
            if kind == 'opening':
                node['opening_segment_xz'] = event.gate.tolist()
            else:
                node['turn_angle_deg'] = event.degrees
            self.nodes[nid] = node
            self.geometry_nodes[key] = nid
            episode.node_poses[nid] = episode.map_poses[step].copy()
            episode.event('decision_node', node=copy.deepcopy(node))
            created = nid
        retained = sorted((node for node in self.nodes.values() if node['retained']),
                          key=lambda node: node['step'])
        self.edges = {(first['id'], second['id']) for first, second in zip(retained, retained[1:])}
        on_floor = [node for node in retained if node['floor'] == mapper.active]
        self.active = self.last = on_floor[-1]['id'] if on_floor else None
        return created


class RGBPlanner(base.NodePlanner):
    def ask(self, prompt, data, images=(), validator=None, **kwargs):
        if prompt in (base.TARGET_PROMPT, base.TARGET_AUDIT_PROMPT):
            original = validator
            def validate(value):
                value = original(value) if original else value
                ids = value.get('room_top10')
                if (not isinstance(ids, list) or len(ids) != 10 or
                        any(type(i) is not int or not 0 <= i < len(ROOM_TYPE_VOCABULARY) for i in ids)
                        or len(set(ids)) != 10):
                    raise ValueError('room_top10 must contain exactly ten distinct supplied integer IDs')
                return value
            prompt += ('\nAlso return room_top10: an array of exactly TEN distinct integer IDs from '
                       'the supplied room vocabulary, ranked by relevance to the ENTIRE original '
                       'instruction. Include its explicitly named room types first, then plausible '
                       'related or intervening types. Selection is a textual hypothesis, never a '
                       'claim that any room has been observed. Do not rename the supplied categories. '
                       'Keep the existing regions/stuffs/clause extraction schema as well.')
            data += '\nROOM VOCABULARY: ' + json.dumps(dict(enumerate(ROOM_TYPE_VOCABULARY)))
            result = super().ask(prompt, data, images, validator=validate, **kwargs)
            self.room_top10 = [ROOM_TYPE_VOCABULARY[i] for i in result['room_top10']]
            return result
        return super().ask(prompt, data, images, validator=validator, **kwargs)

    def extract(self):
        targets = super().extract()
        self.map.object_targets = {row['id']: row for row in targets if row['kind'] == 'object'}
        # Use the final audited selection (or the validated draft if audit failed).
        self.room_top10 = [ROOM_TYPE_VOCABULARY[i] for i in self.inventory['room_top10']]
        self.map.room_tracker.set_categories(self.room_top10)
        atomic_json(self.map.room_output / 'room_top10.json', dict(
            instruction=self.instruction, categories=self.room_top10, source='instruction_target_extraction'))
        self.log(dict(phase='room_top10_selected', categories=self.room_top10))
        return targets

    def review(self, history, *, route_requested, stop_requested):
        step = history[-1].physical_step
        mode = ('periodic_and_native_stop' if route_requested and stop_requested else
                'periodic' if route_requested else 'native_stop')
        fallback = dict(route_state='uncertain', current_region_id=None, current_region_type=None,
                        region_evidence='Unavailable',
                        position_review=dict(reason='No usable map review'),
                        stop_proposal=dict(decision='uncertain', timing='uncertain', reason='No usable map review'),
                        recovery=dict(needed=False, node_id=None, point_id=None),
                        map_evidence={})
        candidates = {}
        waypoints = {}
        if self.pose.valid and not self.map.transition:
            self.map.floor_view()
            candidates = {item['id']: item for item in self.map.object_list(include_candidates=True)}
            regions = {item['id']: item for item in self.targets if item['kind'] == 'region'}
            grid = self.map.grid()
            points = self.map.reachable_points(grid, self.args.point_max_distance)
            waypoints = {item['id']: item for item in points}
            nodes = {node['id'] for node in self.node_candidates}
            route_steps = {row['step'] for row in self.map.route if row['floor'] is not None}
            hypotheses = [dict(item, query=self.map.object_targets[item['target_id']]['query'],
                               modifiers=self.map.object_targets[item['target_id']]['modifiers'])
                          for item in candidates.values()]
            map_metadata = self.map.metadata(grid)
            map_metadata.pop('route_segments', None)
            data = dict(physical_step=step, route_review_requested=route_requested,
                        stop_review_requested=stop_requested, region_targets=list(regions.values()),
                        room_type_vocabulary=self.room_top10,
                        route_segments=self.map.room_tracker.evidence(),
                        object_targets=list(self.map.object_targets.values()), object_proposals=hypotheses,
                        map=map_metadata, rollback_nodes=self.node_candidates,
                        waypoints=points, earlier_reviews=self.summaries)
            messages = [
                {'role': 'system', 'content': RGB_REVIEW},
                base.pack_message('ORIGINAL GLOBAL INSTRUCTION:\n' + self.instruction
                                  + '\nREVIEW INPUT:\n' + json.dumps(data, ensure_ascii=False)),
                base.pack_message('LATEST MAP with numbered object proposals and P waypoints',
                                  self.map.render(grid, points, include_candidates=True)),
            ]
            self.log(dict(phase='navigation_evidence', mode=mode, physical_step=step, source='map_only'))
            try:
                raw = self.client.generate(messages, self.args.review_max_new_tokens)
                result = self.validate_review(base.json_object(raw), candidates, regions, route_steps,
                                              nodes, waypoints, route_requested, stop_requested,
                                              room_types=self.room_top10)
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError) as exc:
                self.map.apply_object_review({oid: 'uncertain' for oid in candidates})
                self.log(dict(phase='rgb_review_abstained', mode=mode, physical_step=step, error=str(exc)))
                result = fallback
            else:
                verdicts = result['objects']
                self.map.apply_object_review({row['id']: row['status'] for row in verdicts})
                self.log(dict(phase='rgb_object_review', floor=self.map.active, verdicts=verdicts))
                point_id = result['recovery'].get('point_id')
                if point_id is not None:
                    result['recovery']['waypoint'] = waypoints[point_id]['world']
        else:
            fallback['position_review']['reason'] = 'Map localization unavailable'
            fallback['stop_proposal']['reason'] = 'Map localization unavailable'
            result = fallback
        result.update(route_checked=route_requested, stop_checked=stop_requested,
                      route_check=dict(off_route=result['route_state'] == 'off_route',
                                       reason=result['position_review']['reason']))
        result['stop_proposal']['reject_vln_stop'] = result['stop_proposal']['decision'] == 'continue'
        self.position_reviews.append(dict(state=len(history)-1, observed_at=step, source=mode,
                                          route_state=result['route_state'], **result['position_review']))
        return result

    @staticmethod
    def validate_review(result, candidates, regions, route_steps, nodes, waypoints, route_requested, stop_requested,
                        room_types=None):
        rows = result.get('objects')
        if not isinstance(rows, list):
            raise ValueError('Return an object verdict list')
        confirmed, seen = set(), set()
        for row in rows:
            if (not isinstance(row, dict) or type(row.get('id')) is not int or row['id'] not in candidates
                    or row['id'] in seen or row.get('status') not in ('confirmed', 'rejected', 'uncertain')
                    or not isinstance(row.get('evidence'), str) or not row['evidence'].strip()):
                raise ValueError('Use supplied numeric IDs and give a visual reason for each verdict')
            seen.add(row['id'])
            if row['status'] == 'confirmed':
                confirmed.add(row['id'])
        if seen != set(candidates):
            raise ValueError('Review every supplied object')
        region = result.get('current_region_id')
        if region is not None and (not isinstance(region, str) or region not in regions):
            raise ValueError('Use a supplied region target ID or null')
        # Invalid optional type names cannot invalidate an otherwise usable review
        # or become new categories. Instruction-region IDs retain their own checks.
        region_type = result.get('current_region_type')
        allowed_types = ROOM_TYPE_VOCABULARY if room_types is None else room_types
        if not isinstance(region_type, str) or region_type not in allowed_types:
            region_type = None
        result['current_region_type'] = region_type
        reason = result.get('region_evidence')
        position = result.get('position_review')
        if not isinstance(reason, str) or not isinstance(position, dict) or any(
                not isinstance(position.get(key), str) for key in
                ('stage', 'completion_status', 'location', 'history_match', 'reason')):
            raise ValueError('Provide the region evidence and current position review')
        completed = position.get('completed')
        if not isinstance(completed, list) or any(not isinstance(item, str) for item in completed):
            raise ValueError('position_review.completed must be a string list')
        position.update(current_region_id=region, current_region_type=region_type,
                        region_evidence=reason)
        evidence = result.get('map_evidence')
        if not isinstance(evidence, dict):
            raise ValueError('Provide map_evidence')

        def supported(kind):
            row = evidence.get(kind)
            if not isinstance(row, dict) or type(row.get('sufficient')) is not bool or not isinstance(
                    row.get('relation'), str) or not row['relation'].strip():
                raise ValueError('Provide the map evidence for ' + kind)
            for name, typ, allowed in (('region_ids', str, regions), ('object_ids', int, confirmed),
                                        ('route_steps', int, route_steps)):
                refs = row.get(name)
                if not isinstance(refs, list) or any(type(ref) is not typ or ref not in allowed for ref in refs):
                    raise ValueError('Use existing evidence IDs and only currently confirmed objects')
            return row['sufficient'] and any(row[name] for name in ('region_ids', 'object_ids', 'route_steps'))

        for kind, requested in (('route', route_requested), ('stop', stop_requested)):
            if not requested:
                evidence[kind] = dict(sufficient=False, region_ids=[], object_ids=[], route_steps=[],
                                      relation='Not requested')
        route = result.get('route_state') if route_requested else 'uncertain'
        if route not in ('normal', 'off_route', 'uncertain'):
            raise ValueError('Invalid route state')
        if route_requested and (not supported('route') or route == 'off_route' and not evidence['route']['route_steps']):
            route = 'uncertain'
        result['route_state'] = route
        if not stop_requested:
            result['stop_proposal'] = dict(decision='uncertain', timing='uncertain', reason='STOP not requested')
        else:
            stop = result.get('stop_proposal')
            if not isinstance(stop, dict) or stop.get('decision') not in ('stop', 'continue', 'uncertain') or (
                    stop.get('timing') not in ('early', 'late', 'on_time', 'uncertain')) or not isinstance(stop.get('reason'), str):
                raise ValueError('Invalid STOP verdict')
            if (not supported('stop') or not evidence['stop']['route_steps']
                    or stop['decision'] == 'stop' and stop['timing'] != 'on_time'
                    or stop['decision'] == 'stop' and route_requested and route != 'normal'):
                stop.update(decision='uncertain', timing='uncertain')
        recovery = result.get('recovery')
        if not isinstance(recovery, dict) or type(recovery.get('needed')) is not bool:
            raise ValueError('Provide recovery.needed')
        node, point = recovery.get('node_id'), recovery.get('point_id')
        if node is not None and (not isinstance(node, str) or node not in nodes):
            raise ValueError('Use a supplied reachable rollback node ID or null')
        if recovery['needed'] and node is None:
            raise ValueError('Recovery requires a reachable node ID')
        if point is not None and (not isinstance(point, str) or point not in waypoints):
            raise ValueError('Use a supplied waypoint ID or null')
        result['recovery'] = dict(needed=recovery['needed'], node_id=node, point_id=point)
        return result

    def orient(self, panorama):
        data = 'Targets: ' + json.dumps(self.targets, ensure_ascii=False)
        data += '\nVerified object list: ' + json.dumps(self.map.object_list(), ensure_ascii=False)

        def validate(value):
            sector = value.get('sector')
            if 'sector' not in value or (sector is not None and (type(sector) is not int or not 0 <= sector < 24)):
                raise ValueError('sector must be 0..23 or null')
            description = value.get('initial_description')
            if not isinstance(description, dict) or any(
                    not isinstance(description.get(key), str) for key in ('area', 'front', 'right', 'back', 'left')):
                raise ValueError('Include the initial area and four directional descriptions')
            return value

        reply = base.HistoryPlanner.ask(self, RGB_ORIENT, data,
            [('INITIAL PANORAMA', panorama), ('LATEST MAP with verified objects', self.map.render())],
            validator=validate)
        self.initial_location = reply['initial_description']
        self.log(dict(phase='panorama_orientation', **reply))
        return reply['sector']


class RGBEpisode(base.NodeEpisode):
    def __init__(self, *args):
        global _ROOM_CLIENT
        super().__init__(*args)
        self.graph = RGBDecisionNodes()
        self.map.graph = self.graph
        if _ROOM_CLIENT is None:
            _ROOM_CLIENT = PEClient(self.args, ROOM_TYPE_VOCABULARY)
            print('[room-pe] ready: PE-Core-L14-336; text cached; vision device=' +
                  self.args.room_device, flush=True)
        self.room_tracker = TrajectoryRooms(_ROOM_CLIENT)
        self.map.room_tracker = self.room_tracker
        self.map.room_output = Path(self.folder)

    def initialize_checkpoint(self):
        self.graph.checkpoint(self, 'start')

    def candidates_for_review(self):
        return sorted((dict(node, return_method='inverse_executed_actions')
                       for node in self.graph.nodes.values()
                       if node.get('retained', True) and node['floor'] == self.map.active
                       and node['state'] < len(self.history)), key=lambda node: node['step'], reverse=True)

    def fuse(self, raw, view='front', yaw=0., refresh=False):
        if not self.pose.valid:
            return
        obs = self.observation(raw, view, yaw)
        if refresh:
            self.map.seen.discard((obs.step, obs.view))
        if (obs.step, obs.view) in self.map.seen:
            return
        targets = [dict(target, name=target['query']) for target in self.vlm.targets if target['kind'] == 'object']
        detections = self.segmenter.predict(obs.rgb, targets) if targets else []
        self.map.update(obs, detections)
        if (view == 'front' and self.map.route and self.map.route[-1]['step'] == obs.step
                and self.map.route[-1]['floor'] == self.map.active):
            self.room_tracker.record(obs.step, self.map.active, self.map.position, obs.rgb)
        self.event('map_observation', view=view, instances=len(detections), map_version=self.map.version,
                   floor=self.map.active, transition=self.map.transition, projection=self.map.observation_stats[view])

    def scheduled_review(self, route_requested, stop_requested):
        self.room_tracker.refresh(self.map.route, self.graph.nodes.values())
        self.room_tracker.save(self.folder)
        self.vlm.node_candidates = self.candidates_for_review()
        review = self.vlm.review(self.history, route_requested=route_requested, stop_requested=stop_requested)
        if route_requested:
            if self.graph.active:
                self.graph.nodes[self.graph.active]['route_state'] = review['route_state']
            end = len(self.history) - 1
            if end > self.vlm.boundary:
                position = review['position_review']
                self.vlm.summaries.append(dict(start=self.vlm.boundary, end=end,
                    observed_at=self.steps, text=position['reason'], completed=position.get('completed', []),
                    grounding=review['map_evidence'].get('route', {}).get('relation', '')))
        self.event('governor_review', **review)
        self.map.save(self.folder)
        return review

    def rollback_node(self, nid, reason):
        success = super().rollback_node(nid, reason)
        if success:
            self.room_tracker.record(self.steps, self.map.active, self.map.position,
                                     self.current.observations['rgb'])
        return success

    def recover_map(self, reason, rollback=True, review=None):
        # Execute only recovery already selected in the scheduled reply.
        recovery = (review or {}).get('recovery', {})
        if rollback:
            node = recovery.get('node_id') if recovery.get('needed') else None
            if node is None:
                self.event('recovery_abstained', reason='No supported rollback node in scheduled review')
                return False
            self.corrections += 1
            self.deviations += int(reason == 'off_route')
            if not self.rollback_node(node, reason):
                return True
        self.target = None
        if self.done:
            return True
        point = recovery.get('waypoint')
        if point is not None and not self.map.transition:
            target = np.asarray(point)
            if self.map.plan(target):
                if not rollback:
                    self.corrections += 1
                self.target = target
                self.correction_steps = 0
                self.point_runs += 1
                self.event('correction_start', target_world=target.tolist(), floor=self.map.active)
        if not rollback:
            self.policy.restore(self.history, self.vlm.instruction)
        return True

    def _run(self):
        self.event('episode_start', instruction=self.vlm.instruction, control='rgb_review_every_16_or_native_stop')
        self.video.append_data(self.current.observations['rgb'])
        self.vlm.extract()
        raw = getattr(self.current, 'raw_observations', self.current.observations)
        self.update_pose(raw)
        if self.pose.pose is None:
            raise RuntimeError('Cannot initialize map localization: ' + self.pose.reason)
        panorama, _ = self.panorama()
        sector = self.call('initial_heading', lambda: self.vlm.orient(panorama))
        turns = 0 if sector is None else sector if sector <= 12 else sector - 24
        for _ in range(abs(turns)):
            if not self.move(3 if turns > 0 else 2, 'initial_alignment', controller=True):
                break
        self.policy.restore(self.history, self.vlm.instruction)
        self.initialize_checkpoint()
        self.map.save(self.folder)

        navigation_steps = 0
        next_review = REVIEW_INTERVAL
        idle = 0
        while not self.done:
            correcting = self.target is not None
            action = self.map_action() if correcting else self.policy.predict(self.current)
            native_stop = not correcting and action == 0
            periodic = navigation_steps == next_review
            if periodic or native_stop:
                if periodic:
                    next_review += REVIEW_INTERVAL
                review = self.scheduled_review(periodic, native_stop)
                off_route = review['route_state'] == 'off_route'
                if native_stop:
                    # A STOP-only acceptance already checks the required route
                    # and endpoint in the STOP verdict itself.
                    stop_review = review
                    if not periodic and review['stop_proposal']['decision'] == 'stop':
                        stop_review = dict(review, route_state='normal')
                    if self.immediate_stop(stop_review, native_proposal=True) or self.done:
                        break
                    self.stop_rejections += 1
                    self.native_stop_retries += 1
                    if self.native_stop_retries > 2:
                        self.termination = 'native_stop_unresolved'
                        break
                    late = review['stop_proposal']['timing'] == 'late'
                    reason = 'off_route' if off_route else 'late_stop' if late else 'early_stop'
                    rollback = off_route or late or review['recovery']['needed']
                    if not self.recover_map(reason, rollback=rollback, review=review):
                        self.recover_map(reason, rollback=False)
                    continue
                if off_route and self.recover_map('off_route', review=review):
                    continue
            if action is None:
                if self.done:
                    break
                idle += 1
                if idle >= 3:
                    self.termination = 'correction_no_progress'
                continue
            idle = 0
            if not self.move(action, 'correction' if correcting else 'normal', controller=correcting):
                break
            navigation_steps += 1
            self.native_stop_retries = 0
            if correcting:
                self.correction_steps += 1
                if self.correction_steps >= self.args.correction_max_steps:
                    self.target = None
                    self.policy.restore(self.history, self.vlm.instruction)
            else:
                self.policy.advance()
            self.graph.observe(self)
            if self.blocked >= 4 and not self.done:
                self.blockage_recoveries += 1
                self.blocked = 0
                self.recover_map('blocked_motion', rollback=False)
            self.map.save(self.folder)
        if navigation_steps == next_review:
            self.scheduled_review(True, False)
        self.termination = self.termination or 'max_steps'
        self.map.save(self.folder)
        self.event('episode_end', reason=self.termination)
        return dict(steps=self.steps, effective_steps=len(self.history)-1, num_rewrites=0,
                    corrections=self.corrections, deviations=self.deviations,
                    blockage_recoveries=self.blockage_recoveries, stop_returns=0,
                    stop_rejections=self.stop_rejections, max_task_depth=1, point_runs=self.point_runs,
                    point_completed=self.point_completed, correction_images=[],
                    final_instruction=self.vlm.instruction, termination_reason=self.termination,
                    summaries=self.vlm.summaries, pose_source=self.pose.selected, floors=self.map.floors,
                    decision_nodes=list(self.graph.nodes.values()),
                    intervened=bool(self.corrections or self.stop_rejections or self.termination == 'governor_stop'))


def room_arguments(original):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--room_python', default=None)
    parser.add_argument('--room_pe_repo', default=str(base.ROOT / 'third_party/perception_models'))
    parser.add_argument('--room_checkpoint', default=str(base.ROOT / 'weights/PE-Core-L14-336.pt'))
    parser.add_argument('--room_device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--room_gpu', default=None)
    parser.add_argument('--room_threads', type=int, default=4)
    parser.add_argument('--room_timeout', type=float, default=600.)
    extra, remaining = parser.parse_known_args()
    argv = sys.argv
    try:
        sys.argv = [argv[0]] + remaining
        args = original()
    finally:
        sys.argv = argv
    extra.room_python = extra.room_python or args.seg_python
    extra.room_gpu = extra.room_gpu or args.seg_gpu
    if extra.room_threads < 1 or extra.room_timeout <= 0:
        parser.error('Room threads and timeout must be positive')
    if extra.room_device == 'cuda' and not extra.room_gpu.isdecimal():
        parser.error('--room_gpu must be one physical GPU index')
    extra.room_checkpoint = str(Path(extra.room_checkpoint).expanduser().resolve())
    extra.room_pe_repo = str(Path(extra.room_pe_repo).expanduser().resolve())
    if not Path(extra.room_checkpoint).is_file():
        parser.error('Missing PE weights: ' + extra.room_checkpoint)
    if not (Path(extra.room_pe_repo) / 'core/vision_encoder/pe.py').is_file():
        parser.error('Missing official perception_models source: ' + extra.room_pe_repo)
    vars(args).update(vars(extra))
    return args


def main():
    global _ROOM_CLIENT
    if sys.argv[1:2] == ['--vlm-worker']:
        return base.vlm_worker(sys.argv[2:])
    if sys.argv[1:2] == ['--seg-worker']:
        return base.segmentation_worker_main(sys.argv[2:])
    base.Planner, base.Episode, base.TopdownMap = RGBPlanner, RGBEpisode, GovernorRGBMap
    base.LocalVLM = lambda args: base.APIVLM(args) if args.vlm_backend == 'api' else base._OriginalLocalVLM(args)
    base.Frame.capture = classmethod(base.capture)
    base.VERSION = 'governor_rgb_review16_pe_top10'
    original = base.arguments
    base.arguments = lambda: room_arguments(original)
    try:
        return base.base_main()
    finally:
        base.arguments = original
        if _ROOM_CLIENT is not None:
            _ROOM_CLIENT.close()
            _ROOM_CLIENT = None


if __name__ == '__main__':
    raise SystemExit(main())
