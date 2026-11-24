"""PE inference worker and four-frame route-segment classification.

Torch is imported ONLY in the worker, keeping StreamVLN's environment isolated.
Camera frames go to PE locally; navigation VLM receives only scores and maps.
"""
import argparse
import base64
import contextlib
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np
from PIL import Image


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sample_four(positions):
    """Four distinct nearest frames to equally spaced arc-length bin centers."""
    positions = np.asarray(positions, dtype=float)
    count = len(positions)
    if count <= 4:
        return list(range(count)), 'arc_length'
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1))]
    stationary = distance[-1] < 1e-6
    axis = np.arange(count, dtype=float) if stationary else distance
    selected = []
    for fraction in (.125, .375, .625, .875):
        available = [i for i in range(count) if i not in selected]
        selected.append(min(available, key=lambda i: abs(axis[i] - fraction * axis[-1])))
    return sorted(selected), 'frame_order_stationary' if stationary else 'arc_length'


def split_route(route, nodes):
    """Rebuild from retained geometry, including retrospectively discovered nodes."""
    nodes = [n for n in nodes if n.get('retained', True)]
    start = next((n['step'] for n in nodes if n.get('kind') == 'start'), None)
    route = [r for r in route if start is None or r['step'] >= start]
    by_step = {n['step']: n['id'] for n in nodes}
    chunks, chunk = [], []
    for row in route:
        if row['floor'] is None or chunk and row['floor'] != chunk[-1]['floor']:
            if chunk:
                chunks.append(chunk)
            chunk = []
        if row['floor'] is not None:
            if not chunk or row['step'] != chunk[-1]['step']:
                chunk.append(row)
    if chunk:
        chunks.append(chunk)
    segments = []
    for rows in chunks:
        cuts = [0] + [i for i in range(1, len(rows)) if rows[i]['step'] in by_step]
        if cuts[-1] != len(rows) - 1:
            cuts.append(len(rows) - 1)
        spans = list(zip(cuts, cuts[1:])) if len(cuts) > 1 else [(0, 0)]
        for first, last in spans:
            selected = rows[first:last + 1]
            segments.append({'id': len(segments) + 1, 'floor': selected[0]['floor'],
                             'start_node': by_step.get(selected[0]['step']),
                             'end_node': by_step.get(selected[-1]['step']),
                             'start_step': selected[0]['step'], 'end_step': selected[-1]['step'],
                             'rows': selected})
    return segments


class TrajectoryRooms:
    def __init__(self, matcher):
        self.matcher = matcher
        self.categories = ()
        self.frames = {}
        self.scores = {}
        self.segments = []

    def set_categories(self, categories):
        if len(categories) != 10 or len(set(categories)) != 10:
            raise ValueError('Room matching requires exactly ten distinct category names')
        self.categories = tuple(categories)
        self.scores.clear()
        self.segments = []

    def record(self, step, floor, position, image, pose=None, detections=()):
        if floor is None:
            return
        image = Image.fromarray(np.asarray(image, dtype=np.uint8)[..., :3]).convert('RGB')
        image.thumbnail((512, 512))
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=92)
        data = buffer.getvalue()
        if pose is not None:
            pose = dict(pose)
            if pose.get('horizontal_fov_degrees') is not None:
                focal = image.width / (2 * math.tan(math.radians(pose['horizontal_fov_degrees']) / 2))
                pose['K'] = [[focal, 0., (image.width - 1) / 2],
                             [0., focal, (image.height - 1) / 2], [0., 0., 1.]]
                pose['image_size'] = list(image.size)
        self.frames[int(step)] = dict(image=base64.b64encode(data).decode('ascii'),
            digest=hashlib.sha256(data).hexdigest(), floor=floor,
            position=np.asarray(position, dtype=float).tolist(), pose=pose,
            detections=list(detections), image_size=list(image.size))

    def truncate(self, step):
        self.frames = {k: v for k, v in self.frames.items() if k <= step}
        self.segments = []

    def refresh(self, route, nodes):
        if not self.categories:
            self.segments = []
            return []
        pending, selected_segments = {}, []
        for segment in split_route(route, nodes):
            rows = segment.pop('rows')
            available = [r for r in rows if r['step'] in self.frames and
                         self.frames[r['step']]['floor'] == segment['floor']]
            indices, basis = sample_four([r['position'] for r in available]) if available else ([], 'arc_length')
            selected = [available[i] for i in indices]
            for row in selected:
                frame = self.frames[row['step']]
                if frame['digest'] not in self.scores:
                    pending[frame['digest']] = frame['image']
            segment.update(sample_steps=[r['step'] for r in selected], sampling_basis=basis,
                           frame_count=len(selected), label_position=rows[len(rows) // 2]['position'])
            selected_segments.append(segment)
        if pending:
            matrix = np.asarray(self.matcher.score(list(pending.values()), self.categories), dtype=float)
            if matrix.shape != (len(pending), 10) or not np.isfinite(matrix).all():
                raise RuntimeError('PE returned an invalid similarity matrix')
            self.scores.update(zip(pending, matrix.tolist()))
        for segment in selected_segments:
            values = [self.scores[self.frames[s]['digest']] for s in segment['sample_steps']]
            if not values:
                segment.update(region_type=None, status='no_frames', top10=[], per_frame=[])
                continue
            matrix = np.asarray(values)
            means = matrix.mean(axis=0)
            ranking = np.argsort(-means, kind='stable')
            winner = int(ranking[0])
            agreement = float(np.mean(matrix.argmax(axis=1) == winner))
            segment.update(region_type=self.categories[winner],
                status='four_frame_argmax' if len(values) == 4 else 'provisional_fewer_than_four',
                score=float(means[winner]), margin=float(means[ranking[0]] - means[ranking[1]]),
                agreement=agreement, mixed_views=agreement < .75,
                top10=[dict(name=self.categories[i], cosine=float(means[i])) for i in ranking],
                per_frame=[dict(step=step, top1=self.categories[int(value.argmax())],
                                cosine=float(value.max()))
                           for step, value in zip(segment['sample_steps'], matrix)])
        self.segments = selected_segments
        return self.segments

    def save(self, folder):
        atomic_json(Path(folder) / 'route_segments.json', dict(
            model='PE-Core-L14-336', candidate_types=list(self.categories),
            aggregation='mean of normalized image/text cosine similarities; argmax over top10',
            segments=self.segments))

    def evidence(self):
        # Full ten-way matrices/rankings stay in local output, not repeated in API
        # prompts. VLM gets the winner and uncertainty evidence once per segment.
        keys = ('id', 'floor', 'start_node', 'end_node', 'start_step', 'end_step',
                'frame_count', 'sample_steps', 'region_type', 'status', 'score',
                'margin', 'agreement', 'mixed_views', 'per_frame')
        return [{key: segment[key] for key in keys if key in segment}
                for segment in self.segments]

    def sampled_views(self):
        """Metadata only; the joint review uses images for its object subtask."""
        membership = {}
        for segment in self.segments:
            for step in segment['sample_steps']:
                membership.setdefault(step, []).append(segment['id'])
        return [dict(id='F%d' % step, step=step, segments=membership[step],
                     floor=self.frames[step]['floor'], position=self.frames[step]['position'],
                     pose=self.frames[step].get('pose'),
                     detections=self.frames[step].get('detections', []))
                for step in sorted(membership) if step in self.frames]


class PEClient:
    def __init__(self, args, vocabulary):
        self.timeout = args.room_timeout
        self.responses = queue.Queue()
        env = os.environ.copy()
        for key in ('PYTHONHOME', 'PYTHONPATH', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            env.pop(key, None)
        env['CUDA_VISIBLE_DEVICES'] = str(args.room_gpu) if args.room_device == 'cuda' else ''
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [args.room_python, '-u', str(Path(__file__).resolve()), '--worker',
                   '--pe_repo', args.room_pe_repo, '--checkpoint', args.room_checkpoint,
                   '--device', args.room_device, '--threads', str(args.room_threads)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
        def reader():
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except json.JSONDecodeError:
                    print('[room-pe] worker: ' + line.rstrip(), file=sys.stderr)
            self.responses.put({'ok': False, 'error': 'PE worker exited'})
        threading.Thread(target=reader, daemon=True).start()
        try:
            self.send({'vocabulary': list(vocabulary)})
            self.read()
        except BaseException:
            self.close()
            raise

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()

    def read(self):
        try:
            value = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError('PE worker timed out; check its environment and checkpoint') from None
        if not value.get('ok'):
            raise RuntimeError('Room similarity: ' + value.get('error', 'unknown error'))
        return value

    def score(self, images, categories):
        self.send({'images': images, 'categories': list(categories)})
        return self.read()['scores']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()


class PEEncoder:
    def __init__(self, args, vocabulary):
        print('[room-pe] importing Torch and official PE dependencies', file=sys.stderr, flush=True)
        sys.path.insert(0, str(Path(args.pe_repo).resolve()))
        import torch
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        self.torch, self.device = torch, args.device
        torch.set_num_threads(args.threads)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('PE CUDA device unavailable')
        print('[room-pe] constructing model and loading local checkpoint on CPU', file=sys.stderr, flush=True)
        model = pe.CLIP.from_config('PE-Core-L14-336', pretrained=False).eval()
        weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        elif 'weights' in weights:
            weights = weights['weights']
        weights = {k.removeprefix('module.'): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)
        del weights
        print('[room-pe] checkpoint loaded; preparing tokenizer and image preprocessing', file=sys.stderr, flush=True)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        self.transform = transforms.get_image_transform(model.image_size)
        self.vocabulary = list(vocabulary)
        templates = ('a photo of a {}.', 'a photo of the interior of a {}.')
        # Text stays on CPU, is encoded ONCE, then discarded. Only the visual
        # tower is moved to the GPU, never the complete dual-encoder checkpoint.
        text_features = []
        print('[room-pe] caching %d category text features on CPU (once per process)' % len(self.vocabulary),
              file=sys.stderr, flush=True)
        with torch.inference_mode():
            for index, name in enumerate(self.vocabulary, 1):
                tokens = tokenizer([template.format(name) for template in templates])
                features = model.encode_text(tokens, normalize=True).float().mean(dim=0)
                text_features.append(torch.nn.functional.normalize(features, dim=0))
                if index % 8 == 0 or index == len(self.vocabulary):
                    print('[room-pe] text features=%d/%d' % (index, len(self.vocabulary)), file=sys.stderr, flush=True)
        self.text = torch.stack(text_features).cpu().numpy()
        self.visual = model.visual
        model.visual = None
        del model, tokenizer, text_features
        gc.collect()
        print('[room-pe] releasing text tower; moving vision tower to ' + args.device, file=sys.stderr, flush=True)
        self.visual.eval().requires_grad_(False)
        self.visual.to(args.device)
        if args.device == 'cuda':
            self.visual.half()
        self.cache = {}
        print('[room-pe] ready for four-frame matching against the selected Top10', file=sys.stderr, flush=True)

    def score(self, images, categories):
        if len(categories) != 10 or len(set(categories)) != 10:
            raise ValueError('Exactly 10 unique candidate types required')
        indices = [self.vocabulary.index(name) for name in categories]
        text = self.text[indices]
        values = []
        torch = self.torch
        for data in images:
            digest = hashlib.sha256(data.encode('ascii')).hexdigest()
            vector = self.cache.get(digest)
            if vector is None:
                with Image.open(io.BytesIO(base64.b64decode(data))) as image:
                    tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
                if self.device == 'cuda':
                    tensor = tensor.half()
                with torch.inference_mode():
                    vector = torch.nn.functional.normalize(self.visual(tensor).float(), dim=-1)
                vector = vector[0].cpu().numpy()
                if len(self.cache) >= 512:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[digest] = vector
            values.append((text @ vector).tolist())
        return values


def worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--pe_repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    channel = sys.stdout
    def reply(value):
        channel.write(json.dumps(value) + '\n')
        channel.flush()
    try:
        initial = json.loads(sys.stdin.readline())
        with contextlib.redirect_stdout(sys.stderr):
            encoder = PEEncoder(args, initial['vocabulary'])
        reply({'ok': True, 'ready': True})
    except Exception as exc:
        reply({'ok': False, 'error': str(exc)[:1500]})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                scores = encoder.score(request['images'], request['categories'])
            reply({'ok': True, 'scores': scores})
        except Exception as exc:
            reply({'ok': False, 'error': str(exc)[:1500]})
    return 0


if __name__ == '__main__':
    raise SystemExit(worker())
