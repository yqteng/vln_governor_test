# -*- coding: utf-8 -*-
"""Standalone TEST: four-action/four-view replay with local Qwen + CLIP.

Upload this ONE file into StreamVLN/streamvln/ (or another folder below the repo).
This local-8B test is separate from the formal API-based Governor implementation.
No sibling Governor scripts are needed. Existing Habitat/StreamVLN dependencies,
local Qwen3-VL-8B and CLIP checkpoint directories are still required.

Always retain the embedded MP3D base28. Local Qwen extracts room HEADS from each
instruction, including plausible semantic inferences, and screens only unrelated
dynamic guesses. Aliases are deduplicated; all base + this episode's dynamic heads
are scored, with no Top10 shortlist. Live dynamic candidates are cleared after
each trajectory. Qwen runs in vlnpa, then exits before CLIP starts in governor_map_models. CLIP classifies
the current region every four executed actions from front/left/right/back RGB-D,
with camera pitch -30 degrees and depth tau 1.5m. Consecutive same-label intervals
share one S segment and color. All valid depth pixels are directly backprojected
by default; each topdown cell retains a measured top-surface RGB sample. Vertical
boundaries have a display stroke; unknown floor is not filled. No API, DINO/SAM, entry lines,
decision nodes, room partition or route-deviation audit. RGB/passability maps
and one portable report.html are updated incrementally. Interrupted runs retain
saved results; same settings skip completed trajectories and reuse target caches.

Only our four-view direct projection is computed, with one replay and one map.
Temporal and aggregation comparisons reuse scores. Extra depth
settings run sequential CLIP forwards on the SAME observations and loaded model;
--skip_depth_comparison omits these extra forwards. There is never a stay bonus.
The portable report focuses on region
names, colored runs, aligned timelines, per-segment details and score evidence.
Use --report_only --output_path EXISTING_RESULTS to refresh a previous report
without replay or models; only saved four-view/raw data is displayed.
Use --gpus 1,5 --workers 2 for two replay processes; parent alone publishes HTML.

Default: physical GPU 1, first 10 successful + 10 failed baseline recordings,
at most 100 actions, output results/test_new_compare_20. This replays actions; it does not
generate StreamVLN actions or change the formal API implementation.
"""


# common.py
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().replace('-', ' ').split())

def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as f:
        if str(path).endswith(('.jsonl', '.jsonl.gz')):
            return [json.loads(s) for s in f if s.strip()]
        return json.load(f)

def instructions(value):
    if isinstance(value, list):
        for item in value:
            yield from instructions(item)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in ('instruction', 'instructions', 'instruction_text'):
                yield from instruction_value(v)
            elif isinstance(v, (dict, list)):
                yield from instructions(v)

def instruction_value(value):
    if isinstance(value, str) and value.strip():
        yield value.strip()
    elif isinstance(value, list):
        for v in value:
            yield from instruction_value(v)
    elif isinstance(value, dict):
        for k in ('instruction_text', 'text', 'instruction'):
            if k in value:
                yield from instruction_value(value[k])
                break

def train_instructions(root):
    root = Path(root).expanduser().resolve()
    candidates = [root] if root.is_file() else sorted((p for p in root.rglob('*') if p.is_file()))
    files = []
    unique = {}

    def token(s, word):
        return re.search('(?<![a-z0-9])' + word + '(?![a-z0-9])', s) is not None
    for p in candidates:
        s = p.as_posix().lower()
        if not s.endswith(('.json', '.json.gz', '.jsonl', '.jsonl.gz')):
            continue
        if not token(s, 'r2r') or not token(s, 'train'):
            continue
        if any((token(s, x) for x in ('rxr', 'val', 'unseen', 'seen', 'test', 'aug', 'augmented', 'synthetic', 'envdrop', 'scalevln', 'prevalent'))):
            continue
        texts = list(instructions(read_json(p)))
        if texts:
            files.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), count=len(texts)))
        for text in texts:
            if all((not c.isalpha() or 'LATIN' in unicodedata.name(c, '') for c in text)):
                unique.setdefault(normalize(text), text)
    if not files or not unique:
        raise ValueError('No original R2R TRAIN instructions: use an R2R/train path or R2R_train.json')
    return ([unique[k] for k in sorted(unique)], files)

def load_vocab(path):
    p = Path(path)
    if p.suffix == '.json':
        data = read_json(p)
        if isinstance(data, dict):
            if data.get('complete') is False:
                raise ValueError('Vocabulary extraction incomplete; finish it first')
            data = data['terms']
    else:
        data = p.read_text(encoding='utf-8-sig').splitlines()
    terms = sorted({normalize(x) for x in data if isinstance(x, str) and x.strip()})
    if not terms:
        raise ValueError('Empty vocabulary')
    return terms


# extract_train_vocab.py
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

def context_limit(config):
    text_config = getattr(config, 'text_config', None)
    limit = text_config.get('max_position_embeddings') if isinstance(text_config, dict) else getattr(text_config, 'max_position_embeddings', None)
    limit = limit or getattr(config, 'max_position_embeddings', None)
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError('Cannot determine local Qwen language-model context limit')
    return limit

class LocalQwen:

    def __init__(self, model_path, device):
        import torch
        import transformers
        path = Path(model_path).expanduser().resolve()
        config_path = path / 'config.json'
        if not config_path.is_file():
            raise FileNotFoundError('Missing local Qwen config: ' + str(config_path))
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        self.is_vl = config.get('model_type') == 'qwen3_vl'
        self.torch, self.device = (torch, device)
        cuda = device.startswith('cuda')
        if cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable in this Python environment')
            torch.cuda.set_device(torch.device('cuda:0' if device == 'cuda' else device))
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        if self.is_vl:
            cls = getattr(transformers, 'Qwen3VLForConditionalGeneration', None)
            if cls is None:
                raise RuntimeError('Qwen3-VL requires its existing vlnpa environment. Run with /home/ubuntun/miniconda3/envs/vlnpa/bin/python; this environment lacks Qwen3VLForConditionalGeneration. Do not use the older CLIP environment for this model.')
            print('[vocab] loading local Qwen3-VL (text-only requests): ' + str(path), flush=True)
            self.processor = transformers.AutoProcessor.from_pretrained(str(path), local_files_only=True)
            self.tokenizer = self.processor.tokenizer
            self.model = cls.from_pretrained(str(path), local_files_only=True, dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True).to(device).eval()
        else:
            if config.get('model_type') not in ('qwen2', 'qwen3'):
                raise ValueError('Supported checkpoints: Qwen3-VL or text Qwen2/Qwen3; got ' + str(config.get('model_type')))
            print('[vocab] loading local text Qwen: ' + str(path), flush=True)
            self.processor = None
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, torch_dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True).to(device).eval()
        self.model.requires_grad_(False)
        self.limit = context_limit(self.model.config)
        print('[vocab] model ready; context=%d; device=%s; no API calls' % (self.limit, device), flush=True)

    def generate(self, messages, max_new_tokens):
        if self.is_vl:
            messages = [dict(role=m['role'], content=[dict(type='text', text=m['content'])]) for m in messages]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.device)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prefix_length = inputs['input_ids'].shape[1]
        if prefix_length + max_new_tokens > self.limit:
            raise ValueError('Batch exceeds context window')
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, return_dict_in_generate=False)
        ids = generated[0, prefix_length:]
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        if len(ids) >= max_new_tokens and (not len(ids) or int(ids[-1].item()) not in eos):
            raise ValueError('Qwen output truncated; retry with a smaller batch')
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


# local_targets.py
import argparse
import contextlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import unicodedata

def _base_region(code, name, description, aliases):
    return dict(code=code, name=name, description=description, aliases=[name] + list(aliases))
BASE_REGIONS = [_base_region('a', 'bathroom', 'a bathroom containing a toilet and a washbasin', ('bath room', 'restroom', 'washroom')), _base_region('b', 'bedroom', 'a bedroom', ('bed room',)), _base_region('c', 'closet', 'a walk-in closet or clothes storage room', ('walk in closet', 'storage room', 'storeroom', 'coat room')), _base_region('d', 'dining room', 'a dining room or breakfast room used for eating', ('diningroom', 'dining area', 'dining hall', 'breakfast room')), _base_region('e', 'entryway/foyer/lobby', 'the main entrance hall, foyer or lobby of a building', ('entryway', 'foyer', 'lobby', 'entrance hall', 'entry hall')), _base_region('f', 'family room', 'a family activity room', ('familyroom',)), _base_region('g', 'garage', 'a garage', ()), _base_region('h', 'hallway', 'an indoor corridor or hallway', ('hall way', 'corridor', 'passage', 'passageway')), _base_region('i', 'library', 'a library or library-style reading room', ()), _base_region('j', 'laundry room/mudroom', 'a laundry room or entrance mudroom', ('laundry room', 'laundryroom', 'mudroom', 'mud room')), _base_region('k', 'kitchen', 'a kitchen', ()), _base_region('l', 'living room', 'the main living room in a home', ('livingroom', 'sitting room', 'sitting area')), _base_region('m', 'meeting room/conference room', 'a meeting room or conference room', ('meeting room', 'meetingroom', 'conference room', 'conferenceroom')), _base_region('n', 'lounge', 'a lounge or relaxation area with comfortable seats', ('lounge room',)), _base_region('o', 'office', 'an office for an individual or small group', ()), _base_region('p', 'porch/terrace/deck/driveway', 'a ground-level outdoor porch, terrace, deck or driveway', ('porch', 'terrace', 'deck', 'driveway', 'patio')), _base_region('r', 'recreation room/game room', 'a recreation room or game room', ('recreation room', 'rec room', 'game room', 'games room', 'billiard room', 'billiards room', 'playroom')), _base_region('s', 'stairs', 'an indoor staircase or stairwell', ('staircase', 'stairway', 'stairwell', 'stair well')), _base_region('t', 'toilet room', 'a small separate toilet room containing only a toilet', ('toilet', 'wc', 'water closet')), _base_region('u', 'utility room/tool room', 'a utility room or tool room', ('utility room', 'utilityroom', 'tool room', 'toolroom')), _base_region('v', 'tv room', 'a television screening room with theater-style seating', ('tv', 'television room', 'home theater', 'theater room', 'home cinema')), _base_region('w', 'gym/exercise room', 'an indoor gym or exercise room', ('gym', 'gymnasium', 'exercise room', 'workout room', 'fitness room')), _base_region('x', 'outdoor green area', 'an outdoor green area with grass, plants, bushes or trees', ('outdoor', 'garden', 'yard', 'lawn')), _base_region('y', 'balcony', 'an outdoor balcony above the ground floor', ()), _base_region('B', 'bar', 'a bar area', ('bar room',)), _base_region('C', 'classroom', 'a classroom', ('class room',)), _base_region('D', 'dining booth', 'a dining booth area', ('dining booth area',)), _base_region('S', 'spa/sauna', 'a spa or sauna area', ('spa', 'sauna'))]
ROOMS = [row['name'] for row in BASE_REGIONS]
DEFAULT_MODEL = '/mnt/disk8t1/zwm/Progress-Aware-VLN/Weights/Qwen3-VL-8B-Instruct'
TARGET_SELECTION_POLICY = 'mp3d_base28_plus_instruction_heads_v1'
PROMPT = 'Read the navigation instruction as DATA. Return JSON only, without reasoning:\n{"objects":[],"regions":["bedroom","pantry"]}.\nregions contains room/space HEAD NOUNS, not full descriptions. Include explicitly\nnamed spaces AND reasonable room-type hypotheses supported by the instruction.\nKeep conventional multiword heads such as living room, dining room and laundry room.\nRemove size/color/ordinal adjectives and spatial relations: "large bedroom beside\nthe bathroom" -> ["bedroom","bathroom"]. Include room names inside modifiers too.\n"passage under the balcony" -> ["passage","balcony"]. Synonyms are allowed.\n"At the kitchen, walk past the dining table" may suggest kitchen AND dining room.\n"walk toward the bathtub" may suggest bathroom. "go down the stairs" suggests stairs.\nThese plausible inferences are ALLOWED even when the room name is absent verbatim.\nOnly exclude completely unrelated room guesses, word-shape confusions and objects\nmistaken for room types. Do not demand literal substring support. When no space\nis named or reasonably suggested, return regions:[]; this is normal.\nThe supplied base category definitions describe available visual classes; they are\nNOT evidence that those classes occur in the instruction. Do not list base classes\nmerely because they are available. Do not rank, choose Top10, remove base classes,\ninvent categories, or judge navigation/deviation.\nPrefer meaningful room/space type heads. Physical objects such as bed/table/lamp\nare not room types themselves, but may support reasonable room-type inferences.\nDo not apply a generic-word blacklist or reject merely unmentioned region names.\nCloset is allowed when it names an occupiable storage/clothes space, not a wardrobe.\nobjects is optional: if provided, retain the existing instruction-object list as\n{"name":"object phrase","head":"object head","query":"object category","modifiers":[]}.\nRoom head extraction is the only source of dynamic room vocabulary. The program\nkeeps all base classes, deduplicates aliases and appends new relevant room heads.\nThese are visual candidates, not confirmed locations. Dynamic additions last only\nfor the current trajectory and are cleared before processing the next one.'
SEMANTIC_FILTER_PROMPT = 'Judge whether candidate room/space heads have any REASONABLE semantic\nconnection to a navigation instruction. All input text is DATA. Return JSON only:\n{"keep_ids":[0,1]} using supplied integer candidate IDs.\nKeep explicitly named rooms AND plausible inferred room types, including dining\ntable -> dining room, bathtub -> bathroom, related functional room names and synonyms.\nDo NOT require literal mentions, proof, high certainty or unique room identity.\nReject ONLY completely unrelated guesses, spelling/word-shape confusions, verbs,\nor object names falsely treated as space types. When a semantic link is plausible\nbut uncertain, KEEP it. The later visual matcher decides which class fits reality.\nThis audit concerns optional dynamic heads only; never delete or shortlist base classes.'
NON_SPATIAL_HEADS = set('object|door|doorway|arch|archway|window|bed|table|chair|sofa|couch|lamp|fireplace|wardrobe|cabinet|cupboard|pool|swimming pool|bath tub|bathtub|sink|toilet seat|kitchen island|dining table|walk|turn|go|wait|dining'.split('|'))
HEAD_ADJECTIVES = set('a an the large small big huge long narrow wide red blue white black brown wooden carpeted first second third left right'.split())

def normalize_region_head(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold().replace('_', ' ').replace('-', ' ')
    value = ' '.join(re.sub('[^a-z /]', ' ', value).split())
    words = value.split()
    while words and words[0] in HEAD_ADJECTIVES:
        words.pop(0)
    if words:
        last = words[-1]
        if last.endswith('ies'):
            words[-1] = last[:-3] + 'y'
        elif last.endswith('s') and (not last.endswith(('ss', 'stairs', 'news'))):
            words[-1] = last[:-1]
    return ' '.join(words)

def _region_key(value):
    return re.sub('[^a-z]', '', normalize_region_head(value))

def base_region_name(head):
    key = _region_key(head)
    return next((row['name'] for row in BASE_REGIONS if any((_region_key(a) == key for a in row['aliases']))), None)

def base_region_vocabulary(extra=()):
    names = list(ROOMS)
    seen = {_region_key(name) for name in names}
    for raw in extra:
        name = base_region_name(raw) or normalize_region_head(raw)
        if name and name not in NON_SPATIAL_HEADS and (_region_key(name) not in seen):
            names.append(name)
            seen.add(_region_key(name))
    return names

def region_clip_prompts(vocabulary):
    definitions = {row['name']: row['description'] for row in BASE_REGIONS}
    return {name: ['a photo of ' + definitions[name] + '.', 'a view from within ' + definitions[name] + '.'] for name in vocabulary if name in definitions}

def _head_in_instruction(head, instruction):
    source = ' '.join(re.sub('[^a-z ]', ' ', unicodedata.normalize('NFKC', instruction).casefold().replace('-', ' ')).split())
    variants = {head, head + 's'}
    if head.endswith('y'):
        variants.add(head[:-1] + 'ies')
    if head.endswith(' room'):
        variants.add(head.replace(' ', ''))
        variants.add(head.replace(' ', '') + 's')
    if head in ('bedroom', 'bathroom', 'hallway', 'stairway', 'stairwell'):
        for suffix in ('room', 'way', 'well'):
            if head.endswith(suffix):
                variants.add(head[:-len(suffix)] + ' ' + suffix)
    return any((re.search('(?<![a-z])' + re.escape(variant) + '(?![a-z])', source) for variant in variants))

def _target_json(raw):
    text = re.sub('<think>.*?</think>', '', raw, flags=re.S | re.I).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer('[\\[{]', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return dict(regions=value)
    raise ValueError('Return a complete JSON object with regions:[] or room head names')

def parse_targets(raw, vocabulary, instruction=None):
    result = _target_json(raw)
    base = base_region_vocabulary(vocabulary)
    key = next((k for k in ('regions', 'region_heads', 'rooms') if k in result), None)
    if key is None:
        raise ValueError('Provide regions as a list of room head names; [] is valid')
    heads = result[key]
    if heads is None:
        heads = []
    if isinstance(heads, str):
        heads = [] if heads.strip().casefold() in ('', 'none', 'null', 'no room', 'no rooms') else [heads]
    if not isinstance(heads, list):
        raise ValueError('regions must be a list of room head names')
    regions = []
    dynamic = []
    discarded = []
    seen = set()
    known = {_region_key(name) for name in base}
    for item in heads:
        if isinstance(item, dict):
            item = item.get('head', item.get('name', item.get('query')))
        if not isinstance(item, str):
            raise ValueError('Each region head must be text')
        head = normalize_region_head(item)
        if not head or head in ('none', 'null', 'no room', 'no rooms'):
            continue
        if head in NON_SPATIAL_HEADS or len(head.split()) > 6 or any((w in head.split() for w in ('with', 'beside', 'under', 'below', 'behind', 'near', 'next', 'above', 'containing'))):
            discarded.append(dict(head=item, reason='not a specific room head'))
            continue
        canonical = base_region_name(head) or head
        if head not in seen:
            literal = _head_in_instruction(head, instruction) if instruction is not None else None
            regions.append(dict(name=head, head=head, query=canonical, modifiers=[], mentioned_literally=literal))
            seen.add(head)
        if _region_key(canonical) not in known:
            dynamic.append(canonical)
            known.add(_region_key(canonical))
    objects = []
    source_objects = result.get('objects', []) or []
    if not isinstance(source_objects, list):
        raise ValueError('objects must be a list or omitted')
    for row in source_objects:
        if isinstance(row, str):
            row = dict(name=row, head=row, query=row, modifiers=[])
        if not isinstance(row, dict):
            continue
        name = row.get('name', row.get('head', ''))
        if not isinstance(name, str) or not name.strip():
            continue
        head = row.get('head') or name
        query = row.get('query') or head
        modifiers = row.get('modifiers', [])
        objects.append(dict(name=name, head=head, query=query, modifiers=modifiers if isinstance(modifiers, list) else []))
    return dict(objects=objects, regions=regions, base_regions=base, instruction_region_heads=[r['head'] for r in regions], dynamic_regions=dynamic, region_vocabulary=base + dynamic, discarded_region_heads=discarded, selection_policy=TARGET_SELECTION_POLICY, dynamic_scope='current_trajectory_only')

def filter_dynamic_targets(model, result, instruction, vocabulary):
    candidates = list(result['dynamic_regions'])
    if not candidates:
        result['dynamic_filter'] = dict(status='not_needed', checked_heads=[], removed_heads=[])
        return result
    payload = dict(instruction=instruction, candidates=[dict(id=i, head=head) for i, head in enumerate(candidates)])
    error = None
    for attempt in range(2):
        try:
            request = json.dumps(payload, ensure_ascii=False)
            if error:
                request += '\nReturn keep_ids as a JSON array of supplied integer IDs. ' + error
            with contextlib.redirect_stdout(sys.stderr):
                raw = model.generate([dict(role='system', content=SEMANTIC_FILTER_PROMPT), dict(role='user', content=request)], max(256, len(candidates) * 8 + 32))
            value = _target_json(raw)
            keep = value.get('keep_ids')
            if not isinstance(keep, list) or any((type(i) is not int or not 0 <= i < len(candidates) for i in keep)):
                raise ValueError('keep_ids must use supplied integer candidate IDs')
            allowed = {candidates[i] for i in keep}
            removed = [head for head in candidates if head not in allowed]
            result['dynamic_regions'] = [head for head in candidates if head in allowed]
            result['region_vocabulary'] = list(result['base_regions']) + result['dynamic_regions']
            result['regions'] = [row for row in result['regions'] if row['query'] not in removed]
            result['instruction_region_heads'] = [row['head'] for row in result['regions']]
            result['discarded_region_heads'] += [dict(head=head, reason='local 8B: completely unrelated or not a space type') for head in removed]
            result['dynamic_filter'] = dict(status='ok', checked_heads=candidates, removed_heads=removed)
            return result
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
    result['dynamic_filter'] = dict(status='unavailable_candidates_retained', checked_heads=candidates, removed_heads=[], error=error)
    return result

def cache_key(instruction, vocabulary, model):
    return digest(dict(instruction=instruction, vocabulary=base_region_vocabulary(vocabulary), model=str(Path(model).expanduser().resolve()), prompt=PROMPT, filter_prompt=SEMANTIC_FILTER_PROMPT, base_definitions=BASE_REGIONS, selection_policy=TARGET_SELECTION_POLICY))

class TargetClient:

    def __init__(self, python, model, timeout=600, max_tokens=2048):
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.responses = queue.Queue()
        self.broken = False
        command = [str(python), '-u', str(Path(__file__).resolve()), '--qwen-worker', '--model', str(model)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

        def reader():
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    print('[local-targets] ' + line.rstrip(), file=sys.stderr)
            self.responses.put(dict(ok=False, error='Local Qwen worker exited'))
        threading.Thread(target=reader, daemon=True).start()
        try:
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            reply = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.broken = True
            self.close()
            raise RuntimeError('Local Qwen timed out; completed instruction targets remain cached') from None
        if not reply.get('ok'):
            raise ValueError(reply.get('error', 'Local Qwen failed'))
        return reply

    def extract(self, instruction, vocabulary):
        if self.broken or self.process.poll() is not None:
            raise RuntimeError('Local Qwen worker unavailable')
        self.process.stdin.write(json.dumps(dict(instruction=instruction, vocabulary=vocabulary, tokens=self.max_tokens)) + '\n')
        self.process.stdin.flush()
        return self.read()['result']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream and (not stream.closed):
                stream.close()

def qwen_worker_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--model', default=DEFAULT_MODEL)
    args = p.parse_args()
    channel = sys.stdout

    def send(value):
        channel.write(json.dumps(value, ensure_ascii=False) + '\n')
        channel.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model = LocalQwen(args.model, 'cuda')
        send(dict(ok=True, ready=True))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                vocabulary = request['vocabulary']
                data = dict(instruction=request['instruction'], base_region_definitions=BASE_REGIONS)
                messages = [dict(role='system', content=PROMPT), dict(role='user', content=json.dumps(data, ensure_ascii=False))]
                for attempt in range(2):
                    try:
                        with contextlib.redirect_stdout(sys.stderr):
                            raw = model.generate(messages, request['tokens'])
                        result = parse_targets(raw, vocabulary, request['instruction'])
                        break
                    except (ValueError, TypeError, KeyError) as exc:
                        if attempt:
                            raise
                        messages[-1]['content'] = json.dumps(data, ensure_ascii=False) + '\nReturn complete JSON. regions:[] is valid; do not rank or pad classes. Fix: ' + str(exc)
                result = filter_dynamic_targets(model, result, request['instruction'], vocabulary)
                send(dict(ok=True, result=result))
            except Exception as exc:
                send(dict(ok=False, error=type(exc).__name__ + ': ' + str(exc)))
    except Exception as exc:
        send(dict(ok=False, error=type(exc).__name__ + ': ' + str(exc)))
        return 1
    return 0


# depth_clip.py
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

def depth_bias(depth, size, patch, tau=1.5, minimum=0.02, strength=1.0):
    if tau <= 0 or not 0 < minimum <= 1 or strength < 0:
        raise ValueError('Invalid decay parameters')
    depth = np.asarray(Image.fromarray(np.asarray(depth, np.float32)).resize((size, size), Image.Resampling.NEAREST))
    if size % patch:
        raise ValueError('Image size must be divisible by patch')
    blocks = depth.reshape(size // patch, patch, size // patch, patch).transpose(0, 2, 1, 3).reshape(-1, patch * patch)
    distances = np.full(len(blocks), np.nan, np.float32)
    for i, block in enumerate(blocks):
        valid = block[np.isfinite(block) & (block > 0)]
        if len(valid) >= 0.25 * len(block):
            distances[i] = np.median(valid)
    good = np.isfinite(distances)
    bias = np.full(len(blocks), math.log(minimum), np.float32)
    bias[good] = np.maximum(-strength * distances[good] / tau, math.log(minimum))
    if strength == 0:
        bias[:] = 0
    return bias

class DepthCLIP:

    def __init__(self, model_path, vocabulary, device='cuda', tau=1.5, strength=1.0, minimum=0.02, text_prompts=None):
        import torch
        from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
        self.torch = torch
        self.device = device
        self.tau = tau
        self.strength = strength
        self.minimum = minimum
        self.categories = list(vocabulary)
        model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval().requires_grad_(False)
        self.processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
        templates = ('a photo of a {}.', 'a photo of the interior of a {}.')
        text = []
        with torch.inference_mode():
            for category in self.categories:
                prompts = (text_prompts or {}).get(category) or [s.format(category) for s in templates]
                if not isinstance(prompts, list) or not prompts or any((not isinstance(p, str) or not p.strip() for p in prompts)):
                    raise ValueError('CLIP text prompts must be nonempty strings for ' + category)
                tokens = tokenizer(prompts, padding=True, return_tensors='pt')
                feature = model.get_text_features(**tokens).float()
                feature = torch.nn.functional.normalize(feature, dim=-1).mean(0)
                text.append(torch.nn.functional.normalize(feature, dim=-1))
        self.text = torch.stack(text).cpu().numpy()
        self.visual = model.vision_model
        self.projection = model.visual_projection
        self.size = self.visual.config.image_size
        self.patch = self.visual.config.patch_size
        del model, tokenizer
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.visual.to(device=device, dtype=dtype)
        self.projection.to(device=device, dtype=dtype)
        self.verify_original_path()

    def verify_original_path(self):
        """Fail startup if the manual zero-bias path differs from installed CLIP."""
        torch = self.torch
        rgb = np.zeros((self.size, self.size, 3), np.uint8)
        rgb[:, :, 0] = np.arange(self.size, dtype=np.uint16)[None, :] % 256
        rgb[:, :, 1] = np.arange(self.size, dtype=np.uint16)[:, None] % 256
        pixels = self.processor(images=Image.fromarray(rgb), do_resize=False, do_center_crop=False, return_tensors='pt').pixel_values
        pixels = pixels.to(device=self.device, dtype=next(self.visual.parameters()).dtype)
        with torch.inference_mode():
            reference = torch.nn.functional.normalize(self.projection(self.visual(pixels).pooler_output).float(), dim=-1)[0].cpu().numpy()
        old = self.strength
        try:
            self.strength = 0.0
            actual = self.encode(rgb, np.ones((self.size, self.size), np.float32))
        finally:
            self.strength = old
        cosine = float(reference @ actual)
        if cosine < 0.999:
            raise RuntimeError('CLIP manual attention compatibility failed: cosine=%g' % cosine)
        print('[clip] zero-bias vs official forward cosine=%.6f' % cosine, file=sys.stderr, flush=True)

    def encode(self, rgb, depth):
        torch = self.torch
        rgb = Image.fromarray(np.asarray(rgb, np.uint8)[..., :3]).resize((self.size, self.size), Image.Resampling.BICUBIC)
        pixels = self.processor(images=rgb, do_resize=False, do_center_crop=False, return_tensors='pt').pixel_values
        pixels = pixels.to(device=self.device, dtype=next(self.visual.parameters()).dtype)
        bias = depth_bias(depth, self.size, self.patch, self.tau, self.minimum, self.strength)
        with torch.inference_mode():
            x = self.visual.pre_layrnorm(self.visual.embeddings(pixels))
            b, n, c = x.shape
            if n != len(bias) + 1:
                raise RuntimeError('CLIP patch/depth alignment failed')
            mask = torch.zeros((1, 1, n, n), device=x.device, dtype=x.dtype)
            mask[:, :, 0, 1:] = torch.as_tensor(bias, device=x.device, dtype=x.dtype)
            for layer in self.visual.encoder.layers:
                residual = x
                h = layer.layer_norm1(x)
                a = layer.self_attn

                def heads(proj):
                    return proj(h).reshape(b, n, a.num_heads, a.head_dim).transpose(1, 2)
                q, k, v = (heads(a.q_proj), heads(a.k_proj), heads(a.v_proj))
                scores = q * a.scale @ k.transpose(-1, -2) + mask
                prob = torch.softmax(scores.float(), dim=-1).to(x.dtype)
                h = (prob @ v).transpose(1, 2).reshape(b, n, c)
                x = residual + a.out_proj(h)
                x = x + layer.mlp(layer.layer_norm2(x))
            cls = self.visual.post_layernorm(x[:, 0, :])
            vector = torch.nn.functional.normalize(self.projection(cls).float(), dim=-1)
        return vector[0].cpu().numpy()

    def classify(self, views, categories=None):
        if len(views) != 4:
            raise ValueError('Exactly front/left/right/back RGB-D views required')
        categories = list(self.categories if categories is None else categories)
        if not categories or len(set(categories)) != len(categories) or any((c not in self.categories for c in categories)):
            raise ValueError('CLIP candidates must be distinct members of loaded vocabulary')
        text = self.text[[self.categories.index(c) for c in categories]]
        scores = np.asarray([text @ self.encode(rgb, depth) for rgb, depth in views])
        means = scores.mean(0)
        order = np.argsort(-means)
        best = int(order[0])
        return dict(region=categories[best], cosine=float(means[best]), margin=float(means[best] - means[order[1]]) if len(order) > 1 else None, scores={name: float(s) for name, s in zip(categories, means)}, per_view=[dict(direction=d, region=categories[int(np.argmax(s))], cosine=float(s.max()), scores={name: float(v) for name, v in zip(categories, s)}) for d, s in zip(('front', 'left', 'right', 'back'), scores)], view_agreement=int((scores.argmax(1) == best).sum()), tau=self.tau, depth_strength=self.strength, interpretation='argmax similarity, not calibrated probability; view may span multiple rooms')

    def classify_comparisons(self, views, categories=None, compare_depth=True):
        """One loaded encoder, sequential forwards; each image/depth pose captured once."""
        reference = self.classify(views, categories)
        if not compare_depth:
            return reference
        old_tau, old_strength = (self.tau, self.strength)
        alternatives = {}
        try:
            for key, tau, strength in (('tau_0.75', 0.75, old_strength), ('tau_3', 3.0, old_strength), ('no_depth', old_tau, 0.0)):
                self.tau, self.strength = (tau, strength)
                alternatives[key] = dict(reference) if (tau, strength) == (old_tau, old_strength) else self.classify(views, categories)
        finally:
            self.tau, self.strength = (old_tau, old_strength)
        return dict(reference, comparison_inputs=alternatives)

def clip_worker_main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--tau', type=float, default=1.5)
    p.add_argument('--strength', type=float, default=1.0)
    p.add_argument('--min_weight', type=float, default=0.02)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    channel = sys.stdout

    def send(value):
        channel.write(json.dumps(value, allow_nan=False) + '\n')
        channel.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            metadata = json.loads(Path(args.vocab).read_text(encoding='utf-8-sig')) if Path(args.vocab).suffix == '.json' else {}
            prompts = metadata.get('text_prompts', {}) if isinstance(metadata, dict) else {}
            encoder = DepthCLIP(args.model, load_vocab(args.vocab), args.device, args.tau, args.strength, args.min_weight, prompts)
        send(dict(ok=True, ready=True))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with np.load(request['packet'], allow_pickle=False) as data:
                    views = [(data['rgb%d' % i], data['depth%d' % i]) for i in range(4)]
                with contextlib.redirect_stdout(sys.stderr):
                    result = encoder.classify_comparisons(views, request.get('categories'), request.get('compare_depth', True))
                send(dict(ok=True, result=result))
            except Exception as exc:
                send(dict(ok=False, error=type(exc).__name__ + ': ' + str(exc)))
    except Exception as exc:
        send(dict(ok=False, error=type(exc).__name__ + ': ' + str(exc)))
        return 1
    return 0


# region_stability.py
import colorsys
import hashlib
import math

def region_color(name):
    key = ' '.join(str(name).casefold().split()).encode('utf-8')
    hue = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') / 2 ** 32
    return '#' + ''.join(('%02x' % round(v * 255) for v in colorsys.hsv_to_rgb(hue, 0.82, 0.77)))

class RegionStabilizer:

    def __init__(self, momentum=0.6, stay_bonus=0.0, fast_margin=0.02, fast_min_views=3, enable_fast=True):
        if not all((math.isfinite(x) for x in (momentum, stay_bonus, fast_margin))) or not 0 <= momentum < 1 or stay_bonus < 0 or (fast_margin <= 0) or (fast_min_views not in (2, 3, 4)):
            raise ValueError('Region momentum must be in [0,1), bonus >= 0 and fast margin > 0')
        self.momentum = momentum
        self.fast_margin = fast_margin
        self.fast_min_views = fast_min_views
        self.enable_fast = bool(enable_fast)
        self.floor = None
        self.ema = None
        self.current = None

    def settings(self):
        return dict(momentum=self.momentum, stay_bonus=0.0, bonus_enabled=False, fast_margin=self.fast_margin, fast_min_views=self.fast_min_views, enable_fast=self.enable_fast)

    def update(self, result, floor):
        scores = {str(k): float(v) for k, v in result['scores'].items()}
        if not scores or not all((math.isfinite(v) for v in scores.values())):
            raise ValueError('Region scores must be a nonempty finite mapping')
        ranked = sorted(scores, key=lambda k: -scores[k])
        raw = ranked[0]
        raw_margin = scores[raw] - scores[ranked[1]] if len(ranked) > 1 else 0.0
        views = result.get('per_view', [])
        votes = sum((v.get('region') == raw for v in views))
        previous = self.current
        reset = self.ema is None or floor != self.floor or set(scores) != set(self.ema)
        if reset:
            self.ema = dict(scores)
            self.current = raw
            reason = 'initial' if previous is None else 'floor_or_vocabulary_reset'
            decision = dict(self.ema)
        elif self.enable_fast and raw != self.current and (len(views) == 4) and (votes >= self.fast_min_views) and (raw_margin >= self.fast_margin):
            self.ema = dict(scores)
            self.current = raw
            decision = dict(self.ema)
            reason = 'fast_switch'
        else:
            self.ema = {k: self.momentum * self.ema[k] + (1 - self.momentum) * scores[k] for k in scores}
            best = max(self.ema, key=self.ema.get)
            decision = dict(self.ema)
            if best != self.current and decision[best] > decision[self.current]:
                self.current = best
                reason = 'smoothed_switch'
            elif raw != self.current:
                reason = 'held_by_momentum'
            else:
                reason = 'unchanged'
        self.floor = floor
        selected = self.current
        agreement = sum((v.get('region') == selected for v in views)) if views else result.get('view_agreement', 0) if selected == raw else 0
        alternatives = [v for k, v in scores.items() if k != selected]
        return dict(result, region=selected, cosine=scores[selected], margin=scores[selected] - max(alternatives) if alternatives else 0.0, view_agreement=agreement, raw_region=raw, raw_cosine=scores[raw], raw_margin=raw_margin, raw_view_agreement=votes, scores=scores, smoothed_scores=dict(self.ema), decision_scores=decision, stability=dict(reason=reason, previous_region=previous, applied_bonus=0.0, switched=previous is not None and previous != selected, suppressed=raw != selected, **self.settings()))


# region_experiments.py
import copy
import math
COMPARISON_VERSION = 3

def comparison_specs(momentum=0.6, fast_margin=0.02):
    specs = []

    def add(identifier, title, method='raw', **kw):
        specs.append(dict(id=identifier, title=title, method=method, source='default', aggregation='mean', retro=False, **kw))
    add('raw', '\u65e0\u52a8\u91cf')
    add('current', '\u5f53\u524d EMA + \u5feb\u5207\u6362', 'ema', beta=momentum, fast=True, fast_margin=fast_margin, votes=3)
    for beta, suffix in ((0.3, '03'), (0.6, '06'), (0.85, '085')):
        add('ema_' + suffix, '\u7eaf EMA beta=' + str(beta), 'ema', beta=beta, fast=False, fast_margin=0.02, votes=3)
    for margin, suffix in ((0.01, '001'), (0.02, '002'), (0.04, '004')):
        add('fast_' + suffix, 'EMA 0.6 + \u5feb\u5207\u5206\u5dee ' + str(margin), 'ema', beta=0.6, fast=True, fast_margin=margin, votes=3)
    add('confirm_2', '\u8fde\u7eed 2 \u6b21\u786e\u8ba4', 'confirm', count=2, fast=False, fast_margin=0.02, votes=3)
    add('confirm_2_fast', '\u8fde\u7eed 2 \u6b21 + \u5f3a\u8bc1\u636e\u5feb\u5207', 'confirm', count=2, fast=True, fast_margin=0.02, votes=3)
    add('confirm_3_fast', '\u8fde\u7eed 3 \u6b21 + \u5f3a\u8bc1\u636e\u5feb\u5207', 'confirm', count=3, fast=True, fast_margin=0.02, votes=3)
    add('confirm_2_retro', '2 \u6b21\u786e\u8ba4 + \u5feb\u5207 + \u4e8b\u540e\u56de\u6807', 'confirm', count=2, fast=True, fast_margin=0.02, votes=3)
    specs[-1]['retro'] = True
    for source, title in (('tau_0.75', 'tau=0.75 m'), ('tau_3', 'tau=3 m'), ('no_depth', '\u65e0\u6df1\u5ea6\u8870\u51cf')):
        add('depth_' + source, title)
        specs[-1]['source'] = source
    for aggregation, title in (('median', '\u56db\u5411\u4e2d\u4f4d\u6570'), ('top2', '\u6bcf\u7c7b\u53d6\u6700\u9ad8\u4e24\u5411\u5747\u503c'), ('weighted', '\u6309\u5404\u89c6\u89d2\u5206\u5dee\u52a0\u6743')):
        add('aggregation_' + aggregation, title)
        specs[-1]['aggregation'] = aggregation
    return specs

def comparison_groups():
    return [dict(id='momentum', title='\u52a8\u91cf\u5927\u5c0f', note='\u7eaf EMA\uff0c\u5168\u90e8\u5173\u95ed\u5feb\u901f\u5207\u6362\uff0c\u5355\u72ec\u6bd4\u8f83\u5386\u53f2\u6743\u91cd\u3002', ids=['raw', 'ema_03', 'ema_06', 'ema_085']), dict(id='current', title='\u5f53\u524d\u7b56\u7565\u4e0e\u5019\u9009\u786e\u8ba4', note='\u5f53\u524d\u914d\u7f6e\u4f5c\u4e3a\u53c2\u8003\uff1b\u786e\u8ba4\u6cd5\u4e0d\u4f7f\u7528 EMA\u3002\u5f3a\u8bc1\u636e\u901a\u9053\u53ea\u5728\u6ee1\u8db3\u5206\u5dee\u548c\u6295\u7968\u6570\u65f6\u89e6\u53d1\u3002', ids=['raw', 'current', 'confirm_2_fast', 'confirm_3_fast']), dict(id='margin', title='\u5feb\u901f\u5207\u6362\u5206\u5dee', note='\u56fa\u5b9a beta=0.6\u30013/4 \u7968\uff1b\u6bd4\u8f83\u5173\u95ed\u5feb\u5207\u4e0e\u5206\u5dee 0.01 / 0.02 / 0.04\u3002', ids=['ema_06', 'fast_001', 'fast_002', 'fast_004']), dict(id='confirmation', title='\u5019\u9009\u6301\u7eed\u786e\u8ba4', note='\u77ed\u6682\u5019\u9009\u53d6\u6d88\u540e\u4e0d\u7559\u5206\u6570\u60ef\u6027\uff1b\u4e24\u6b21/\u4e09\u6b21\u662f\u8fde\u7eed\u91c7\u6837\u6b21\u6570\uff0c\u4e0d\u4fdd\u8bc1\u5176\u7a7a\u95f4\u72ec\u7acb\u6216\u5224\u65ad\u6b63\u786e\u3002', ids=['raw', 'confirm_2', 'confirm_2_fast', 'confirm_3_fast']), dict(id='retro', title='\u5b9e\u65f6\u786e\u8ba4\u4e0e\u4e8b\u540e\u56de\u6807', note='\u4e24\u8005\u5b9e\u65f6\u5224\u65ad\u5b8c\u5168\u76f8\u540c\u3002\u53f3\u4fa7\u53ea\u5728\u786e\u8ba4\u540e\uff0c\u5c06\u5019\u9009\u6301\u7eed\u533a\u95f4\u56de\u6807\uff1b\u672a\u786e\u8ba4\u7684\u5c3e\u90e8\u4fdd\u6301\u539f\u6807\u7b7e\u3002', ids=['confirm_2_fast', 'confirm_2_retro']), dict(id='depth', title='\u6df1\u5ea6\u6ce8\u610f\u529b\u8870\u51cf', note='\u5168\u90e8\u4f7f\u7528\u56db\u5411\u5747\u503c\u3001\u65e0\u52a8\u91cf\u3002\u9ed8\u8ba4\u8870\u51cf\u4e0e tau=0.75 / 3 m\u3001\u5b8c\u5168\u5173\u95ed\u8870\u51cf\u6bd4\u8f83\uff1b\u4e00\u6b21\u56de\u653e\u5185\u4e32\u884c\u8bc4\u5206\u3002', ids=['depth_tau_0.75', 'raw', 'depth_tau_3', 'depth_no_depth']), dict(id='aggregation', title='\u56db\u5411\u5206\u6570\u600e\u6837\u6c47\u603b', note='\u56db\u79cd\u65b9\u5f0f\u4f7f\u7528\u540c\u4e00\u4efd\u9010\u65b9\u5411\u5206\u6570\uff0c\u4e0d\u989d\u5916\u63a8\u7406\u3002\u52a0\u6743\u6cd5\u4ee5\u6bcf\u4e2a\u65b9\u5411 Top1-Top2 \u5206\u5dee\u4e3a\u6743\u91cd\uff0c\u4e0b\u9650 0.0001\u3002', ids=['raw', 'aggregation_median', 'aggregation_top2', 'aggregation_weighted'])]

def observation_input(observation, spec, base):
    if spec['source'] == 'default':
        value = {k: v for k, v in observation.items() if k != 'comparison_inputs'}
    else:
        value = observation.get('comparison_inputs', {}).get(spec['source'])
        if value is None:
            raise ValueError('\u672a\u4fdd\u5b58\u8fd9\u7ec4\u6df1\u5ea6\u524d\u5411\u5206\u6570\uff1b\u9700\u8981\u65b0\u7248\u56de\u653e\u624d\u80fd\u6bd4\u8f83\u3002')
        value = dict(value)
    scores = dict(value['scores'])
    views = [dict(v) for v in value.get('per_view', [])]
    if not scores or any((not math.isfinite(float(v)) for v in scores.values())):
        raise ValueError('\u5019\u9009\u5206\u6570\u4e3a\u7a7a\u6216\u542b\u65e0\u6548\u503c\u3002')
    if spec['aggregation'] != 'mean':
        if len(views) != 4 or any((not set(scores).issubset(v.get('scores', {})) for v in views)):
            raise ValueError('\u65e7\u8bb0\u5f55\u672a\u4fdd\u5b58\u5b8c\u6574\u9010\u65b9\u5411\u5206\u6570\uff0c\u65e0\u6cd5\u51c6\u786e\u91cd\u7b97\u8fd9\u9879\u5bf9\u7167\u3002')
        vectors = [{k: float(v['scores'][k]) for k in scores} for v in views]
        for v, vector in zip(views, vectors):
            best = max(vector, key=vector.get)
            v.update(scores=vector, region=best, cosine=vector[best])
        weights = []
        for vector in vectors:
            ranked = sorted(vector.values(), reverse=True)
            weights.append(max(0.0001, ranked[0] - ranked[1] if len(ranked) > 1 else 0.0001))
        for name in scores:
            values = [v[name] for v in vectors]
            if spec['aggregation'] == 'median':
                scores[name] = sum(sorted(values)[1:3]) / 2
            elif spec['aggregation'] == 'top2':
                scores[name] = sum(sorted(values)[-2:]) / 2
            elif spec['aggregation'] == 'weighted':
                scores[name] = sum((a * b for a, b in zip(values, weights))) / sum(weights)
            else:
                scores[name] = sum(values) / 4
    ranked = sorted(scores, key=lambda k: -scores[k])
    best = ranked[0]
    return dict(scores=scores, region=best, cosine=scores[best], tau=value.get('tau'), margin=scores[best] - scores[ranked[1]] if len(ranked) > 1 else 0.0, per_view=views, view_agreement=sum((v.get('region') == best for v in views)))

class CandidateRegionFilter:
    """Causal consecutive-winner confirmation; cancelled candidates do not accumulate."""

    def __init__(self, count=2, fast=True, margin=0.02, votes=3):
        self.count = count
        self.fast = fast
        self.margin = margin
        self.votes = votes
        self.current = None
        self.floor = None
        self.categories = None
        self.pending = None
        self.streak = 0
        self.start = None

    def update(self, value, floor, step):
        raw = value['region']
        previous = self.current
        discarded = None
        onset = None
        categories = set(value['scores'])
        if self.current is None or self.floor != floor or self.categories != categories:
            self.current = raw
            self.pending = None
            self.streak = 0
            self.start = None
            reason = 'initial' if previous is None else 'floor_or_vocabulary_reset'
        elif raw == self.current:
            if self.pending is not None:
                discarded = dict(region=self.pending, start_step=self.start, count=self.streak)
            self.pending = None
            self.streak = 0
            self.start = None
            reason = 'candidate_cancelled' if discarded else 'unchanged'
        else:
            if raw != self.pending:
                if self.pending is not None:
                    discarded = dict(region=self.pending, start_step=self.start, count=self.streak)
                self.pending = raw
                self.streak = 1
                self.start = step
            else:
                self.streak += 1
            strong = self.fast and len(value.get('per_view', [])) == 4 and (value['view_agreement'] >= self.votes) and (value['margin'] >= self.margin)
            reason = 'candidate_pending'
            if strong or self.streak >= self.count:
                onset = self.start
                self.current = raw
                reason = 'candidate_fast_switch' if strong else 'candidate_confirmed'
        accepted = onset is not None
        stability = dict(reason=reason, previous_region=previous, switched=previous is not None and previous != self.current, pending_region=self.pending, pending_count=self.streak, pending_start_step=self.start, candidate_start_step=onset, confirmation_delay_steps=step - onset if accepted else None, discarded_candidate=discarded, applied_bonus=0.0, bonus_enabled=False)
        if accepted:
            self.pending = None
            self.streak = 0
            self.start = None
        self.floor = floor
        self.categories = categories
        return dict(value, region=self.current, cosine=value['scores'][self.current], raw_region=raw, raw_margin=value['margin'], raw_view_agreement=value['view_agreement'], stability=stability)

def _raw_result(value):
    return dict(value, raw_region=value['region'], raw_margin=value['margin'], raw_view_agreement=value['view_agreement'], stability=dict(reason='raw_argmax', switched=False, applied_bonus=0.0, bonus_enabled=False))

def _compact(result, step, floor):
    names = sorted(result['scores'], key=lambda k: -result['scores'][k])[:5]
    if result['region'] not in names:
        names.append(result['region'])
    return dict(step=step, floor=floor, region=result['region'], raw_region=result['raw_region'], raw_margin=result['raw_margin'], raw_view_agreement=result['raw_view_agreement'], cosine=result['scores'][result['region']], stability=copy.deepcopy(result['stability']), score_details=[dict(name=n, raw=result['scores'][n], smoothed=result.get('smoothed_scores', {}).get(n)) for n in names], per_view=[dict(direction=v.get('direction'), region=v.get('region'), cosine=v.get('cosine')) for v in result.get('per_view', [])])

def _segments(intervals, route):
    segments = []
    for item in intervals:
        if segments and (segments[-1]['region'], segments[-1]['floor'], segments[-1]['end_step']) == (item['region'], item['floor'], item['start_step']):
            s = segments[-1]
            s['end_step'] = item['end_step']
            s['sample_steps'].append(item['sample_step'])
            s['confirmation_steps'].append(item['confirmed_at_step'])
        else:
            s = dict(id=len(segments) + 1, region=item['region'], floor=item['floor'], start_step=item['start_step'], end_step=item['end_step'], sample_steps=[item['sample_step']], confirmation_steps=[item['confirmed_at_step']], color=region_color(item['region']))
            segments.append(s)
        s['sample_count'] = len(s['sample_steps'])
    for s in segments:
        points = [r for r in route if r['floor'] == s['floor'] and s['start_step'] <= r['step'] <= s['end_step']]
        distances = [0.0]
        for a, b in zip(points, points[1:]):
            d = math.hypot(a['position'][0] - b['position'][0], a['position'][2] - b['position'][2]) if b['step'] == a['step'] + 1 else 0.0
            distances.append(distances[-1] + d)
        s['distance_m'] = distances[-1]
        if points:
            s['label_position'] = points[min(range(len(points)), key=lambda i: abs(distances[i] - distances[-1] / 2))]['position']
        s['transition_step_window'] = [s['start_step'], s['sample_steps'][0]]
    return segments

def build_experiments(observations, route, base, momentum=0.6, fast_margin=0.02):
    experiments = []
    for spec in comparison_specs(momentum, fast_margin):
        intervals = []
        results = []
        previous_step = 0
        previous_floor = None
        last_raw = None
        raw_start = 0
        filter_ = None
        if spec['method'] == 'ema':
            filter_ = RegionStabilizer(spec['beta'], 0.0, spec['fast_margin'], spec['votes'], spec['fast'])
        elif spec['method'] == 'confirm':
            filter_ = CandidateRegionFilter(spec['count'], spec['fast'], spec['fast_margin'], spec['votes'])
        delays = []
        stale = 0
        try:
            for original in observations:
                step, floor = (original['step'], original['floor'])
                value = observation_input(original, spec, set(base))
                if (floor, value['region']) != (previous_floor, last_raw):
                    raw_start = step
                if spec['method'] == 'ema':
                    result = filter_.update(value, floor)
                elif spec['method'] == 'confirm':
                    result = filter_.update(value, floor, step)
                else:
                    result = _raw_result(value)
                if result['stability'].get('switched') and result['stability']['reason'] != 'floor_or_vocabulary_reset':
                    if result['region'] == value['region']:
                        delays.append(step - raw_start)
                    else:
                        stale += 1
                item = _compact(result, step, floor)
                results.append(item)
                if step > 0:
                    intervals.append(dict(start_step=previous_step, end_step=step, sample_step=step, floor=floor, region=result['region'], confirmed_at_step=step))
                    onset = result['stability'].get('candidate_start_step')
                    if spec['retro'] and onset is not None:
                        for interval in intervals:
                            if interval['floor'] == floor and onset <= interval['sample_step'] <= step:
                                interval.update(region=result['region'], confirmed_at_step=step)
                previous_step = step
                previous_floor = floor
                last_raw = value['region']
            segs = _segments(intervals, route)
            raw_by_step = {o['step']: o.get('raw_region', o['region']) for o in observations}
            experiments.append(dict(**spec, available=True, segments=segs, results=results, metrics=dict(segments=len(segs), switches=sum((a['floor'] == b['floor'] for a, b in zip(segs, segs[1:]))), single_sample_segments=sum((s['sample_count'] == 1 for s in segs)), aba_returns=sum((a['floor'] == b['floor'] == c['floor'] and a['region'] == c['region'] for a, b, c in zip(segs, segs[1:], segs[2:]))), changed_samples=sum((r['step'] > 0 and r['region'] != raw_by_step.get(r['step']) for r in results)), fast_switches=sum((r['stability']['reason'] in ('fast_switch', 'candidate_fast_switch') for r in results)), cancelled_candidates=sum((bool(r['stability'].get('discarded_candidate')) for r in results)), mean_switch_delay_steps=sum(delays) / len(delays) if delays else None, matched_switch_count=len(delays), stale_switches=stale, retro_relabelled_samples=sum((i['confirmed_at_step'] > i['sample_step'] for i in intervals)))))
        except (KeyError, ValueError, TypeError) as exc:
            experiments.append(dict(**spec, available=False, unavailable_reason=str(exc), segments=[], results=[], metrics={}))
    return dict(version=COMPARISON_VERSION, groups=comparison_groups(), experiments=experiments, semantics='No ground-truth accuracy. switch delay is relative to the current raw-winner run, not a physical boundary. Retro maps are revised after confirmation, not causal predictions.')


# report.py
import base64
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
_REPORT_IMAGE_CACHE = {}

def write_report(output):
    output = Path(output)
    cases, assets, warnings = ([], {}, [])

    def embed(folder, name):
        if not name:
            return None
        path = (folder / name).resolve()
        try:
            path.relative_to(output.resolve())
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            old = _REPORT_IMAGE_CACHE.get(str(path))
            if old is None or old[0] != stamp:
                with Image.open(path) as source:
                    image = source.convert('RGB')
                    image.thumbnail((1600, 1600))
                    buffer = io.BytesIO()
                    image.save(buffer, format='WEBP', quality=92)
                blob = buffer.getvalue()
                old = (stamp, hashlib.sha256(blob).hexdigest(), 'data:image/webp;base64,' + base64.b64encode(blob).decode('ascii'))
                _REPORT_IMAGE_CACHE[str(path)] = old
            assets[old[1]] = old[2]
            return old[1]
        except (OSError, ValueError):
            warnings.append(folder.name + '/' + str(name))
            return None
    for path in sorted(output.glob('*/trajectory.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            warnings.append(path.parent.name + '/trajectory.json')
            continue
        maps = []
        for floor in data.get('maps', []):
            four = next((v for v in floor.get('variants', []) if v.get('views') == 4), None)
            chosen = four or floor
            stable = chosen.get('images', {})
            raw = floor.get('comparisons', {}).get('raw', {})
            background = {k: embed(path.parent, floor.get('background_images', {}).get(k)) for k in ('rgb', 'passability')}
            maps.append(dict(floor=floor['floor'], height=floor.get('height'), geometry=chosen.get('geometry') or floor.get('geometry', {}), images={k: None if background[k] else embed(path.parent, stable.get(k)) for k in ('rgb', 'passability')}, raw_images={k: None if background[k] else embed(path.parent, raw.get(k)) for k in ('rgb', 'passability')}, background_images=background, views=4, shared_geometry=floor.get('shared_geometry', False)))
        data['maps'] = maps
        if data.get('region_comparisons', {}).get('version') != COMPARISON_VERSION:
            stabilization = data.get('region_stabilization', {})
            data['region_comparisons'] = build_experiments(data.get('region_observations', []), data.get('route', []), (data.get('instruction_targets') or {}).get('base_regions', ROOMS), stabilization.get('momentum', 0.6), stabilization.get('fast_margin', 0.02))
        data['latest_views_image'] = embed(path.parent, data.get('latest_views_image'))
        cases.append(data)
    config_path = output / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.exists() else {}
    payload = json.dumps(dict(cases=cases, assets=assets, config=config, warnings=warnings), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'report.html.tmp'
    temporary.write_text(REPORT_HTML.replace('__DATA__', payload), encoding='utf-8')
    temporary.replace(output / 'report.html')
REPORT_HTML = '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>\u56db\u5411\u533a\u57df\u5206\u6bb5\u4e0e\u547d\u540d</title><style>\n:root{font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.65;color:#172f40;background:#f3f6f8;--muted:#6d7f8d;--line:#dfe7eb;--teal:#087e81;--amber:#ac6c15}*{box-sizing:border-box}body{margin:0}button,input,select{font:inherit}button,select,summary{cursor:pointer}button{color:inherit;border:1px solid var(--line);border-radius:8px;background:white;padding:7px 12px}button:hover{border-color:#8ab7bd}button.active{color:white;background:var(--teal);border-color:var(--teal)}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #56b4b8;outline-offset:3px}h1,h2,h3,h4,p{margin:0}h1{font-size:26px;letter-spacing:-.6px}h2{font-size:21px}h3{font-size:16px}h4{font-size:13px}.muted{color:var(--muted)}.small{font-size:12px}.eyebrow{font-size:10px;letter-spacing:2px;font-weight:700;color:#6caeb1}.header{background:#102e3a;color:white;padding:23px 30px;display:flex;align-items:center;justify-content:space-between;gap:20px}.header p{color:#acc3cb;font-size:12px;margin-top:5px}.progress{font-size:13px;text-align:right;white-space:nowrap}.progress b{color:#91ded4;font-size:23px;margin-right:3px}.workspace{max-width:1900px;margin:auto;display:grid;grid-template-columns:248px minmax(0,1fr);gap:24px;padding:24px}.sidebar{align-self:start;position:sticky;top:18px;background:white;border:1px solid var(--line);border-radius:13px;overflow:hidden}.side-top{padding:18px 16px 14px;border-bottom:1px solid var(--line)}.side-top h2{font-size:14px;display:flex;justify-content:space-between}.search{width:100%;min-width:0;margin-top:13px;border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;background:#f8fafb}.case-list{padding:8px;max-height:calc(100vh - 260px);overflow:auto}.case-item{display:block;text-align:left;width:100%;border:0;border-radius:8px;padding:10px;background:transparent;margin:3px 0}.case-item.active{background:#eaf6f4;color:#09676c}.case-name{font-size:12px;font-weight:650;overflow-wrap:anywhere}.meta{display:flex;justify-content:space-between;gap:6px;color:var(--muted);font-size:11px;margin-top:4px}.folder{color:#9badb8;margin-right:6px}.case-item.active .folder{color:var(--teal)}.side-footer{font-size:11px;color:var(--muted);padding:12px 16px;border-top:1px solid var(--line)}.content{min-width:0}.panel{background:white;border:1px solid var(--line);border-radius:13px;padding:22px;margin-bottom:18px}.case-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.status{border-radius:20px;background:#e9f5f1;color:#26745f;font-size:11px;padding:4px 10px}.status.pending{background:#fff3dc;color:#966716}.instruction{font-size:14px;padding:13px 15px;margin-top:15px;background:#f5f8fa;border-left:3px solid #7baab2;border-radius:0 8px 8px 0;overflow-wrap:anywhere}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:16px}.metric{padding-right:14px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.4}.metric span{color:var(--muted);font-size:11px}.teal{color:var(--teal)}.amber{color:var(--amber)}.vocab-head{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.chip{font-size:11px;background:#f1f5f7;border:1px solid #e2e9ed;border-radius:5px;padding:3px 8px}.chip.dynamic{color:#11777c;background:#ecf8f5;border-color:#cfe9e0}details{margin-top:12px}summary{font-size:12px;color:#577585}details[open]>summary{margin-bottom:10px}.word-list{font-size:12px;line-height:1.9;color:#526c7e;margin:8px 0 12px;overflow-wrap:anywhere}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:16px}.toolbar p{font-size:11px;color:var(--muted);margin-top:3px}.controls{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.controls button,.controls select{font-size:12px}.controls select{padding:7px;border:1px solid var(--line);border-radius:8px;background:white;color:#172f40}.duo{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.map-card{border:1px solid var(--line);border-radius:10px;min-width:0;overflow:hidden}.map-title{padding:13px 15px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:10px}.map-title p{font-size:11px;color:var(--muted);margin-top:2px}.mode-number{display:inline-grid;place-items:center;width:25px;height:25px;border-radius:7px;font-size:12px;background:#fff2dc;color:var(--amber);margin-right:8px}.stable .mode-number{background:#e4f5ef;color:var(--teal)}.map-area{padding:12px;position:relative;background:white;min-height:140px}.map-stage{position:relative;cursor:zoom-in;max-width:100%;margin:auto}.map-stage img{width:100%;height:auto;display:block}.map-stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.map-missing{min-height:200px;display:grid;place-items:center;color:var(--muted);font-size:12px;background:#f8fafb}.map-footer{padding:11px 12px;border-top:1px solid var(--line);background:#fbfcfd}.segment-pills{display:flex;gap:6px;flex-wrap:wrap;max-height:130px;overflow:auto}.seg-pill{display:flex;align-items:center;gap:5px;padding:4px 7px;border:1px solid #e0e7eb;border-radius:5px;font-size:11px;background:white;white-space:nowrap}.seg-pill.selected{box-shadow:0 0 0 2px var(--teal);border-color:var(--teal)}.dot{height:8px;width:8px;border-radius:3px;display:inline-block;flex-shrink:0}.legend-note{font-size:10px;color:var(--muted);margin-top:9px}.timeline-wrap{margin-top:21px;border-top:1px solid var(--line);padding-top:17px}.timeline-head{display:flex;justify-content:space-between;gap:10px;font-size:12px;margin-bottom:11px}.track-row{display:grid;grid-template-columns:68px minmax(0,1fr);align-items:center;gap:8px;margin:9px 0}.track-label{font-size:11px;color:var(--muted)}.track{height:31px;background:#eff3f6;position:relative;border-radius:6px;overflow:hidden}.track button{position:absolute;top:0;bottom:0;border:0;border-radius:0;opacity:.88;padding:0;min-width:2px;color:white;font-size:10px;overflow:hidden;text-shadow:0 1px 3px #0008}.track button:hover,.track button.selected{opacity:1;box-shadow:inset 0 0 0 2px #142f3c;z-index:1}.axis{display:flex;justify-content:space-between;padding-left:76px;font-size:10px;color:var(--muted)}.notice{font-size:11px;color:#6d7f8d;margin-top:12px;line-height:1.7}.warning{background:#fff6e6;color:#946117;border:1px solid #efddba;padding:10px 12px;border-radius:8px;font-size:12px;margin-bottom:14px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:10px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:9px 11px;text-align:left;border-bottom:1px solid #e9eef1;vertical-align:top}th{font-weight:500;color:#647b8b;background:#f6f9fa;font-size:11px}tbody tr:last-child td{border-bottom:0}.clickable{cursor:pointer}.clickable:hover{background:#f5faf9}tr.selected{background:#ecf7f4}.segment-detail{margin-top:18px;background:#f7fafb;border:1px solid var(--line);border-radius:10px;padding:17px}.detail-head{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}.detail-stats{font-size:12px;color:#5b7384;margin:8px 0;overflow-wrap:anywhere}.score-row{margin-top:0}.score-row summary{font-size:11px}.directions{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin-top:10px}.direction{border:1px solid var(--line);border-radius:6px;background:white;padding:7px;font-size:10px;overflow-wrap:anywhere}.direction b{display:block;font-size:11px;margin-top:3px}.reason{white-space:normal;min-width:110px;max-width:200px}.settings-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.setting{border-bottom:1px solid var(--line);padding:8px 0;font-size:12px}.setting span{display:block;font-size:11px;color:var(--muted)}.formula{font:12px/1.8 Consolas,monospace;white-space:pre-wrap;background:#f1f6f7;padding:12px;border-radius:8px;margin-top:10px}.sample{width:min(100%,640px);display:block;margin-top:10px;cursor:zoom-in}.empty{padding:90px 25px;text-align:center}.empty h2{font-size:22px;margin-bottom:10px}.empty p{font-size:13px;color:var(--muted)}.page-footer{font-size:11px;color:var(--muted);padding:0 4px 20px}dialog{border:0;border-radius:13px;padding:15px;max-width:96vw;max-height:96vh}dialog::backdrop{background:#122f40d9}dialog img{max-width:90vw;max-height:82vh;object-fit:contain;display:block;margin-top:10px}.zoom-caption{font-size:12px;color:var(--muted)}.zoom-tools{display:flex;gap:20px;justify-content:space-between;align-items:center}\n@media(min-width:1600px){.map-area{padding:18px 28px}}@media(max-width:1100px){.workspace{grid-template-columns:205px minmax(0,1fr);gap:16px;padding:16px}.panel{padding:17px}.duo{gap:10px}.map-title{padding:12px}.map-title h3{font-size:14px}.settings-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.metric b{font-size:23px}}\n@media(max-width:780px){.header{padding:20px;display:block}h1{font-size:23px}.progress{text-align:left;margin-top:12px}.progress b{font-size:18px}.workspace{display:block;padding:12px}.sidebar{position:static;margin-bottom:14px}.case-list{display:flex;max-height:180px;overflow:auto}.case-item{min-width:170px;width:170px;margin:0 3px;flex-shrink:0}.side-top{padding:12px}.side-top h2{display:none}.search{margin:0}.side-footer{display:none}.duo{grid-template-columns:1fr}.metrics{gap:8px}.metric{padding-right:6px}.metric b{font-size:22px}.metric span{font-size:10px}.map-area{padding:14px}.directions{grid-template-columns:repeat(2,minmax(0,1fr))}.panel{padding:15px}.case-heading h2{font-size:18px;overflow-wrap:anywhere}.settings-grid{grid-template-columns:1fr 1fr}.timeline-head{flex-wrap:wrap}.header p{max-width:310px}}\n@media print{.sidebar,.controls,.progress,.page-footer{display:none}.workspace{display:block;padding:0}.panel{break-inside:avoid}.map-stage img{max-height:400px;object-fit:contain}.header{background:white;color:#172f40}.duo{grid-template-columns:1fr 1fr}.segment-pills{max-height:none}}\n/* grouped-comparison-layout */\n.comparison-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.group-note{padding:11px 14px;margin:0 0 15px;background:#edf6f5;color:#39676e;border-radius:8px;font-size:12px}.map-canvas{display:block;width:auto;height:auto;max-width:100%;max-height:500px;margin:auto;cursor:zoom-in}.legacy-map{width:100%;height:auto;display:block;cursor:zoom-in}.track-row{grid-template-columns:175px minmax(0,1fr)}.axis{padding-left:183px}.track-label{font-size:11px;overflow-wrap:anywhere}.map-title h3{font-size:14px}.map-title{align-items:start}.map-missing{padding:20px;white-space:normal}.controls select{max-width:100%}.reason{min-width:180px;max-width:300px}.map-title .muted{white-space:nowrap}.map-area{display:flex;align-items:center;justify-content:center;flex-wrap:wrap}.map-area>.notice{width:100%}\n@media(min-width:1750px){.comparison-grid{grid-template-columns:repeat(4,minmax(0,1fr))}.map-title{min-height:100px}.map-area{min-height:340px}.map-canvas{max-height:420px}}\n@media(max-width:780px){.comparison-grid{grid-template-columns:1fr}.track-row{grid-template-columns:100px minmax(0,1fr)}.axis{padding-left:108px}.controls{max-width:100%}.map-canvas{max-height:500px}.track-label{font-size:10px}}\n</style></head><body>\n<header class="header"><div><div class="eyebrow">FOUR-VIEW \xb7 REGION LABELING</div><h1>\u56db\u5411\u533a\u57df\u5206\u6bb5\u4e0e\u547d\u540d</h1><p>\u540c\u4e00\u8f68\u8ff9 \xb7 \u540c\u4e00\u7ec4\u56db\u5411\u89c2\u6d4b \xb7 \u591a\u6863\u52a8\u91cf \xb7 \u5019\u9009\u786e\u8ba4 \xb7 \u5207\u6362\u4e0e\u5206\u7c7b\u5bf9\u7167 \xb7 bonus = 0</p></div><div class="progress" id="progress"></div></header>\n<div class="workspace"><aside class="sidebar"><div class="side-top"><h2>\u8f68\u8ff9\u76ee\u5f55 <span class="muted small" id="count"></span></h2><input class="search" id="search" placeholder="\u641c\u7d22\u8f68\u8ff9\u3001\u623f\u95f4\u6216\u6307\u4ee4" aria-label="\u641c\u7d22\u8f68\u8ff9"></div><div class="case-list" id="case-list"></div><div class="side-footer">\u9009\u62e9\u4e00\u6761\u8f68\u8ff9\u67e5\u770b\u5206\u6bb5\u3002<br>\u5168\u90e8\u56fe\u7247\u5df2\u5185\u5d4c\uff0c\u53ef\u5355\u6587\u4ef6\u79bb\u7ebf\u6253\u5f00\u3002</div></aside><main class="content" id="content"></main></div>\n<dialog id="zoom"><div class="zoom-tools"><span class="zoom-caption" id="zoom-caption"></span><button id="close-zoom">\u5173\u95ed</button></div><img id="large" alt="\u5730\u56fe\u653e\u5927"></dialog>\n<script id="data" type="application/json">__DATA__</script><script>\n\'use strict\';\nconst {cases,assets,config,warnings}=JSON.parse(document.getElementById(\'data\').textContent),settings=config.settings||{};\nconst $=id=>document.getElementById(id),el=(tag,text,parent,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;if(parent)parent.append(n);return n};\nconst fmt=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):\'\u2014\',source=k=>assets[k]||null;\nconst reasons={initial:\'\u521d\u59cb\u5316\',floor_or_vocabulary_reset:\'\u697c\u5c42\uff0f\u8bcd\u8868\u91cd\u7f6e\',fast_switch:\'\u5f3a\u8bc1\u636e\u5feb\u5207\',smoothed_switch:\'EMA \u63a5\u53d7\u5207\u6362\',held_by_momentum:\'EMA \u4fdd\u7559\u65e7\u6807\u7b7e\',unchanged:\'\u7ef4\u6301\u6807\u7b7e\',raw_argmax:\'\u76f4\u63a5\u53d6\u6700\u5927\u5206\u6570\',candidate_pending:\'\u5019\u9009\u5f85\u786e\u8ba4\',candidate_cancelled:\'\u53d6\u6d88\u77ed\u6682\u5019\u9009\',candidate_confirmed:\'\u8fde\u7eed\u5019\u9009\u5df2\u786e\u8ba4\',candidate_fast_switch:\'\u5019\u9009\u5f3a\u8bc1\u636e\u5feb\u5207\'};\nconst dirs={front:\'\u524d\',left:\'\u5de6\',right:\'\u53f3\',back:\'\u540e\'},imageCache=new Map();\nlet selectedCase=0,selectedFloor=null,selectedGroup=\'momentum\',selectedSegment=null,mapMode=\'rgb\',mapCards=[];\nconst experiments=c=>c.region_comparisons?.experiments||[],getExp=(c,id)=>experiments(c).find(e=>e.id===id),segments=(c,id)=>getExp(c,id)?.segments||[];\nconst lastStep=c=>c.route?.at(-1)?.step??0,label=s=>s.region||\'\u672a\u547d\u540d\',floorNo=f=>Number(f)+1,color=s=>s.color||\'#668a97\';\nconst done=cases.filter(c=>c.complete).length,planned=config.jobs?.length||cases.length;\n$(\'progress\').append(document.createTextNode(\'\u5df2\u5b8c\u6210 \'));el(\'b\',done,$(\'progress\'));$(\'progress\').append(document.createTextNode(`/ ${planned} \u6761 \xb7 \u5df2\u4fdd\u5b58 ${cases.length} \u6761`));\nfunction table(parent,headers){const w=el(\'div\',null,parent,\'table-wrap\'),t=el(\'table\',null,w),tr=el(\'tr\',null,el(\'thead\',null,t));headers.forEach(h=>el(\'th\',h,tr));return el(\'tbody\',null,t)}\nfunction row(body,values){const tr=el(\'tr\',null,body);values.forEach(v=>el(\'td\',v==null?\'\u2014\':String(v),tr));return tr}\nfunction zoomData(data,title){if(!data)return;$(\'large\').src=data;$(\'zoom-caption\').textContent=title||\'\';$(\'zoom\').showModal()}\n$(\'close-zoom\').onclick=()=>$(\'zoom\').close();$(\'zoom\').onclick=e=>{if(e.target===$(\'zoom\'))$(\'zoom\').close()};\nfunction loadImage(key){if(!imageCache.has(key))imageCache.set(key,new Promise(resolve=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>resolve(null);i.src=source(key)}));return imageCache.get(key)}\nfunction pill(parent,s,id){const b=el(\'button\',null,parent,\'seg-pill\');b.dataset.mode=id;b.dataset.sid=s.id;el(\'span\',null,b,\'dot\').style.background=color(s);el(\'span\',`S${s.id} \xb7 ${label(s)}`,b);b.title=`${s.start_step}\u2013${s.end_step} \u6b65`;b.onclick=()=>selectSegment(id,s.id);return b}\nfunction drawList(){const list=$(\'case-list\'),q=$(\'search\').value.trim().toLowerCase();list.replaceChildren();let count=0;cases.forEach((c,i)=>{if(q&&![c.name,c.instruction,...(c.instruction_targets?.region_vocabulary||[])].join(\' \').toLowerCase().includes(q))return;count++;const b=el(\'button\',null,list,\'case-item\'+(selectedCase===i?\' active\':\'\')),name=el(\'div\',null,b,\'case-name\');b.setAttribute(\'aria-pressed\',String(selectedCase===i));el(\'span\',selectedCase===i?\'\u25be\':\'\u25b8\',name,\'folder\');name.append(document.createTextNode(c.name));const meta=el(\'div\',null,b,\'meta\');el(\'span\',c.complete?\'\u5df2\u5b8c\u6210\':\'\u8fdb\u884c\u4e2d \xb7 \u5df2\u4fdd\u5b58\',meta);el(\'span\',`${lastStep(c)} \u6b65`,meta);b.onclick=()=>{selectedCase=i;selectedFloor=null;selectedSegment=null;drawList();buildCase(c)}});$(\'count\').textContent=`${count} \u6761`;if(!count)el(\'p\',\'\u6ca1\u6709\u5339\u914d\u7684\u8f68\u8ff9\',list,\'muted small\')}\n$(\'search\').oninput=drawList;\nfunction groupFor(c){return c.region_comparisons?.groups?.find(g=>g.id===selectedGroup)||c.region_comparisons?.groups?.[0]}\nfunction currentIds(c){return groupFor(c)?.ids||[]}\nfunction matchedSamples(c,s,id){return(getExp(c,id)?.results||[]).filter(r=>r.floor===s.floor&&(s.sample_steps||[]).includes(r.step))}\nfunction buildCase(c){const root=$(\'content\');root.replaceChildren();mapCards=[];if(!c)return;\nconst head=el(\'section\',null,root,\'panel\'),title=el(\'div\',null,head,\'case-heading\');el(\'h2\',c.name,title);el(\'span\',c.complete?\'\u5df2\u5b8c\u6210\u56de\u653e\':\'\u8fdb\u884c\u4e2d \xb7 \u5f53\u524d\u5feb\u7167\',title,\'status\'+(c.complete?\'\':\' pending\'));el(\'p\',c.instruction||\'\u672a\u4fdd\u5b58\u6307\u4ee4\',head,\'instruction\');\nconst metrics=el(\'div\',null,head,\'metrics\');for(const[value,name]of[[lastStep(c),\'\u5df2\u56de\u653e\u52a8\u4f5c\'],[c.region_observations?.length??0,\'\u56db\u5411\u91c7\u6837\u70b9\'],[experiments(c).filter(e=>e.available).length,\'\u53ef\u7528\u5bf9\u7167\u7b56\u7565\'],[c.instruction_targets?.region_vocabulary?.length??\'\u2014\',\'\u5b8c\u6574\u5019\u9009\u7c7b\u522b\']]){const m=el(\'div\',null,metrics,\'metric\');el(\'b\',value,m);el(\'span\',name,m)}\nconst targets=c.instruction_targets||{},vocab=el(\'details\',null,head);el(\'summary\',`\u7edf\u4e00\u8bcd\u8868\uff1abase ${targets.base_regions?.length??28} \u7c7b \uff0b \u5f53\u6761\u52a8\u6001 ${targets.dynamic_regions?.length??0} \u7c7b`,vocab);\nfor(const[k,name]of[[\'objects\',\'8B \u63d0\u53d6\u7684\u7269\u4f53\'],[\'regions\',\'8B \u63d0\u53d6\u7684\u533a\u57df\u4e2d\u5fc3\u8bed\'],[\'base_regions\',\'\u56fa\u5b9a base \u8bcd\u8868\'],[\'dynamic_regions\',\'\u672c\u6761\u8f68\u8ff9\u52a8\u6001\u65b0\u589e\'],[\'region_vocabulary\',\'\u672c\u6761\u5b8c\u6574\u5339\u914d\u8bcd\u8868\']]){el(\'h4\',name,vocab);el(\'p\',(targets[k]||[]).map(x=>typeof x===\'string\'?x:x.head||x.name).join(\' \xb7 \')||\'\u65e0\',vocab,\'word-list\')}\nel(\'p\',\'\u6240\u6709\u7b56\u7565\u4f7f\u7528\u540c\u4e00\u4efd\u5b8c\u6574 base\uff0b\u52a8\u6001\u8bcd\u8868\u3002\u52a8\u6001\u7c7b\u522b\u53ea\u5bf9\u672c\u6761\u8f68\u8ff9\u751f\u6548\uff0c\u7ed3\u675f\u540e\u6062\u590d base\u3002\',vocab,\'notice\');\nconst overview=el(\'details\',null,head);el(\'summary\',\'\u5168\u90e8\u5df2\u4fdd\u5b58\u8f68\u8ff9\u7684\u7b56\u7565\u6c47\u603b\',overview);const complete=cases.filter(x=>x.complete),pool=complete.length?complete:cases;el(\'p\',`\u7edf\u8ba1 ${pool.length} \u6761${complete.length?\'\u5df2\u5b8c\u6210\':\'\u5f53\u524d\u5df2\u4fdd\u5b58\'}\u8f68\u8ff9\u3002\u7edf\u8ba1\u91cf\u63cf\u8ff0\u7a33\u5b9a\u6027\uff0c\u4e0d\u662f\u51c6\u786e\u7387\uff1b\u672a\u63d0\u4f9b\u623f\u95f4\u771f\u503c\uff0c\u4e0d\u80fd\u81ea\u52a8\u636e\u6b64\u9009\u201c\u6700\u597d\u201d\u7684\u7b56\u7565\u3002`,overview,\'notice\');\nconst summaryBody=table(overview,[\'\u7b56\u7565\',\'\u53ef\u7528\u8f68\u8ff9\',\'\u5e73\u5747\u6bb5\u6570\',\'\u5e73\u5747\u5355\u91c7\u6837\u6bb5\',\'\u5e73\u5747 A\u2192B\u2192A\',\'\u5e73\u5747\u5feb\u5207\u6b21\u6570\']);for(const e of experiments(c)){const list=pool.map(x=>getExp(x,e.id)).filter(x=>x?.available&&x.results?.some(r=>r.step>0)),mean=k=>list.length?list.reduce((a,x)=>a+(x.metrics[k]||0),0)/list.length:null;row(summaryBody,[e.title,list.length,fmt(mean(\'segments\')),fmt(mean(\'single_sample_segments\')),fmt(mean(\'aba_returns\')),fmt(mean(\'fast_switches\'))])}\nconst panel=el(\'section\',null,root,\'panel\'),bar=el(\'div\',null,panel,\'toolbar\'),left=el(\'div\',null,bar);el(\'h3\',\'\u540c\u4e00\u8f68\u8ff9\uff0c\u6309\u56e0\u7d20\u67e5\u770b\u5bf9\u7167\',left);el(\'p\',\'\u5730\u56fe\u59cb\u7ec8\u4e3a\u6211\u4eec\u7684\u56db\u5411\u53cd\u6295\u5f71\u3002\u5207\u6362\u5bf9\u7167\u7ec4\u4e0d\u4f1a\u91cd\u65b0\u8fd0\u884c\u8f68\u8ff9\u3002\',left);\nconst controls=el(\'div\',null,bar,\'controls\'),groupSelect=el(\'select\',null,controls);groupSelect.setAttribute(\'aria-label\',\'\u9009\u62e9\u5bf9\u7167\u7ec4\');for(const g of c.region_comparisons?.groups||[]){const option=el(\'option\',g.title,groupSelect);option.value=g.id}const group=groupFor(c);if(group){selectedGroup=group.id;groupSelect.value=group.id}groupSelect.onchange=()=>{selectedGroup=groupSelect.value;selectedSegment=null;buildCase(c)};\nconst floorSelect=el(\'select\',null,controls);floorSelect.setAttribute(\'aria-label\',\'\u9009\u62e9\u697c\u5c42\');for(const f of c.maps||[]){const o=el(\'option\',`\u697c\u5c42 ${floorNo(f.floor)}`,floorSelect);o.value=f.floor}if(!c.maps?.some(f=>f.floor===selectedFloor))selectedFloor=c.maps?.[0]?.floor??0;floorSelect.value=selectedFloor;floorSelect.onchange=()=>{selectedFloor=Number(floorSelect.value);selectedSegment=null;buildCase(c)};\nfor(const[k,name]of[[\'rgb\',\'RGB \u5730\u56fe\'],[\'passability\',\'\u901a\u884c\u6027\u5730\u56fe\']]){const b=el(\'button\',name,controls,mapMode===k?\'active\':\'\');b.onclick=()=>{mapMode=k;buildCase(c)}}\nel(\'p\',group?.note||\'\u6682\u65e0\u5bf9\u7167\u8bb0\u5f55\',panel,\'group-note\');if(selectedGroup===\'depth\')el(\'p\',`\u201c\u9ed8\u8ba4\u201d\u7b56\u7565\u5b9e\u9645 tau=${settings.depth_tau??1.5} m\u3001strength=${settings.depth_strength??1}\u3002\u5176\u4ed6\u7ec4\u56fa\u5b9a\u4f7f\u7528\u9ed8\u8ba4\u8870\u51cf\u3002`,panel,\'notice\');\nconst f=c.maps?.find(f=>f.floor===selectedFloor),grid=el(\'div\',null,panel,\'comparison-grid\');\nfor(const[id,index]of currentIds(c).map((id,i)=>[id,i])){const e=getExp(c,id);if(!e)continue;const card=el(\'article\',null,grid,\'map-card \'+(index?\'stable\':\'raw\')),ch=el(\'div\',null,card,\'map-title\'),a=el(\'div\',null,ch),h=el(\'h3\',null,a);el(\'span\',String(index+1).padStart(2,\'0\'),h,\'mode-number\');h.append(document.createTextNode(e.title));el(\'p\',e.retro?\'\u4e8b\u540e\u4fee\u8ba2\u540e\u7684\u5206\u6bb5\uff0c\u975e\u5b9e\u65f6\u8f93\u51fa\':e.method===\'confirm\'?`\u8fde\u7eed ${e.count} \u6b21\u786e\u8ba4${e.fast?\'\uff0c\u5141\u8bb8\u5f3a\u8bc1\u636e\u5feb\u5207\':\'\'}`:e.method===\'ema\'?`beta=${e.beta} \xb7 ${e.fast?\'\u5feb\u5207\u5f00\u542f\':\'\u5feb\u5207\u5173\u95ed\'} \xb7 bonus=0`:\'\u5f53\u524d\u5206\u6570\u76f4\u63a5\u51b3\u5b9a\u6807\u7b7e\',a);el(\'span\',`${segments(c,id).filter(s=>s.floor===selectedFloor).length} \u6bb5`,ch,\'muted small\');\nconst area=el(\'div\',null,card,\'map-area\');if(!e.available){el(\'div\',e.unavailable_reason,area,\'map-missing\');continue}\nconst background=f?.background_images?.[mapMode];if(source(background)){const canvas=el(\'canvas\',null,area,\'map-canvas\');canvas.setAttribute(\'role\',\'img\');canvas.setAttribute(\'aria-label\',`${c.name} ${e.title}`);const g=f.geometry;if(g?.image_size){canvas.width=g.image_size[0];canvas.height=g.image_size[1]}const item={canvas,id,key:background,geometry:g,c};mapCards.push(item);paint(item);canvas.onclick=()=>zoomData(canvas.toDataURL(\'image/png\'),`${c.name} / ${e.title}`)}else if((id===\'raw\'||id===\'current\')&&source((id===\'raw\'?f?.raw_images:f?.images)?.[mapMode])){const im=el(\'img\',null,area,\'legacy-map\');const key=(id===\'raw\'?f.raw_images:f.images)[mapMode];im.src=source(key);im.alt=e.title;im.onclick=()=>zoomData(source(key),e.title);el(\'p\',\'\u65e7\u7248\u4fdd\u5b58\u5730\u56fe\uff1b\u5176\u4ed6\u7b56\u7565\u9700\u8981\u65e0\u8f68\u8ff9\u5e95\u56fe\u624d\u80fd\u51c6\u786e\u91cd\u7ed8\u3002\',area,\'notice\')}else el(\'div\',\'\u65e7\u8bb0\u5f55\u6ca1\u6709\u65e0\u8f68\u8ff9\u5e95\u56fe\u3002\u53ef\u5148\u6bd4\u8f83\u4e0b\u65b9\u65f6\u95f4\u5e26\u548c\u5206\u6570\uff1b\u65b0\u7248\u56de\u653e\u4f1a\u4fdd\u5b58\u5e95\u56fe\u3002\',area,\'map-missing\');\nconst foot=el(\'div\',null,card,\'map-footer\'),pills=el(\'div\',null,foot,\'segment-pills\');segments(c,id).filter(s=>s.floor===selectedFloor).forEach(s=>pill(pills,s,id));el(\'p\',\'\u540c\u7c7b\u522b\u56fa\u5b9a\u540c\u8272\uff1bS \u7f16\u53f7\u5728\u6bcf\u79cd\u7b56\u7565\u5185\u72ec\u7acb\u3002\',foot,\'legend-note\')}\nconst timeline=el(\'div\',null,panel,\'timeline-wrap\');el(\'h4\',\'\u540c\u523b\u5ea6\u5206\u6bb5\u65f6\u95f4\u5e26\',timeline);const end=Math.max(lastStep(c),1);\nfor(const id of currentIds(c)){const e=getExp(c,id);if(!e)continue;const line=el(\'div\',null,timeline,\'track-row\');el(\'span\',e.title,line,\'track-label\');const track=el(\'div\',null,line,\'track\');if(!e.available){el(\'span\',\'\u8fd9\u9879\u6570\u636e\u672a\u4fdd\u5b58\',track,\'muted small\');continue}segments(c,id).filter(s=>s.floor===selectedFloor).forEach(s=>{const b=el(\'button\',`S${s.id}`,track);b.style.left=`${100*s.start_step/end}%`;b.style.width=`${100*(s.end_step-s.start_step)/end}%`;b.style.background=color(s);b.dataset.mode=id;b.dataset.sid=s.id;b.title=`${e.title} / S${s.id} ${label(s)} / ${s.start_step}\u2013${s.end_step} \u6b65`;b.setAttribute(\'aria-label\',b.title);b.onclick=()=>selectSegment(id,s.id)})}\nconst axis=el(\'div\',null,timeline,\'axis\');[0,Math.round(end/2),end].forEach(n=>el(\'span\',`${n} \u6b65`,axis));el(\'p\',\'\u6807\u7b7e\u7531\u533a\u95f4\u7ec8\u70b9\u7684\u56db\u5411\u89c2\u6d4b\u5206\u914d\u7ed9\u524d\u4e00\u4e2a\u52a8\u4f5c\u533a\u95f4\uff1b\u5207\u6362\u4f4d\u7f6e\u6700\u591a\u7cbe\u786e\u5230\u91c7\u6837\u95f4\u9694\u3002\u539f\u5730\u89c2\u5bdf\u3001\u540c\u540d\u4f46\u4e0d\u540c\u5b9e\u4f8b\u7684\u623f\u95f4\u90fd\u4e0d\u80fd\u636e\u6b64\u76f4\u63a5\u786e\u5b9a\u771f\u5b9e\u623f\u95f4\u8fb9\u754c\u3002\',panel,\'notice\');\nconst stats=el(\'section\',null,root,\'panel\');el(\'h3\',\'\u8fd9\u7ec4\u7b56\u7565\u6539\u53d8\u4e86\u4ec0\u4e48\',stats);const st=table(stats,[\'\u7b56\u7565\',\'\u8f68\u8ff9\u6bb5\',\'\u5355\u91c7\u6837\u6bb5\',\'A\u2192B\u2192A\',\'\u4e0d\u540c\u4e8e\u9ed8\u8ba4\u539f\u59cb\xb9\',\'\u5feb\u5207\',\'\u53d6\u6d88\u5019\u9009\',\'\u539f\u59cb\u8fde\u7eed\u80dc\u51fa\u540e\u5207\u6362\xb2\',\'\u5f02\u4e8e\u5f53\u6b21\u539f\u59cb\u7684\u5207\u6362\xb3\',\'\u56de\u6807\u91c7\u6837\']);\nfor(const id of currentIds(c)){const e=getExp(c,id),m=e?.metrics||{};row(st,[e?.title||id,e?.available?m.segments:\'\u672a\u4fdd\u5b58\',m.single_sample_segments,m.aba_returns,m.changed_samples,m.fast_switches,m.cancelled_candidates,m.mean_switch_delay_steps==null?\'\u2014\':`${fmt(m.mean_switch_delay_steps,1)} \u6b65 / ${m.matched_switch_count} \u6b21`,m.stale_switches,m.retro_relabelled_samples])}\nel(\'p\',\'\xb9 \u6309\u5f53\u65f6\u5b9e\u65f6\u8f93\u51fa\u6bd4\u8f83\uff0c\u4e0d\u6309\u4e8b\u540e\u56de\u6807\u8ba1\u7b97\u3002\xb2 \u4ec5\u7edf\u8ba1\u5207\u6362\u6807\u7b7e\u4e0e\u5f53\u6b21\u539f\u59cb Top1 \u76f8\u540c\u7684\u4e8b\u4ef6\uff0c\u53c2\u7167\u8fd9\u6b21\u539f\u59cb\u6807\u7b7e\u8fde\u7eed\u80dc\u51fa\u7684\u8d77\u70b9\uff1b\u4e0d\u662f\u76f8\u5bf9\u771f\u5b9e\u8fc7\u95e8\u4f4d\u7f6e\u7684\u5ef6\u8fdf\u3002\xb3 \u5207\u6362\u65f6\uff0c\u65b0\u7684\u7a33\u5b9a\u6807\u7b7e\u4ecd\u4e0e\u5f53\u6b21\u539f\u59cb Top1 \u4e0d\u540c\uff0c\u53ef\u7528\u4e8e\u68c0\u67e5\u5386\u53f2\u5206\u6570\u662f\u5426\u628a\u6b64\u524d\u5019\u9009\u62d6\u5230\u540e\u9762\u3002\u4e0a\u8ff0\u7edf\u8ba1\u90fd\u4e0d\u4ee3\u8868\u771f\u503c\u9519\u8bef\uff1b\u5355\u91c7\u6837\u6bb5\u548c A\u2192B\u2192A \u4e5f\u53ef\u80fd\u662f\u771f\u5b9e\u77ed\u533a\u57df\u3002\',stats,\'notice\');\nconst details=el(\'section\',null,root,\'panel\'),db=el(\'div\',null,details,\'toolbar\');el(\'h3\',\'\u5206\u6bb5\u4e0e\u9010\u6b21\u5224\u65ad\u4f9d\u636e\',db);const choose=el(\'select\',null,db);choose.setAttribute(\'aria-label\',\'\u9009\u62e9\u8be6\u60c5\u7b56\u7565\');for(const id of currentIds(c)){const e=getExp(c,id);if(e?.available){const o=el(\'option\',e.title,choose);o.value=id}}\nlet selected=selectedSegment&&getExp(c,selectedSegment.mode)?.available&&currentIds(c).includes(selectedSegment.mode)&&segments(c,selectedSegment.mode).find(s=>s.id===selectedSegment.id&&s.floor===selectedFloor);if(!selected){const id=currentIds(c).find(id=>getExp(c,id)?.available&&segments(c,id).some(s=>s.floor===selectedFloor));selectedSegment=id?{mode:id,id:segments(c,id).find(s=>s.floor===selectedFloor).id}:null}choose.value=selectedSegment?.mode||\'\';choose.onchange=()=>{const s=segments(c,choose.value).find(s=>s.floor===selectedFloor);if(s)selectSegment(choose.value,s.id)};el(\'div\',null,details).id=\'segment-table\';el(\'div\',null,details,\'segment-detail\').id=\'segment-detail\';\nconst extra=el(\'section\',null,root,\'panel\'),last=el(\'details\',null,extra);el(\'summary\',`\u6700\u540e\u4e00\u6b21\u56db\u5411\u89c2\u6d4b \xb7 \u7b2c ${c.region_observations?.at(-1)?.step??0} \u6b65`,last);el(\'p\',\'\u4ec5\u4e3a\u6700\u540e\u4e00\u7ec4\u89c2\u6d4b\uff0c\u4e0d\u662f\u5f53\u524d\u9009\u4e2d\u6bb5\u7684\u5168\u90e8\u5386\u53f2\u56fe\u50cf\u3002\',last,\'notice\');if(source(c.latest_views_image)){const im=el(\'img\',null,last,\'sample\');im.src=source(c.latest_views_image);im.alt=\'\u6700\u540e\u4e00\u6b21\u524d\u3001\u5de6\u3001\u53f3\u3001\u540e\u89c2\u6d4b\';im.loading=\'lazy\';im.onclick=()=>zoomData(im.src,\'\u6700\u540e\u4e00\u6b21\u56db\u5411\u89c2\u6d4b\')}\nconst method=el(\'details\',null,extra);el(\'summary\',\'\u6240\u6709\u5bf9\u7167\u7ec4\u4e0e\u5171\u540c\u8bbe\u7f6e\',method);const mb=table(method,[\'\u5bf9\u7167\u7ec4\',\'\u6bd4\u8f83\u5185\u5bb9\']);for(const g of c.region_comparisons?.groups||[])row(mb,[g.title,g.note]);el(\'p\',`\u56db\u5411\u521d\u59cb\u5316\u3001\u6bcf 4 \u6b65\u66f4\u65b0\uff1b\u4f4e\u5934 ${c.camera_settings?.pitch_degrees??-30}\xb0\uff1b\u9ed8\u8ba4\u6df1\u5ea6 tau=${settings.depth_tau??1.5} m\uff1b\u5b8c\u6574 base\uff0b\u52a8\u6001\u8bcd\u8868\uff1b\u6240\u6709\u7b56\u7565 bonus=0\u3002\u5730\u56fe\u4ec5\u6784\u5efa\u4e00\u6b21\uff0c\u539f\u59cb\u5206\u6570\u53ea\u4fdd\u5b58\u4e00\u4efd\uff1b\u4e0d\u540c\u6df1\u5ea6\u8bbe\u7f6e\u590d\u7528\u6a21\u578b\uff0c\u4e32\u884c\u505a\u989d\u5916\u524d\u5411\u3002`,method,\'notice\');el(\'div\',\'EMA: m\u209c = beta * m\u209c\u208b\u2081 + (1 - beta) * s\u209c\\n\u5019\u9009\u786e\u8ba4\uff1a\u65b0\u7c7b\u522b\u8fde\u7eed\u80dc\u51fa\u8fbe\u5230\u6b21\u6570\u540e\u5207\u6362\uff1b\u77ed\u6682\u5019\u9009\u53d6\u6d88\uff0c\u4e0d\u79ef\u7d2f EMA\u3002\\n\u5f3a\u8bc1\u636e\u5feb\u5207\uff1a\u65b0\u7c7b\u522b\u539f\u59cb\u5206\u5dee\u8fbe\u5230\u9608\u503c\uff0c\u4e14\u65b9\u5411\u6295\u7968\u8fbe\u6807\u3002\\n\u4e8b\u540e\u56de\u6807\uff1a\u786e\u8ba4\u540e\u624d\u4fee\u8ba2\u5019\u9009\u6301\u7eed\u533a\u95f4\uff0c\u5e76\u5355\u72ec\u8bb0\u5f55\u786e\u8ba4\u6b65\u9aa4\u3002\',method,\'formula\');\nel(\'p\',\'\u5355\u6587\u4ef6\u79bb\u7ebf\u62a5\u544a \xb7 \u8fd0\u884c\u4e2d\u66f4\u65b0 \xb7 \u4e0d\u4f7f\u7528\u6bb5\u6570\u4ee3\u66ff\u51c6\u786e\u7387\',root,\'page-footer\');if(selectedSegment)selectSegment(selectedSegment.mode,selectedSegment.id);else $(\'segment-detail\').textContent=\'\u5c1a\u65e0\u5df2\u6267\u884c\u52a8\u4f5c\u533a\u95f4\u3002\';\n}\nasync function paint(card){const{canvas,c,id,geometry:g}=card;if(!g?.image_size)return;const im=await loadImage(card.key);if(!im||!canvas.isConnected)return;const ctx=canvas.getContext(\'2d\'),[w,h]=g.image_size;canvas.width=w;canvas.height=h;ctx.drawImage(im,0,0,w,h);const pitch=g.meters_per_pixel,org=g.origin_xz,xy=p=>[Math.floor(p[0]/pitch)-Math.round(org[0]/pitch),Math.floor(p[2]/pitch)-Math.round(org[1]/pitch)],rs=(c.route||[]).filter(r=>r.floor===selectedFloor);\nfunction path(s,stroke,width){ctx.beginPath();const points=rs.filter(r=>s.start_step<=r.step&&r.step<=s.end_step);points.forEach((r,i)=>{const p=xy(r.position);if(i&&r.step===points[i-1].step+1)ctx.lineTo(...p);else ctx.moveTo(...p)});ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.lineJoin=\'round\';ctx.lineCap=\'round\';ctx.stroke()}\nfor(const s of segments(c,id).filter(s=>s.floor===selectedFloor)){path(s,color(s),3);if(s.label_position){const[x,y]=xy(s.label_position);ctx.font=\'13px sans-serif\';ctx.lineWidth=3;ctx.strokeStyle=\'white\';ctx.strokeText(\'S\'+s.id,x+4,y+3);ctx.fillStyle=color(s);ctx.fillText(\'S\'+s.id,x+4,y+3)}}\nif(selectedSegment?.mode===id){const s=segments(c,id).find(s=>s.id===selectedSegment.id);if(s){path(s,\'white\',8);path(s,color(s),4)}}if(rs.length){const[x,y]=xy(rs.at(-1).position);ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle=\'#ed4b57\';ctx.fill();ctx.strokeStyle=\'white\';ctx.lineWidth=1;ctx.stroke()}}\nfunction selectSegment(id,sid){const c=cases[selectedCase],e=getExp(c,id),s=e?.segments.find(s=>s.id===sid);if(!s)return;selectedSegment={mode:id,id:sid};for(const node of document.querySelectorAll(\'[data-mode][data-sid]\'))node.classList.toggle(\'selected\',node.dataset.mode===id&&Number(node.dataset.sid)===sid);mapCards.forEach(paint);const chooser=document.querySelector(\'[aria-label="\u9009\u62e9\u8be6\u60c5\u7b56\u7565"]\');if(chooser)chooser.value=id;\nconst list=$(\'segment-table\');list.replaceChildren();const tb=table(list,[\'\u8f68\u8ff9\u6bb5 / \u533a\u57df\',\'\u6b65\u6570\u8303\u56f4\',\'\u6c34\u5e73\u8def\u7a0b\',\'\u91c7\u6837\u6b65\u9aa4\',\'\u786e\u8ba4\u6b65\u9aa4\']);for(const t of segments(c,id).filter(s=>s.floor===selectedFloor)){const tr=row(tb,[`S${t.id} \xb7 ${label(t)}`,`${t.start_step}\u2013${t.end_step}`,`${fmt(t.distance_m)} m`,t.sample_steps.join(\' / \'),[...new Set(t.confirmation_steps||[])].join(\' / \')]);tr.dataset.mode=id;tr.dataset.sid=t.id;tr.className=\'clickable\'+(t.id===sid?\' selected\':\'\');tr.tabIndex=0;tr.setAttribute(\'role\',\'button\');tr.onclick=()=>selectSegment(id,t.id);tr.onkeydown=ev=>{if(ev.key===\'Enter\'||ev.key===\' \'){ev.preventDefault();selectSegment(id,t.id)}}}\nconst box=$(\'segment-detail\');box.replaceChildren();const head=el(\'div\',null,box,\'detail-head\');el(\'h3\',`${e.title} / S${sid} \xb7 ${label(s)}`,head);el(\'span\',(c.instruction_targets?.dynamic_regions||[]).includes(s.region)?\'\u672c\u6761\u52a8\u6001\u7c7b\u522b\':\'\u56fa\u5b9a base \u7c7b\u522b\',head,\'chip\');el(\'p\',`${s.start_step}\u2013${s.end_step} \u6b65 \xb7 \u697c\u5c42 ${floorNo(s.floor)} \xb7 \u6c34\u5e73\u8def\u7a0b ${fmt(s.distance_m)} m \xb7 ${s.sample_steps.length} \u6b21\u91c7\u6837`,box,\'detail-stats\');if(e.retro)el(\'p\',\'\u4ee5\u4e0b\u4fdd\u7559\u5404\u65f6\u523b\u5b9e\u9645\u8f93\u51fa\uff1b\u672c\u56fe\u7684\u5386\u53f2\u6807\u7b7e\u53ef\u80fd\u5728\u540e\u7eed\u786e\u8ba4\u540e\u56de\u6807\u3002\u8bf7\u540c\u65f6\u770b\u201c\u56fe\u4e0a\u6700\u7ec8\u6807\u7b7e\u201d\u548c\u201c\u5f53\u65f6\u5b9e\u65f6\u6807\u7b7e\u201d\u3002\',box,\'warning\');\nconst evidence=table(box,[\'\u6b65\u9aa4\',\'\u539f\u59cb Top1\',\'\u5f53\u65f6\u5b9e\u65f6\u6807\u7b7e\',\'\u56fe\u4e0a\u6700\u7ec8\u6807\u7b7e\',\'\u5206\u5dee / \u652f\u6301\',\'\u5019\u9009\u72b6\u6001 / \u5224\u65ad\u539f\u56e0\',\'\u5206\u6570\']);for(const r of matchedSamples(c,s,id)){const st=r.stability||{},tr=row(evidence,[r.step,r.raw_region,r.region,s.region,`${fmt(r.raw_margin,4)} / ${r.raw_view_agreement}/4`,`${reasons[st.reason]||st.reason||\'\u2014\'}${st.pending_region?\'\uff1b\u5019\u9009 \'+st.pending_region+\' \xd7\'+st.pending_count:\'\'}${st.discarded_candidate?\'\uff1b\u53d6\u6d88 \'+st.discarded_candidate.region:\'\'}`]);tr.children[5].className=\'reason\';const td=el(\'td\',null,tr),d=el(\'details\',null,td,\'score-row\');el(\'summary\',\'\u5c55\u5f00\u5206\u6570\',d);const scoreBody=table(d,[\'\u7c7b\u522b\',\'\u5f53\u524d\u539f\u59cb\',\'EMA\']);(r.score_details||[]).forEach(v=>row(scoreBody,[v.name,fmt(v.raw,6),fmt(v.smoothed,6)]));const view=el(\'div\',null,d,\'directions\');for(const v of r.per_view||[]){const a=el(\'div\',null,view,\'direction\');el(\'span\',dirs[v.direction]||v.direction,a);el(\'b\',v.region,a);el(\'span\',fmt(v.cosine,4),a)}}\nconst transitions=(e.results||[]).filter(r=>r.step>=s.start_step&&r.step<=s.end_step&&r.stability?.switched);if(transitions.length)el(\'p\',\'\u672c\u6bb5\u5207\u6362\u8bb0\u5f55\uff1a\'+transitions.map(r=>`\u7b2c ${r.step} \u6b65 ${r.stability.previous_region} \u2192 ${r.region}${r.stability.candidate_start_step!=null?\'\uff1b\u5019\u9009\u9996\u6b21\u51fa\u73b0\u4e8e\u7b2c \'+r.stability.candidate_start_step+\' \u6b65\':\'\'}`).join(\'\uff1b\'),box,\'notice\');el(\'p\',\'\u539f\u59cb Top1\u3001\u6295\u7968\u548c\u5206\u5dee\u5c5e\u4e8e\u672c\u7b56\u7565\u7684\u8f93\u5165\uff1b\u76f8\u4f3c\u5ea6\u4e0d\u662f\u6982\u7387\u3002\u623f\u95f4\u5206\u6bb5\u6ca1\u6709\u771f\u503c\u5224\u5206\uff0c\u4fdd\u7559\u53ef\u68c0\u67e5\u7684\u8bc1\u636e\u4f9b\u4eba\u5de5\u6bd4\u8f83\u3002\',box,\'notice\');\n}\ndrawList();if(cases.length)buildCase(cases[0]);else{const panel=el(\'section\',null,$(\'content\'),\'panel empty\');el(\'h2\',\'\u7b49\u5f85\u7b2c\u4e00\u6761\u8f68\u8ff9\',panel);el(\'p\',\'\u603b\u7f51\u9875\u5df2\u5efa\u7acb\uff0c\u8f68\u8ff9\u4fdd\u5b58\u540e\u66f4\u65b0\u3002\u8fd0\u884c\u4e2d\u5237\u65b0\u6216\u91cd\u65b0\u6253\u5f00\u672c\u6587\u4ef6\u67e5\u770b\u8fdb\u5ea6\u3002\',panel)}\nif(warnings?.length)el(\'p\',`\u6709 ${warnings.length} \u4e2a\u6587\u4ef6\u6682\u672a\u8bfb\u5230\uff0c\u5df2\u5c55\u793a\u5176\u4f59\u53ef\u7528\u7ed3\u679c\u3002`,$(\'content\'),\'warning\');\n</script></body></html>'


# chunk_mapping.py
import base64
import copy
import colorsys
import html
import io
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import numpy as np
from PIL import Image, ImageDraw
MAP_RENDER_VERSION = 12

def completed_current_map(path):
    path = Path(path)
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding='utf-8'))
    return bool(state.get('complete') and state.get('map_render_version') == MAP_RENDER_VERSION and (state.get('target_selection_policy') == TARGET_SELECTION_POLICY))

def segment_color(sid):
    rgb = colorsys.hsv_to_rgb((sid - 1) * 0.61803398875 % 1.0, 0.85, 0.8)
    return '#' + ''.join(('%02x' % round(x * 255) for x in rgb))

def metric_depth(raw, cfg):
    d = np.asarray(raw, dtype=np.float32).squeeze().copy()
    upper = 1.0 if cfg.normalize_depth else float(cfg.max_depth)
    valid = np.isfinite(d) & (d > 0) & (d < upper)
    if cfg.normalize_depth:
        d = d * (cfg.max_depth - cfg.min_depth) + cfg.min_depth
    valid &= d >= max(0.1, float(cfg.min_depth))
    d[~valid] = np.nan
    return d

def capture_four(env, sensors):
    """Virtual capture only; exact agent + sensor state restored, no action consumed."""
    import quaternion
    sim = env.sim
    saved = copy.deepcopy(sim.get_agent_state())
    prev = getattr(sim, '_prev_sim_obs', None)
    rgb_key = getattr(sensors.rgb_sensor, 'uuid', 'rgb')
    depth_key = getattr(sensors.depth_sensor, 'uuid', 'depth')
    views = []
    try:
        for yaw in (0.0, 90.0, -90.0, 180.0):
            rotation = saved.rotation * quaternion.from_rotation_vector([0.0, math.radians(yaw), 0.0])
            raw = sim.get_observations_at(saved.position, rotation, keep_agent_at_new_pose=True)
            state = sim.get_agent_state()
            sensor = state.sensor_states[depth_key]
            matrix = quaternion.as_rotation_matrix(sensor.rotation)
            if (-matrix[:, 2])[1] >= -0.1:
                raise ValueError('Depth camera must look downward 30 degrees; set sensor orientation before Env creation')
            views.append(dict(rgb=np.asarray(raw[rgb_key])[..., :3].copy(), depth=metric_depth(raw[depth_key], sensors.depth_sensor), position=np.asarray(sensor.position).copy(), rotation=matrix.copy(), hfov=float(sensors.depth_sensor.hfov)))
    finally:
        sim.get_agent(0).set_state(saved, reset_sensors=False, infer_sensor_states=False)
        if hasattr(sim, '_prev_sim_obs'):
            sim._prev_sim_obs = prev
    return views

def project_view(view, stride=1):
    d = view['depth']
    h, w = d.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = d[v, u]
    valid = np.isfinite(z) & (z > 0)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    f = w / (2 * math.tan(math.radians(view['hfov']) / 2))
    camera = np.column_stack(((u - (w - 1) / 2) * z / f, ((h - 1) / 2 - v) * z / f, -z))
    points = camera @ view['rotation'].T + view['position']
    rgb = np.asarray(Image.fromarray(view['rgb']).resize((w, h), Image.Resampling.BILINEAR))[v, u]
    return (points, rgb)

class PointMap:
    """Pinhole depth backprojection followed by a top-surface XZ raster.

    Each map cell keeps a measured surface point and its RGB, not a 3D voxel
    mean or a fitted surface. No HSGM, triangulation, interpolation or hole fill.
    """

    def __init__(self, pixel=0.025, stride=1, lower=-0.4):
        self.pixel = pixel
        self.stride = stride
        self.lower = float(lower)
        self.floors = []

    def floor_id(self, height):
        if not self.floors or min((abs(f['height'] - height) for f in self.floors)) > 0.55:
            self.floors.append(dict(height=float(height), cells={}))
        return min(range(len(self.floors)), key=lambda i: abs(self.floors[i]['height'] - height))

    def add(self, views, height):
        fid = self.floor_id(height)
        floor = self.floors[fid]
        for view in views:
            xyz, rgb = project_view(view, self.stride)
            self.add_points(fid, xyz, rgb, view['position'])
        return fid

    def add_points(self, fid, xyz, rgb, camera_position=None):
        """Write directly projected samples; topmost measured surface wins.

        Equal-height samples prefer the nearer observation, with the newest RGB
        breaking an exact tie. Height spans only support display boundary strokes.
        """
        floor = self.floors[fid]
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        rgb = np.asarray(rgb).reshape(-1, 3)
        rel = xyz[:, 1] - floor['height']
        valid = np.isfinite(xyz).all(1) & (rel >= self.lower) & (rel < 1.8)
        xyz, rgb = (xyz[valid], rgb[valid])
        if not len(xyz):
            return
        distance = np.sum((xyz - np.asarray(camera_position)) ** 2, axis=1) if camera_position is not None else np.zeros(len(xyz))
        keys, inverse = np.unique(np.floor(xyz[:, [0, 2]] / self.pixel).astype(np.int64), axis=0, return_inverse=True)
        count = np.bincount(inverse)
        low = np.full(len(keys), np.inf)
        high = np.full(len(keys), -np.inf)
        np.minimum.at(low, inverse, xyz[:, 1])
        np.maximum.at(high, inverse, xyz[:, 1])
        order = np.lexsort((-distance, xyz[:, 1], inverse))
        chosen = order[np.r_[inverse[order][1:] != inverse[order][:-1], True]]
        for cell, index in enumerate(chosen):
            key = tuple(map(int, keys[cell]))
            old = floor['cells'].get(key)
            if old is None:
                floor['cells'][key] = [float(xyz[index, 1]), rgb[index].copy(), float(distance[index]), float(low[cell]), float(high[cell]), int(count[cell])]
            else:
                if xyz[index, 1] > old[0] or (xyz[index, 1] == old[0] and distance[index] <= old[2]):
                    old[:3] = [float(xyz[index, 1]), rgb[index].copy(), float(distance[index])]
                old[3] = min(old[3], float(low[cell]))
                old[4] = max(old[4], float(high[cell]))
                old[5] += int(count[cell])

    def render(self, fid, route, segments, max_size=1800, grid=None, draw_overlays=True):
        floor = self.floors[fid]
        cells = floor['cells']
        pixel = self.pixel
        keys = np.asarray(list(cells), dtype=np.int64).reshape(-1, 2)
        rows = [r for r in route if r['floor'] == fid]
        route_xz = np.asarray([r['position'] for r in rows], float).reshape(-1, 3)[:, [0, 2]]
        parts = [route_xz]
        if len(keys):
            parts.extend([keys * pixel, (keys + 1) * pixel])
        bounds = np.concatenate(parts)
        if not len(bounds):
            bounds = np.zeros((1, 2))
        lo = bounds.min(0) - 0.3
        hi = bounds.max(0) + 0.3
        pitch = max(pixel, float(np.max(hi - lo)) / max(1, max_size - 2))
        origin = np.floor(lo / pitch).astype(np.int64)
        wh = np.maximum(1, np.floor(hi / pitch).astype(np.int64) - origin + 1)
        width, height = map(int, wh)
        if grid is not None:
            pitch = float(grid['meters_per_pixel'])
            origin = np.rint(np.asarray(grid['origin_xz']) / pitch).astype(np.int64)
            width, height = map(int, grid['image_size'])

        def transform(points):
            return np.floor(np.asarray(points) / pitch).astype(np.int64) - origin
        rgb = np.full((height, width, 3), 255, np.uint8)
        gray = np.full_like(rgb, 255)
        if len(keys):
            entries = list(cells.values())
            levels = np.asarray([entry[0] for entry in entries])
            distances = np.asarray([entry[2] for entry in entries])
            uv = np.floor((keys + 0.5) * (pixel / pitch)).astype(np.int64) - origin
            order = np.lexsort((-distances, levels, uv[:, 0], uv[:, 1]))
            sorted_uv = uv[order]
            visible = order[np.r_[np.any(sorted_uv[1:] != sorted_uv[:-1], axis=1), True]]
            colors = np.asarray([entries[i][1] for i in visible])
            colors = np.clip(colors, 0, 255).astype(np.uint8)
            u, v = uv[visible].T
            rgb[v, u] = colors
            occupied = levels[visible] - floor['height'] > 0.2
            gray[v, u] = np.where(occupied[:, None], 0, 200)
            obstacle = np.flatnonzero(occupied)
            if len(obstacle):
                offsets = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
                targets = (uv[visible[obstacle], None, :] + offsets).reshape(-1, 2)
                source = np.repeat(obstacle, len(offsets))
                distance = np.tile(np.sum(offsets ** 2, axis=1), len(obstacle))
                good = (targets[:, 0] >= 0) & (targets[:, 0] < width) & (targets[:, 1] >= 0) & (targets[:, 1] < height)
                targets, source, distance = (targets[good], source[good], distance[good])
                flat = targets[:, 1] * width + targets[:, 0]
                order = np.lexsort((levels[visible[source]], -distance, flat))
                selected = order[np.r_[flat[order][1:] != flat[order][:-1], True]]
                x, y = targets[selected].T
                rgb[y, x] = colors[source[selected]]
                gray[y, x] = 0
            vertical = np.asarray([entry[4] - entry[3] >= 0.5 and entry[5] >= 3 for entry in entries])
            wall_uv = uv[vertical]
            if len(wall_uv):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) + abs(dy) > 2:
                            continue
                        x, y = (wall_uv + [dx, dy]).T
                        good = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                        rgb[y[good], x[good]] = (35, 35, 35)
                        gray[y[good], x[good]] = 0
        images = [Image.fromarray(rgb), Image.fromarray(gray)]
        geometry = dict(origin_xz=(origin * pitch).tolist(), meters_per_pixel=pitch, image_size=[width, height], rasterization='world_aligned_floor_index', projection='direct_depth_backprojection_top_surface', point_stride=self.stride, height_cut_relative_m=[self.lower, 1.8], obstacle_stroke_radius_pixels=1, boundary_stroke_radius_pixels=2)
        return (self.annotate(images, fid, route, segments, geometry) if draw_overlays else images, geometry)

    @staticmethod
    def annotate(background, fid, route, segments, geometry):
        """Overlay two label strategies on one shared raster, without projecting again."""
        images = [image.copy() for image in background]
        pitch = geometry['meters_per_pixel']
        origin = np.rint(np.asarray(geometry['origin_xz']) / pitch).astype(np.int64)

        def transform(points):
            return np.floor(np.asarray(points) / pitch).astype(np.int64) - origin
        rows = [r for r in route if r['floor'] == fid]
        by_step = {r['step']: r for r in rows}
        for image in images:
            pen = ImageDraw.Draw(image)
            for s in segments:
                if s['floor'] != fid:
                    continue
                points = [r for r in rows if s['start_step'] <= r['step'] <= s['end_step']]
                for a, b in zip(points, points[1:]):
                    if b['step'] == a['step'] + 1:
                        pen.line(list(map(tuple, transform(np.asarray([a['position'], b['position']])[:, [0, 2]]))), fill=s['color'], width=3)
                r = by_step.get(s['end_step'])
                label_position = s.get('label_position', r['position'] if r else None)
                if label_position is not None and s['floor'] == fid:
                    x, y = transform(np.asarray(label_position)[[0, 2]])
                    pen.text((x + 4, y + 3), 'S%d' % s['id'], fill=s['color'], stroke_width=1, stroke_fill='white')
            if rows:
                x, y = transform(np.asarray(rows[-1]['position'])[[0, 2]])
                pen.ellipse((x - 3, y - 3, x + 3, y + 3), fill='red', outline='white')
        return images

class CLIPClient:

    def __init__(self, python, model, vocab, tau=1.5, strength=1.0, minimum=0.02, timeout=600):
        self.timeout = timeout
        self.queue = queue.Queue()
        self.categories = None
        self.compare_depth = True
        self.temp = tempfile.TemporaryDirectory(prefix='chunk_clip_')
        command = [str(python), '-u', str(Path(__file__).resolve()), '--clip-worker', '--model', str(model), '--vocab', str(vocab), '--tau', str(tau), '--strength', str(strength), '--min_weight', str(minimum)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

        def reader():
            for line in self.process.stdout:
                try:
                    self.queue.put(json.loads(line))
                except ValueError:
                    print('[clip] ' + line.rstrip(), file=sys.stderr)
            self.queue.put(dict(ok=False, error='CLIP process exited'))
        threading.Thread(target=reader, daemon=True).start()
        try:
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            reply = self.queue.get(timeout=self.timeout)
        except queue.Empty:
            raise RuntimeError('CLIP worker timeout') from None
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error', 'CLIP worker failed'))
        return reply

    def classify(self, views):
        packet = Path(self.temp.name) / 'views.npz'
        data = {}
        for i, v in enumerate(views):
            data['rgb%d' % i] = v['rgb']
            data['depth%d' % i] = v['depth']
        np.savez_compressed(packet, **data)
        self.process.stdin.write(json.dumps(dict(packet=str(packet), categories=self.categories, compare_depth=self.compare_depth)) + '\n')
        self.process.stdin.flush()
        return self.read()['result']

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
        self.temp.cleanup()

class ChunkMapper:
    """Live adapter: record EVERY executed pose; call observe only at chunk boundaries.

    initialize(env,sensors) at step0; record_pose(step, position) after each action;
    update(env,sensors,step,final=False) after 4 actions or an early STOP. Observation
    classifies CURRENT endpoint and labels the preceding interval. Adjacent
    intervals with the same category/floor form one colored run. This is not
    proof of an exact physical room boundary or a unique room instance.
    """

    def __init__(self, client, folder, name, instruction='', pixel=0.025, stride=1, region_momentum=0.6, region_stay_bonus=0.0, region_fast_margin=0.02):
        self.client = client
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.instruction = instruction
        self.map = PointMap(pixel, stride)
        self.stabilizer = RegionStabilizer(region_momentum, region_stay_bonus, region_fast_margin)
        self.camera_settings = {}
        self.latest_views = None
        self.route = []
        self.segments = []
        self.initial_region = None
        self.last_update = 0
        self.complete = False
        self.observations = []
        self.raw_segments = []
        self.instruction_targets = None

    def record_pose(self, step, position):
        position = np.asarray(position, float)
        fid = self.map.floor_id(position[1])
        if self.route and step <= self.route[-1]['step']:
            raise ValueError('Pose steps must increase strictly')
        self.route.append(dict(step=int(step), floor=fid, position=position.tolist()))

    def initialize(self, env, sensors):
        self.record_pose(0, env.sim.get_agent_state().position)
        views = capture_four(env, sensors)
        self.add_views(views, initial=True)
        self.initial_region = self.stabilizer.update(self.client.classify(views), self.route[-1]['floor'])
        self.observations.append(dict(step=0, floor=self.route[-1]['floor'], position=self.route[-1]['position'].copy(), **self.initial_region))
        self.save()

    def add_views(self, views, initial=False):
        self.latest_views = views
        self.camera_settings = dict(hfov=views[0]['hfov'], pitch_degrees=-30, depth_image_size=list(views[0]['depth'].shape[::-1]), rgb_image_size=list(views[0]['rgb'].shape[1::-1]))
        fid = self.map.floor_id(self.route[-1]['position'][1])
        for view in views:
            xyz, rgb = project_view(view, self.map.stride)
            self.map.add_points(fid, xyz, rgb, view['position'])
        return fid

    def update(self, env, sensors, step, final=False):
        if step <= self.last_update:
            raise ValueError('Duplicate/out-of-order chunk boundary')
        if not final and step - self.last_update != 4:
            raise ValueError('Update once per FOUR executed actions')
        if not self.route or self.route[-1]['step'] != step:
            raise ValueError('Record every executed pose before update')
        views = capture_four(env, sensors)
        fid = self.add_views(views)
        result = self.stabilizer.update(self.client.classify(views), fid)
        self.observations.append(dict(step=step, floor=fid, position=self.route[-1]['position'].copy(), **result))
        previous = self.segments[-1] if self.segments else None
        if previous and previous['region'] == result['region'] and (previous['floor'] == fid):
            segment = previous
            segment.update(result)
            segment.update(end_step=step, sampled_at_step=step)
            segment['sample_steps'].append(step)
        else:
            sid = len(self.segments) + 1
            segment = dict(id=sid, start_step=self.last_update, end_step=step, sampled_at_step=step, floor=fid, color=region_color(result['region']), sample_steps=[step], **result)
            if previous:
                segment['transition_step_window'] = [self.last_update, step]
            self.segments.append(segment)
        segment['sample_count'] = len(segment['sample_steps'])
        segment['current_position'] = self.route[-1]['position'].copy()
        points = np.asarray([r['position'] for r in self.route if r['floor'] == fid and segment['start_step'] <= r['step'] <= segment['end_step']])
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1))]
        segment['label_position'] = points[np.argmin(np.abs(lengths - lengths[-1] / 2))].tolist()
        raw = dict(result, region=result['raw_region'], cosine=result['raw_cosine'], view_agreement=result['raw_view_agreement'])
        prior = self.raw_segments[-1] if self.raw_segments else None
        if prior and prior['region'] == raw['region'] and (prior['floor'] == fid):
            prior.update(raw)
            prior.update(end_step=step, sampled_at_step=step)
            prior['sample_steps'].append(step)
        else:
            prior = dict(id=len(self.raw_segments) + 1, start_step=self.last_update, end_step=step, sampled_at_step=step, floor=fid, color=region_color(raw['region']), sample_steps=[step], **raw)
            self.raw_segments.append(prior)
        prior['sample_count'] = len(prior['sample_steps'])
        pts = np.asarray([r['position'] for r in self.route if r['floor'] == fid and prior['start_step'] <= r['step'] <= step])
        prior['label_position'] = pts[len(pts) // 2].tolist()
        self.last_update = step
        self.complete = bool(final)
        self.save()

    def save(self):
        maps = []
        for fid, floor in enumerate(self.map.floors):
            background, geometry = self.map.render(fid, self.route, [], draw_overlays=False)
            stable = PointMap.annotate(background, fid, self.route, self.segments, geometry)
            raw = PointMap.annotate(background, fid, self.route, self.raw_segments, geometry)
            groups = {}
            background_files = {}
            for kind, image in zip(('rgb', 'passability'), background):
                name = 'background_%s_%02d.png' % (kind, fid + 1)
                path = self.folder / name
                temporary = path.with_suffix('.tmp')
                image.save(temporary, format='PNG', compress_level=3)
                temporary.replace(path)
                background_files[kind] = name
            for mode, images in (('stable', stable), ('raw', raw)):
                files = {}
                for kind, image in zip(('rgb', 'passability'), images):
                    prefix = '' if mode == 'stable' else 'raw_'
                    name = '%s%s_%02d.png' % (prefix, kind, fid + 1)
                    path = self.folder / name
                    temp = path.with_suffix('.tmp')
                    image.save(temp, format='PNG', compress_level=3)
                    temp.replace(path)
                    files[kind] = name
                groups[mode] = files
            maps.append(dict(floor=fid, height=floor['height'], geometry=geometry, images=groups['stable'], comparisons=dict(raw=groups['raw']), background_images=background_files, views=4, observed_area_m2=len(floor['cells']) * self.map.pixel ** 2, shared_geometry=True))
        contact = None
        if self.latest_views:
            sheet = Image.new('RGB', (640, 520), 'white')
            pen = ImageDraw.Draw(sheet)
            for i, (view, title) in enumerate(zip(self.latest_views, ('Front', 'Left', 'Right', 'Back'))):
                image = Image.fromarray(view['rgb']).convert('RGB')
                image.thumbnail((320, 236))
                x, y = (i % 2 * 320, i // 2 * 260)
                sheet.paste(image, (x + (320 - image.width) // 2, y + 24))
                pen.text((x + 8, y + 6), title, fill='black')
            contact = 'latest_views.jpg'
            path = self.folder / contact
            temp = path.with_suffix('.tmp')
            sheet.save(temp, format='JPEG', quality=90)
            temp.replace(path)
        settings = self.stabilizer.settings()
        experiments = build_experiments(self.observations, self.route, (self.instruction_targets or {}).get('base_regions', []), settings['momentum'], settings['fast_margin'])
        atomic_json(self.folder / 'trajectory.json', dict(name=self.name, instruction=self.instruction, complete=self.complete, map_render_version=MAP_RENDER_VERSION, report_layout='four_view_region_experiments_v1', mapping_views=4, map_branches=1, replay_passes=1, camera_pitch_degrees=-30, update_interval_actions=4, initial_region=self.initial_region, region_stabilization=settings, region_comparisons=experiments, camera_settings=self.camera_settings, latest_views_image=contact, color_semantics='same region category always has the same color', region_semantics='four-view endpoint predictions label preceding intervals; adjacent same-label intervals on the same floor merge; switch onset is localized only to an observation interval', instruction_targets=self.instruction_targets, target_selection_policy=self.instruction_targets.get('selection_policy') if self.instruction_targets else None, route=self.route, segments=self.segments, raw_segments=self.raw_segments, region_observations=self.observations, maps=maps))


# replay_environment.py
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'config/vln_r2r.yaml').is_file()), Path.cwd())
_BACKWARD = False
ACTION_KEYS = {0: 'stop', 1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 4: 'governor_move_backward'}

def json_values(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    decoder, offset = (json.JSONDecoder(), 0)
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        value, offset = decoder.raw_decode(text, offset)
        yield value

def result_rows(value):
    if isinstance(value, list):
        for item in value:
            yield from result_rows(item)
    elif isinstance(value, dict):
        if 'episode_id' in value and 'scene_id' in value:
            yield value
        else:
            for key in ('episodes', 'results', 'records', 'data'):
                if isinstance(value.get(key), (list, dict)):
                    yield from result_rows(value[key])

def scene_name(value):
    path = Path(str(value).replace('\\', '/'))
    return path.parent.name if path.suffix else path.name

def trajectory_key(path):
    match = re.fullmatch('(.+?)_(\\d+)(?:_[0-9a-f]{12})?', path.parent.name)
    if not match:
        raise ValueError('Cannot infer scene/episode from ' + path.parent.name)
    return (match[1], str(int(match[2])))

def natural_key(path):
    return [int(s) if s.isdecimal() else s.lower() for s in re.split('(\\d+)', path.as_posix())]

def select_jobs(args):
    outcomes = {}
    for value in json_values(args.baseline_result_path):
        for row in result_rows(value):
            metrics = row.get('metrics')
            success = row.get('success')
            if success is None and isinstance(metrics, dict):
                success = metrics.get('success')
            if success not in (0, 1) or isinstance(success, str):
                continue
            key = (scene_name(row['scene_id']), str(int(row['episode_id'])))
            if key in outcomes and outcomes[key] != bool(success):
                raise ValueError('Conflicting success labels for ' + str(key))
            outcomes[key] = bool(success)
    root = Path(args.records_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Records directory does not exist: ' + str(root))
    paths = sorted(root.rglob('actions.jsonl'), key=natural_key)
    if args.start_folder:
        matches = [i for i, p in enumerate(paths) if p.parent.name == args.start_folder]
        if len(matches) != 1:
            raise ValueError('--start_folder must identify one trajectory folder')
        paths = paths[matches[0]:]
    groups = {True: [], False: []}
    unknown = 0
    seen = set()
    for path in paths:
        key = trajectory_key(path)
        if key not in outcomes:
            unknown += 1
            continue
        if key in seen:
            raise ValueError('Duplicate action trajectories for scene/episode: ' + str(key))
        seen.add(key)
        groups[outcomes[key]].append(dict(name=path.parent.name, scene=key[0], episode_id=key[1], success=outcomes[key], actions_file=str(path.resolve())))
    for success, count in ((True, args.success_count), (False, args.failure_count)):
        if len(groups[success]) < count:
            raise ValueError('Need %d %s trajectories but found %d; unknown outcomes=%d' % (count, 'successful' if success else 'failed', len(groups[success]), unknown))
    jobs = groups[True][:args.success_count] + groups[False][:args.failure_count]
    print('[test-label] selected success=%d failure=%d; unknown outcomes skipped=%d' % (args.success_count, args.failure_count, unknown), flush=True)
    return jobs

def load_actions(path, max_steps):
    rows = []
    previous = None
    for value in json_values(path):
        if not isinstance(value, dict) or type(value.get('action')) is not int or value['action'] not in ACTION_KEYS:
            raise ValueError('Each actions.jsonl row must contain integer action 0..4')
        index = value.get('step_id', len(rows))
        if type(index) is not int or (previous is not None and index <= previous):
            raise ValueError('Action step_id must increase strictly')
        previous = index
        rows.append(value)
        if value['action'] == 0:
            break
    if not rows:
        raise ValueError('Empty action file: ' + str(path))
    truncated = bool(max_steps and len(rows) > max_steps)
    return (rows[:max_steps] if max_steps else rows, truncated)

def make_environment(args, job, count):
    global _BACKWARD
    sys.path[:0] = [str(ROOT / 'streamvln'), str(ROOT)]
    if importlib.util.find_spec('habitat') is None or importlib.util.find_spec('habitat_baselines') is None:
        root = Path(args.habitat_root)
        sys.path[:0] = [str(root / 'habitat-lab'), str(root / 'habitat-baselines')]
    import habitat
    from habitat_baselines.config.default import get_config
    importlib.import_module('habitat_extensions')
    importlib.import_module('habitat_extensions.measures')
    if not _BACKWARD:
        import habitat_sim
        from habitat.core.embodied_task import SimulatorTaskAction
        from habitat.core.registry import registry

        @registry.register_task_action
        class LabelReplayBackward(SimulatorTaskAction):

            def step(self, *unused, **kwargs):
                self._sim.get_agent(0).agent_config.action_space['governor_move_backward'] = habitat_sim.ActionSpec('move_backward', habitat_sim.ActuationSpec(amount=0.25))
                return self._sim.step('governor_move_backward')
        _BACKWARD = True
    from habitat.config.default_structured_configs import ActionConfig
    config = get_config(args.habitat_config_path)
    with habitat.config.read_write(config):
        cfg = config.habitat
        cfg.seed = args.seed
        cfg.dataset.split = args.eval_split
        if args.data_path:
            cfg.dataset.data_path = args.data_path
        cfg.dataset.content_scenes = ['*']
        cfg.task.measurements.clear()
        cfg.task.actions['governor_move_backward'] = ActionConfig(type='LabelReplayBackward')
        cfg.environment.max_episode_steps = count + 2
        cfg.environment.max_episode_seconds = 0
        cfg.simulator.turn_angle = 15
        cfg.simulator.forward_step_size = 0.25
        cfg.simulator.habitat_sim_v0.gpu_device_id = 0
        cfg.simulator.habitat_sim_v0.enable_physics = False
        agent = cfg.simulator.agents[cfg.simulator.agents_order[cfg.simulator.default_agent_id]]
        sensors = agent.sim_sensors
        if 'rgb_sensor' not in sensors or 'depth_sensor' not in sensors:
            raise ValueError('Need RGB and depth sensors')
        sensors.rgb_sensor.orientation = [-math.pi / 6, 0.0, 0.0]
        sensors.depth_sensor.orientation = [-math.pi / 6, 0.0, 0.0]
        if abs(sensors.rgb_sensor.hfov - sensors.depth_sensor.hfov) > 0.0001 or not np.allclose(sensors.rgb_sensor.position, sensors.depth_sensor.position) or (not np.allclose(sensors.rgb_sensor.orientation, sensors.depth_sensor.orientation)):
            raise ValueError('RGB/depth intrinsics/extrinsics must be aligned')
    dataset = habitat.make_dataset(cfg.dataset.type, config=cfg.dataset)
    matches = [e for e in dataset.episodes if str(e.episode_id) == job['episode_id'] and (scene_name(e.scene_id) == job['scene'] or Path(e.scene_id).stem == job['scene'])]
    if len(matches) != 1:
        raise ValueError('Expected one Habitat episode for ' + job['name'])
    dataset.episodes = matches
    env = habitat.Env(config=config, dataset=dataset)
    env.seed(args.seed)
    env.current_episode = matches[0]
    return (env, sensors, matches[0])


# replay_workers.py
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

def replay_prepared(args, jobs, prepared, output, publish=True):
    client = None

    def report():
        if publish:
            write_report(output)
    try:
        if prepared:
            client = CLIPClient(args.clip_python, args.clip_model, output / '.cache/clip_vocabulary.json', args.depth_tau, args.depth_strength, args.depth_min_weight)
            client.compare_depth = not getattr(args, 'skip_depth_comparison', False)
        for i, job in enumerate(jobs, 1):
            folder = output / job['name']
            marker = folder / 'trajectory.json'
            if completed_current_map(marker) or job['name'] not in prepared:
                continue
            item = prepared[job['name']]
            rows = item['rows']
            client.categories = list(item['targets']['region_vocabulary'])
            env = None
            mapper = None
            try:
                env, sensors, _ = make_environment(args, job, len(rows))
                print('[replay] GPU %s %d/%d %s' % (args.gpu, i, len(jobs), job['name']), flush=True)
                env.reset()
                mapper = ChunkMapper(client, folder, job['name'], item['instruction'], args.pixel_size, args.point_stride, args.region_momentum, args.region_stay_bonus, args.region_fast_margin)
                mapper.instruction_targets = item['targets']
                mapper.initialize(env, sensors)
                report()
                for step, row in enumerate(rows, 1):
                    if env.episode_over:
                        break
                    env.step({'action': ACTION_KEYS[row['action']]})
                    mapper.record_pose(step, env.sim.get_agent_state().position)
                    final = row['action'] == 0 or env.episode_over or step == len(rows)
                    if step % 4 == 0 or final:
                        mapper.update(env, sensors, step, final=final)
                        report()
                    if final:
                        break
                if not mapper.complete:
                    mapper.complete = True
                    mapper.save()
                state = json.loads(marker.read_text())
                state['action_limit_truncated'] = item['truncated']
                atomic_json(marker, state)
                print('[replay] completed ' + job['name'], flush=True)
            finally:
                client.categories = list(item['targets']['base_regions'])
                try:
                    if mapper is not None and (not mapper.complete):
                        mapper.save()
                finally:
                    if env is not None:
                        env.close()
                    report()
    finally:
        if client is not None:
            client.close()
        report()
    return 0

def replay_worker_main(manifest):
    data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    args = SimpleNamespace(**data['args'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.chdir(data['root'])
    return replay_prepared(args, data['jobs'], data['prepared'], Path(data['output']), publish=False)

def run_replay_pool(args, jobs, prepared, output):
    gpus = [g.strip() for g in (args.gpus or args.gpu).split(',')]
    workers = args.workers or len(gpus)
    jobs = [j for j in jobs if j['name'] in prepared and (not completed_current_map(output / j['name'] / 'trajectory.json'))]
    if not jobs:
        return 0
    workers = min(workers, len(jobs))
    if workers == 1:
        args.gpu = gpus[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        return replay_prepared(args, jobs, prepared, output)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    cache = output / '.cache/workers'
    cache.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    previous = None
    seen = set()
    try:
        for i in range(workers):
            gpu = gpus[i % len(gpus)]
            assigned = jobs[i::workers]
            settings = dict(vars(args), gpu=gpu)
            manifest = cache / ('worker_%02d.json' % i)
            atomic_json(manifest, dict(args=settings, jobs=assigned, prepared={j['name']: prepared[j['name']] for j in assigned}, root=str(Path.cwd()), output=str(output)))
            logfile = logs / ('worker_%02d_gpu_%s.log' % (i, gpu))
            handle = logfile.open('w', encoding='utf-8')
            handles.append(handle)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env[key] = '1'
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--replay-worker', str(manifest)], stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=os.name != 'nt')
            processes.append((process, logfile))
            print('[replay] worker=%d GPU=%s trajectories=%d log=%s' % (i, gpu, len(assigned), logfile), flush=True)
        while True:
            snapshot = tuple(((p.as_posix(), p.stat().st_mtime_ns) for p in sorted(output.glob('*/trajectory.json'))))
            if snapshot != previous:
                write_report(output)
                previous = snapshot
                for job in jobs:
                    name = job['name']
                    if name not in seen and completed_current_map(output / name / 'trajectory.json'):
                        seen.add(name)
                        print('[report] %d/%d completed: %s' % (len(seen), len(jobs), name), flush=True)
            if all((p.poll() is not None for p, _ in processes)):
                break
            time.sleep(1.0)
        return 2 if any((p.returncode for p, _ in processes)) else 0
    finally:
        for process, _ in processes:
            if process.poll() is None:
                if os.name != 'nt':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
        for process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        for handle in handles:
            handle.close()
        write_report(output)


# run_chunk_map.py
import argparse
import json
import math
import os
from pathlib import Path
import sys

def instruction_for_job(args, job, rows):
    provided = [r['instruction'] for r in rows if isinstance(r.get('instruction'), str) and r['instruction'].strip()]
    if provided:
        if any((' '.join(t.split()) != ' '.join(provided[0].split()) for t in provided)):
            raise ValueError('Action records contain conflicting instructions: ' + job['name'])
        return provided[0]
    env = None
    try:
        env, _, episode = make_environment(args, job, len(rows))
        fallback = getattr(episode, 'instruction', '')
        instruction = str(getattr(fallback, 'instruction_text', fallback))
        if not instruction.strip():
            raise ValueError('Missing original instruction: ' + job['name'])
        return instruction
    finally:
        if env is not None:
            env.close()

def prepare_targets(args, jobs, output, vocab):
    """Finish and close Qwen before constructing CLIP. Cache each episode separately."""
    cache = output / '.cache' / 'targets'
    cache.mkdir(parents=True, exist_ok=True)
    prepared = {}
    failed = []
    qwen = None
    try:
        for i, job in enumerate(jobs, 1):
            marker = output / job['name'] / 'trajectory.json'
            if completed_current_map(marker):
                continue
            rows, truncated = load_actions(job['actions_file'], args.max_steps)
            instruction = instruction_for_job(args, job, rows)
            key = cache_key(instruction, vocab, args.local_vlm_model)
            path = cache / (job['name'] + '.json')
            saved = None
            if path.exists():
                candidate = json.loads(path.read_text(encoding='utf-8'))
                if candidate.get('key') == key:
                    saved = candidate
            if saved is None:
                if qwen is None:
                    print('[chunk-map] loading local Qwen3-VL-8B for instruction targets and dynamic room HEADS only', flush=True)
                    qwen = TargetClient(args.local_vlm_python, args.local_vlm_model, args.local_vlm_timeout, args.target_max_tokens)
                print('[chunk-map] targets %d/%d %s' % (i, len(jobs), job['name']), flush=True)
                try:
                    result = qwen.extract(instruction, vocab)
                except ValueError as exc:
                    atomic_json(cache / (job['name'] + '.error.json'), dict(stage='instruction_targets', reason=str(exc), pending=True))
                    failed.append(job['name'])
                    print('[chunk-map] targets pending: ' + str(exc), flush=True)
                    continue
                saved = dict(key=key, instruction=instruction, targets=result)
                atomic_json(path, saved)
            (cache / (job['name'] + '.error.json')).unlink(missing_ok=True)
            prepared[job['name']] = dict(rows=rows, truncated=truncated, **saved)
            print('[chunk-map] %s: base=%d dynamic=%s candidates=%d (no Top10)' % (job['name'], len(saved['targets']['base_regions']), ', '.join(saved['targets']['dynamic_regions']) or '(none)', len(saved['targets']['region_vocabulary'])), flush=True)
    finally:
        if qwen is not None:
            qwen.close()
            print('[chunk-map] Qwen process closed; releasing its GPU memory before CLIP', flush=True)
    return (prepared, failed)

def compatible_run_config(saved, current):
    """GPU assignment may change; semantic settings and base/dynamic policy may not."""
    left = dict(saved)
    right = dict(current)
    left.pop('execution', None)
    right.pop('execution', None)
    return left == right

def write_clip_vocabulary(output, base, prepared):
    """Encode the union once; each replay still scores ONLY its own base+dynamic list."""
    combined = list(base)
    for item in prepared.values():
        categories = item['targets']['region_vocabulary']
        if not set(base).issubset(categories):
            raise ValueError('All base classes must remain eligible in every trajectory')
        for name in categories:
            if name not in combined:
                combined.append(name)
    atomic_json(output / '.cache/base_vocabulary.json', dict(complete=True, terms=base, definitions=BASE_REGIONS))
    atomic_json(output / '.cache/clip_vocabulary.json', dict(complete=True, terms=combined, text_prompts=region_clip_prompts(combined), selection_policy=TARGET_SELECTION_POLICY, scope='encoder union only; each trajectory scores its own region_vocabulary'))
    return combined

def main():
    root = next((p for p in Path(__file__).resolve().parents if (p / 'config/vln_r2r.yaml').is_file()), Path.cwd())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='1')
    p.add_argument('--clip_python', default='/home/ubuntun/miniconda3/envs/governor_map_models/bin/python')
    p.add_argument('--gpus', help='Physical replay GPUs, comma separated, e.g. 1,5')
    p.add_argument('--workers', type=int, default=0, help='Replay process count; 0 means one per listed GPU')
    p.add_argument('--hsgm_repo', default=None, help=argparse.SUPPRESS)
    p.add_argument('--hsgm_pcd_resolution', type=float, default=0.025, help=argparse.SUPPRESS)
    p.add_argument('--hsgm_grid_resolution', type=float, default=0.05, help=argparse.SUPPRESS)
    p.add_argument('--without_hsgm', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--clip_model', default=str(root / 'weights/clip-vit-large-patch14-336'))
    p.add_argument('--vocab', help='Optional extra static room heads, appended to the always-retained MP3D base28')
    p.add_argument('--local_vlm_python', default='/home/ubuntun/miniconda3/envs/vlnpa/bin/python')
    p.add_argument('--local_vlm_model', default=DEFAULT_MODEL)
    p.add_argument('--local_vlm_timeout', type=float, default=600.0)
    p.add_argument('--target_max_tokens', type=int, default=2048)
    p.add_argument('--depth_tau', type=float, default=1.5)
    p.add_argument('--depth_strength', type=float, default=1.0)
    p.add_argument('--depth_min_weight', type=float, default=0.02)
    p.add_argument('--region_momentum', type=float, default=0.6, help='EMA weight of historical raw scores, in [0,1)')
    p.add_argument('--region_stay_bonus', type=float, default=0.0, help='Deprecated compatibility option; ignored, stay bonus is removed')
    p.add_argument('--region_fast_margin', type=float, default=0.02, help='Raw top1-top2 margin for immediate switch with >=3/4 agreeing views')
    p.add_argument('--records_root', default=str(root / 'results_base/val_unseen/streamvln_baseline_full/records'))
    p.add_argument('--baseline_result_path', default=str(root / 'results_base/val_unseen/streamvln_baseline_full/result.json'))
    p.add_argument('--success_count', type=int, default=10)
    p.add_argument('--failure_count', type=int, default=10)
    p.add_argument('--start_folder')
    p.add_argument('--max_steps', type=int, default=100)
    p.add_argument('--output_path', default='results/test_new_compare_20')
    p.add_argument('--skip_depth_comparison', action='store_true', help='Skip extra CLIP depth forwards; all saved-score comparisons still run')
    p.add_argument('--habitat_config_path', default=str(root / 'config/vln_r2r.yaml'))
    p.add_argument('--habitat_root', default='/mnt/disk8t1/zwm/Progress-Aware-VLN/habitat-lab')
    p.add_argument('--eval_split', default='val_unseen')
    p.add_argument('--data_path')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pixel_size', type=float, default=0.025)
    p.add_argument('--point_stride', type=int, default=1, help='Depth pixel stride; 1 uses every valid pixel')
    p.add_argument('--report_only', action='store_true', help='Rebuild only report.html from existing trajectory.json/maps; no replay or model loading')
    p.add_argument('--list_only', action='store_true')
    args = p.parse_args()
    output = (root / Path(args.output_path).expanduser()).resolve()
    if args.report_only:
        if not output.is_dir():
            p.error('--report_only output directory does not exist')
        write_report(output)
        print('[chunk-map] updated report only: ' + str(output / 'report.html'), flush=True)
        return 0
    if args.region_stay_bonus != 0:
        print('[chunk-map] --region_stay_bonus is obsolete and ignored; stay bonus is removed', flush=True)
    args.region_stay_bonus = 0.0
    gpu_list = [g.strip() for g in (args.gpus or args.gpu).split(',')]
    if any((not g.isdecimal() for g in gpu_list)) or len(set(gpu_list)) != len(gpu_list) or args.workers < 0:
        p.error('--gpus must contain distinct numeric physical GPU indices; workers >= 0')
    if args.gpus:
        args.gpu = gpu_list[0]
    args.hsgm_repo = None
    args.without_hsgm = True
    RegionStabilizer(args.region_momentum, args.region_stay_bonus, args.region_fast_margin)
    if not args.gpu.isdecimal() or min(args.success_count, args.failure_count, args.max_steps) < 0 or args.success_count + args.failure_count < 1 or (args.point_stride < 1) or (not math.isfinite(args.pixel_size)) or (args.pixel_size <= 0) or (not all((math.isfinite(x) for x in (args.depth_tau, args.depth_strength, args.depth_min_weight)))) or (args.depth_tau <= 0) or (args.depth_strength < 0) or (not 0 < args.depth_min_weight <= 1) or (args.target_max_tokens < 1) or (not math.isfinite(args.local_vlm_timeout)) or (args.local_vlm_timeout <= 0):
        p.error('Invalid counts/GPU/resolution/decay parameters')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.chdir(root)
    output.mkdir(parents=True, exist_ok=True)
    vocab = base_region_vocabulary(load_vocab(args.vocab) if args.vocab else ())
    jobs = select_jobs(args)
    config = dict(settings={k: v for k, v in vars(args).items() if k not in ('output_path', 'report_only', 'list_only', 'gpu', 'gpus', 'workers', 'hsgm_repo', 'without_hsgm', 'hsgm_pcd_resolution', 'hsgm_grid_resolution')}, vocab=vocab, model=str(Path(args.clip_model).resolve()), jobs=jobs, version=12, computation_scope='one_four_view_map_grouped_region_comparisons_no_bonus', execution=dict(gpus=gpu_list, workers=args.workers or len(gpu_list)), local_vlm_role='instruction_targets_and_dynamic_room_heads_only', route_audit=False, base_region_definitions=BASE_REGIONS, region_candidate_policy=TARGET_SELECTION_POLICY)
    path = output / 'config.json'
    if path.exists() and (not compatible_run_config(json.loads(path.read_text()), config)):
        raise ValueError('Settings changed; use a new output directory')
    atomic_json(path, config)
    write_report(output)
    if args.list_only:
        return 0
    print('[chunk-map] always-on MP3D base=%d; local 8B adds instruction room heads; full base+dynamic scoring; no Top10, API or deviation audit' % len(vocab), flush=True)
    print('[chunk-map] direct RGB-D backprojection; point_stride=%d; colors change only when consecutive region labels change' % args.point_stride, flush=True)
    print('[chunk-map] region stabilization: EMA momentum + strong-evidence fast switch; no stay bonus', flush=True)
    print('[chunk-map] FOUR-VIEW ONLY: one replay, one map; temporal/aggregation comparisons share saved scores; one base+dynamic vocabulary; no bonus', flush=True)
    print('[chunk-map] depth comparisons: ' + ('disabled' if args.skip_depth_comparison else 'default + tau 0.75/3 m + no decay; sequential forwards using one loaded CLIP'), flush=True)
    prepared, failed = prepare_targets(args, jobs, output, vocab)
    write_clip_vocabulary(output, vocab, prepared)
    status = run_replay_pool(args, jobs, prepared, output)
    print('[chunk-map] report: ' + str(output / 'report.html'), flush=True)
    if failed:
        print('[chunk-map] instruction extraction pending for: ' + ', '.join(failed) + '. Rerun same command to retry; completed trajectories remain cached.', flush=True)
    return 2 if failed or status else 0


def dispatch():
    if '--replay-worker' in sys.argv:
        index = sys.argv.index('--replay-worker')
        return replay_worker_main(sys.argv[index + 1])
    if '--qwen-worker' in sys.argv:
        sys.argv.remove('--qwen-worker')
        return qwen_worker_main()
    if '--clip-worker' in sys.argv:
        sys.argv.remove('--clip-worker')
        return clip_worker_main()
    return main()


if __name__ == '__main__':
    try:
        raise SystemExit(dispatch())
    except KeyboardInterrupt:
        print('[chunk-map] interrupted; saved maps, target caches and report.html remain. Rerun same command to resume completed work.', file=sys.stderr)
        raise SystemExit(130)
