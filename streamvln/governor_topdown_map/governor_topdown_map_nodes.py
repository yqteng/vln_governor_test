"""StreamVLN V6.4: original observed map engine with larger rendering and GRID-FAST nodes.
Copy this SINGLE file into StreamVLN/streamvln/governor_topdown_map_nodes.py.
No local companion modules are required. Existing model weights/environments remain required.
API configuration stays in a private JSON outside the project. Default backend is local.
Junction detector: GRID-FAST, Fredriksson, Saradagi and Nikolakopoulos (2024),
DOI 10.1007/s10846-024-02180-6. Python occupancy/gap/opening/region adaptation;
bounded local contour optimization, not an exact port of the complete ROS optimizer.
Topology filtering never removes raw navigation obstacle evidence.
Projection, grid generation, wall extraction, clustering and portal/passage decisions
are restored from the user's original working file, including original defaults/caps.
The original TopdownMap render uses larger output and region colors that do not
repeat every five IDs. Floor/explored lower acceptance bounds are independently
relaxed to -0.30m; the stored floor reference and obstacle height rules are unchanged.
ROSE2 extension is bounded by first unknown cell / existing wall; all fitted
segments are additionally clipped at unknown gaps. Other extraction/clustering/
doorway/passage settings are retained. Only node markers are drawn, no purple outline.
Portal candidates additionally use raw measured walls; connectivity excludes all
measured obstacles. Wider entrances require observed widening on both sides.
Default map resolution is 0.01m/cell; the 9000-cell active extent preserves the
previous 90m physical window. Metric doorway/passage/robot thresholds are unchanged.
Normal execution: route at decision nodes, STOP after a native policy STOP only.
After a rejected native STOP: four-forward/turn-end STOP checks are enabled during
correction, then disabled on correction completion or consistent node handoff.
This version has not been checked, tested, benchmarked or run on the server, per user request.
"""
# --- core.py ---
"""Original policy, history, replay, records and evaluation interfaces."""
import math
import argparse
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import select
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CORRECTION_MODE = "topdown_map"
VERSION = "governor_topdown_map_v6_4_6_grid_1cm"
ENTRYPOINT = Path(__file__).resolve()
ROOT = next(
    (parent for parent in ENTRYPOINT.parents
     if (parent / 'streamvln').is_dir() and (parent / 'config/vln_r2r.yaml').is_file()),
    ENTRYPOINT.parents[1],
)
WINDOW = 16
DIRECTIONS = ("front", "right", "back", "left")
TURNS = {"front": 0, "right": -6, "back": 12, "left": 6}
ACTIONS = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 4: "BACKWARD"}
TOKENS = {0: "STOP", 1: "↑", 2: "←", 3: "→"}


def json_object(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("VLM did not return a JSON object")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def pack_message(text, image=None):
    content = [{"type": "text", "text": text}]
    if image is not None:
        buffer = io.BytesIO()
        if text.startswith("LATEST MAP"):
            image.convert("RGB").save(buffer, format="PNG")
        else:
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
        content.append({"type": "image", "image": base64.b64encode(buffer.getvalue()).decode()})
    return {"role": "user", "content": content}


class LocalVLM:
    def __init__(self, args):
        env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH", "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=str(args.local_vlm_gpu), PYTHONUNBUFFERED="1")
        self.process = subprocess.Popen([
            args.local_vlm_python, "-u", str(ENTRYPOINT), "--vlm-worker",
            "--model", args.local_vlm_model_path, "--attention", args.local_vlm_attn_implementation,
            "--pixels", str(args.local_vlm_max_pixels)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, env=env)
        self.timeout = args.local_vlm_request_timeout
        self.read(args.local_vlm_startup_timeout)

    def read(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            self.process.kill()
            raise TimeoutError("VLM request timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("VLM worker exited")
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result.get("error", "VLM failure"))
        return result

    def generate(self, messages, tokens):
        self.process.stdin.write(json.dumps({"messages": messages, "tokens": tokens}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        result = self.read(self.timeout)
        if result.get("truncated"):
            error = ValueError("VLM output reached the token limit")
            error.raw_reply = result["text"]
            raise error
        return result["text"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def vlm_worker(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--pixels", type=int, default=802816)
    args = parser.parse_args(argv)
    protocol = sys.stdout
    sys.stdout = sys.stderr
    def emit(value):
        protocol.write(json.dumps(value, ensure_ascii=False) + "\n")
        protocol.flush()
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        torch.manual_seed(0)
        torch.cuda.set_device(0)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                                  min_pixels=12544, max_pixels=args.pixels)
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation=args.attention).to("cuda:0").eval()
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50
        emit({"ok": True})
    except Exception as exc:
        emit({"ok": False, "error": str(exc)})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            messages = request["messages"]
            for msg in messages:
                # Qwen's multimodal processor expects typed content, including system text.
                if isinstance(msg["content"], str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
                for item in msg["content"]:
                    if item["type"] == "image":
                        item["image"] = Image.open(io.BytesIO(base64.b64decode(item["image"]))).convert("RGB")
            batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=request["tokens"], do_sample=False,
                                        num_beams=1, use_cache=True, return_dict_in_generate=False)
            tokens = output[0, batch["input_ids"].shape[1]:]
            eos = model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos)
            truncated = len(tokens) >= request["tokens"] and int(tokens[-1]) not in eos
            emit({"ok": True, "text": processor.tokenizer.decode(tokens, skip_special_tokens=True),
                  "truncated": truncated})
            del batch, output, tokens  # No conversation/KV survives this request; weights stay loaded.
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            emit({"ok": False, "error": type(exc).__name__ + ": " + str(exc)})
    return 0


@dataclass
class Frame:
    observations: dict
    position: object
    rotation: object
    physical_step: int
    phase: str = "normal"
    action: int = None
    query: bool = False

    @classmethod
    def capture(cls, observations, sim, step, phase):
        state = sim.get_agent_state()
        return cls({k: observations[k].copy() for k in ("rgb", "depth", "gps", "compass")},
                   state.position.copy(), copy.deepcopy(state.rotation), step, phase)

    def image(self):
        return Image.fromarray(self.observations["rgb"]).convert("RGB")

    @property
    def height(self):
        return float(self.position[1])


def register_backward(config):
    import habitat
    import habitat_sim
    from habitat.config.default_structured_configs import ActionConfig
    from habitat.core.embodied_task import SimulatorTaskAction
    from habitat.core.registry import registry
    @registry.register_task_action
    class GovernorMoveBackwardAction(SimulatorTaskAction):
        def step(self, *args, **kwargs):
            self._sim.get_agent(0).agent_config.action_space["governor_move_backward"] = habitat_sim.ActionSpec(
                "move_backward", habitat_sim.ActuationSpec(amount=0.25))
            return self._sim.step("governor_move_backward")
    with habitat.config.read_write(config):
        config.habitat.task.actions["governor_move_backward"] = ActionConfig(type="GovernorMoveBackwardAction")


def four_views(sim):
    import quaternion
    state, previous = copy.deepcopy(sim.get_agent_state()), sim._prev_sim_obs
    views = {}
    try:
        for i, name in enumerate(DIRECTIONS):
            rotation = state.rotation * quaternion.from_rotation_vector(np.array([0., -i * np.pi / 2, 0.]))
            obs = sim.get_observations_at(state.position, rotation, keep_agent_at_new_pose=False)
            views[name] = Image.fromarray(obs["rgb"].copy()).convert("RGB")
    finally:
        sim.get_agent(0).set_state(state, reset_sensors=False, infer_sensor_states=False)
        sim._prev_sim_obs = previous
    return views


def validate_plan_reply(reply):
    decision = reply.get("decision")
    if decision == CORRECTION_MODE:
        if CORRECTION_MODE == "point":
            point = np.asarray(reply.get("point"), dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all() or np.any((point <= 0) | (point >= 1)):
                raise ValueError("point must be ONE interior normalized ground pixel [x,y], each strictly between 0 and 1")
            if "path" in reply or "points" in reply:
                raise ValueError("return one point, not a path or multiple points")
        elif not isinstance(reply.get("instruction"), str) or not reply["instruction"].strip():
            raise ValueError("instruction must be nonempty")
    else:
        raise ValueError("planning decision must be " + CORRECTION_MODE + "; global STOP is decided by the shared review")
    if not isinstance(reply.get("reason"), str) or not reply["reason"].strip():
        raise ValueError("reason must explain the decision")
    return reply


def project_ground_point(frame, sim, evaluator, point, max_distance):
    import quaternion
    cfg = evaluator.sim_sensors_config.depth_sensor
    rgb_cfg = evaluator.sim_sensors_config.rgb_sensor
    state = sim.get_agent_state()
    camera, rgb = state.sensor_states["depth"], state.sensor_states["rgb"]
    if (abs(cfg.hfov-rgb_cfg.hfov) > 1e-4 or np.linalg.norm(camera.position-rgb.position) > 1e-4
            or quaternion.rotation_intrinsic_distance(camera.rotation, rgb.rotation) > 1e-4):
        raise ValueError("RGB/depth camera pose and field of view must agree")
    depth = np.asarray(frame.observations["depth"]).squeeze()
    if cfg.normalize_depth:
        depth = depth*(cfg.max_depth-cfg.min_depth)+cfg.min_depth
    h,w = depth.shape
    rh,rw = frame.observations["rgb"].shape[:2]
    if abs(w/h-rw/rh)>1e-4:
        raise ValueError("RGB/depth aspect ratios differ")
    u,v = np.asarray(point)*[w-1,h-1]
    z = float(depth[int(round(v)),int(round(u))])
    if not np.isfinite(z) or not max(.02,cfg.min_depth)<z<cfg.max_depth-1e-4:
        raise ValueError("selected point has invalid/clipped depth; select visible nearby floor")
    f = w/(2*np.tan(np.deg2rad(cfg.hfov)/2))
    local = np.array([(u-(w-1)/2)*z/f, -(v-(h-1)/2)*z/f, -z])
    target = np.asarray(camera.position)+quaternion.rotate_vectors(camera.rotation,local)
    distance = float(np.linalg.norm((target-frame.position)[[0,2]]))
    floor_error = abs(float(target[1]-frame.height))
    if floor_error > .18:
        raise ValueError("point is %.2f m above/below current floor; choose a ground pixel" % floor_error)
    if not .15 <= distance <= max_distance:
        raise ValueError("point is %.2f m away; choose visible floor between 0.15 and %.2f m" % (distance,max_distance))
    # Keep the actual projected 3D point; changing its height would change its pixel.
    return target


def reproject_point(frame, sim, evaluator, target):
    import quaternion
    camera = sim.get_agent_state().sensor_states["rgb"]
    local = quaternion.rotate_vectors(camera.rotation.conjugate(), target-np.asarray(camera.position))
    z = -float(local[2])
    if z <= 1e-6:
        return None, False
    h,w = frame.observations["rgb"].shape[:2]
    f = w/(2*np.tan(np.deg2rad(evaluator.sim_sensors_config.rgb_sensor.hfov)/2))
    uv = [(local[0]*f/z+(w-1)/2)/(w-1), (-local[1]*f/z+(h-1)/2)/(h-1)]
    return [float(x) for x in uv], bool(0 <= uv[0] <= 1 and 0 <= uv[1] <= 1)


def point_action(frame, target):
    import quaternion
    delta = target-frame.position
    if np.linalg.norm(delta[[0,2]]) < .125:
        return 1  # Completion is checked by reprojection, not a distance threshold.
    local = quaternion.rotate_vectors(frame.rotation.conjugate(),delta)
    angle = np.arctan2(-local[0],-local[2])
    return (2 if angle > 0 else 3) if abs(angle)>np.deg2rad(7.5)+1e-6 else 1


def draw_point(frame, point, filename=None):
    image = frame.image()
    w,h = image.size
    x,y = round(point[0]*(w-1)),round(point[1]*(h-1))
    draw = ImageDraw.Draw(image)
    draw.ellipse((x-8,y-8,x+8,y+8),fill=(255,90,45),outline="white",width=2)
    draw.text((x+11,y-10),"P",fill="white",stroke_width=2,stroke_fill="black")
    if filename is not None:
        image.save(filename)
    return image

# Task prompts are composed explicitly; no other task's conversation is inherited.
MOTION = """
The robot's physical actions are fixed: LEFT/RIGHT rotate 15 degrees in place;
FORWARD attempts 0.25 m. Four forward actions attempt 1 m, six turns rotate 90
 degrees, twelve rotate 180 degrees. Collisions can prevent movement. Four camera
views are front/right/back/left, 90 degrees apart, taken at ONE physical position.
Rendering these views does not move or turn the robot. After a selected heading
is physically aligned, that old heading becomes the NEW FRONT. A newly visible
room after rotation is not a room the robot has entered.
""".strip()
INSTRUCTION_EVIDENCE = """
Ground semantic decisions ONLY in targets, regions and spatial relations named
in the original global instruction. An unrelated salient object must not become
a route cue, even indirectly through an assumed typical scene layout. For a named
area or spatial relation, use visible structure and geometry that support that
specific area or relation. An unrelated object does not establish the identity,
location or direction of the requested destination. Do not substitute object
co-occurrence or scene stereotypes for instruction-grounded visual evidence.
Unmentioned obstacles may constrain clearance but cannot choose the goal.
Preserve route-relative left/right and the original landmark's identity. Do not
invent hidden rooms, connections, destinations, distances or waiting durations.
Completed tasks describe what MUST NOT be repeated. However, a completed room
can contain the exit needed for the next task: do not exclude that exit merely
because the room was visited. Choose for the unfinished requirement, not because
a direction resembles the completed destination.
""".strip()
VISUAL_CONTINUITY = """
Ground the CURRENT location in the retained VISUAL SEQUENCE, not just the latest
picture. Historical anchor images carry state IDs; they are actual past views,
not current views and not established verdicts. Match the SAME instruction-named
opening, passage, room boundary or landmark across these images: its shape,
relative boundary arrangement, approach perspective and transition into its
interior. Explain whether the robot is approaching, inside, or past the place
previously visible, or whether correspondence is uncertain. Similar-looking
openings alone do not prove identity; use the chronological connecting views.
Sparse anchor images do not prove what happened in an unobserved gap. Summaries
are fallible supporting context; compare them with the supplied actual images.

A previously visible instruction-relevant target or region boundary can leave
the forward field of view while the robot approaches, enters, passes or changes
orientation relative to it. Determine which relation is supported by the actual
visual transition; do not assume any one transition has occurred. Establish
whether the current surrounding space corresponds to the interior or vicinity
of the SAME previously observed region, or whether evidence supports another
location. Track target identity, relative position, boundary geometry and changes
in perspective across time. Current visibility and current spatial relation are
separate questions: a target can be visible from outside its required region,
or absent from the forward view while the robot is within that region.

Current non-visibility alone proves neither departure/deviation nor arrival or
completion. Compare supported approach/entry/passage with alternatives such as
rotation, occlusion or a mistaken identity. Preserve uncertainty if the evidence
does not resolve them. Do not invent a forward action or a crossing where only
a camera-view change is supported. Apply these criteria to the actual instruction
and observations without assuming a particular target category or scene layout.

Do not assert 'entered a new room' merely because the current picture differs
from the old exterior view. Uncertain identity means uncertain localization;
retain this uncertainty in the correspondence description. A past match is not
proof of current identity when subsequent images contradict the connection.
""".strip()
ENDPOINT_MEANING = """
Treat terminal 'wait', 'stop', 'stay' and equivalent wording as a spatial endpoint
requirement: once the required route and CURRENT spatial relation are satisfied,
executing STOP here fulfills it. The robot need not already be stationary, have
issued STOP, or have waited for an invented duration. Motion immediately before
assessment is not an unfinished spatial requirement. Distinguish a future action
needed to finish the episode from movement still needed to reach its endpoint.

Anchor the final target identity and stopping relation to the ORIGINAL instruction
and its approach context. Keep the same meaning across time, camera headings,
summaries and recovery. Do not silently replace at/before/beside/in with beyond,
a deeper interior, another region or an imagined precise distance. Determine a
visually supported stopping REGION consistent with the wording, not an invented
exact coordinate. Being in the correct room or on the correct route is not enough
if the required stopping relation has already been passed. Conversely, being at
the required endpoint does not require an unmentioned further crossing.
Seeing a target is not arrival; losing sight of it is not required for every
arrival. Current location relative to the specified target is what matters.
""".strip()
PROGRESS_RULES = MOTION + "\n" + ENDPOINT_MEANING + "\n" + """
Separate SEEING a target, APPROACHING it, REACHING it, and PASSING THROUGH/PAST it.
A target in view is not proof of arrival. A forward command is not proof of
movement. Establish arrival from the SAME target's chronological visual identity,
real approach supported by executed actions AND image changes, and the actual
current position satisfying the specified relation. Do not require losing sight
of the target merely to arrive at/beside/before it. For a through/past/cross clause,
require a visible target/boundary, real forward approach and its disappearance
from the forward view due to crossing/passing, together with evidence of the
resulting spatial relation. Disappearance due to rotation, occlusion, a missing
picture or changed target identity does not establish that crossing.
Never mark 'through/past/cross' complete merely because the target was visible
and some forward commands occurred. Check the visual change across the crossing.
Turning alone can complete a pure orientation clause, but cannot complete room
entry, passage, approach or translation. Split compound instructions: 'turn and
walk through' is not complete when only the turn is complete. Keep partial and
uncertain progress explicit; do not upgrade an old summary's speculation to fact.
Only the supplied retained prefix is valid. Never use discarded history after a
rollback checkpoint. Preserve confirmed earlier progress, but correct it when
actual images contradict it. A past arrival does not prove the current position
is still the required final stopping region. Stop/wait is a current-position
condition, not a permanently completed movement clause.
""".strip()
SUMMARY_PROMPT = PROGRESS_RULES + "\n" + INSTRUCTION_EVIDENCE + "\n" + VISUAL_CONTINUITY + "\n" + """
Your ONLY task is to compress this retained history interval into task progress.
Use the original instruction, earlier supplied summaries, the interval's images,
and ACTUAL action records together. This request explicitly permits action
history; do not apply the image-only restriction of a route monitor here.
State which instruction parts are confirmed completed, which are partial or
uncertain, the current location/route stage, and what remains next. Preserve the
identity and relation of the original final target. Do not merely narrate objects
or commands. A turns-only interval must still update fulfilled orientation parts
and retain earlier movement progress; do not fabricate a new room transition.
Organize progress as: confirmed completed instruction clauses; the supported
current location and relation to the instruction's target; remaining clauses and
uncertainties. Fill these only from the actual instruction and supplied evidence,
without inserting a fixed target, region, route or scene-specific example.
Do not output off-route verdicts, STOP proposals, new tasks or action plans.
Also retain a short VISUAL correspondence description: identify an instruction-
named place in an earlier state, how its boundary/approach changed, and whether
the interval ends before, inside or past that SAME place. Keep uncertainty and
alternative identities explicit; no raw action ledger in this visual description.
Select TWO distinct supplied integer state IDs as anchor_steps from this interval. Prefer
a clear pre-entry/approach view and an informative entry/interior view of the
current instruction-relevant transition. If uncertain, select the interval start
and end as neutral continuity anchors; they do not imply task completion. Never
return an empty list. Use the actual integer state labels, not quoted numbers.
These pictures will accompany independent visual tasks across summary
boundaries, including position reviews and route monitoring. Do not select arbitrary decorative objects or unsupported frame IDs.
Return {"summary":"concise completed/current/remaining task description",
"grounding":"visual correspondence with state IDs; uncertainty",
"anchor_steps":[earlier_state_id,later_state_id]}.
""".strip()
LOCATION_PROMPT = INSTRUCTION_EVIDENCE + "\n" + MOTION + "\n" + """
Describe the area and front/right/back/left views at this ONE actual position.
The labels are relative to the current body orientation. The robot is facing
FRONT; viewing the other images did not execute any actions. Identify visible
instruction-relevant structure, uncertainty and traversable openings. Do not
invent unseen connections or claim a task was completed from seeing its target.
Do not choose a direction, reason about route progress, propose STOP or rewrite
the instruction. Initialization uses only this descriptive task, before navigation.
Return {"area":"...","front":"...","right":"...","back":"...","left":"..."}.
""".strip()
DIRECTION_PROMPT = INSTRUCTION_EVIDENCE + "\n" + MOTION + "\n" + """
Choose a heading for the NEXT UNFINISHED requirement using this recovery's
four-view observations and progress description. Completed requirements tell you
where NOT to return merely to repeat them. Do not select a heading because it
contains the completed target. The choice must be positively connected to an
unfinished instruction requirement by visible instruction-relevant evidence.
Compare front/right/back/left. A completed room's forward exit may still support
the remaining route. For blockage, choose a visibly clear instruction-consistent
approach around the obstacle; do not select an unmentioned object as a new goal.
Keep front when appropriate. Choosing another heading causes real 15-degree
turns before planning; do not assume the robot has entered the newly visible area.
If no direction has grounded support, choose none: the controller keeps the
actual current heading and still asks for a locally grounded recovery plan.
Do not output STOP, coordinates or the recovery instruction in this step.
Return {"direction":"front|right|back|left|none","reason":"unfinished requirement and visible support"}.
""".strip()
REWRITE_PROMPT = INSTRUCTION_EVIDENCE + "\n" + PROGRESS_RULES + "\n" + """
Generate ONE SHORT LOCAL CORRECTION instruction for StreamVLN, not a paraphrase
of the global instruction. The shared review already establishes current progress
and decides global STOP. Your ONLY job here is the specific next local maneuver.

Use the ORIGINAL global instruction, the shared review's completed/remaining
requirements, the ACTUAL rollback observations and four-direction images, and
the recorded alignment. Identify what needs correction NOW: which nearby route
area needs to be approached and from which direction. Choose the nearest useful
visible subgoal that repairs that issue. It need not be the global destination.
Do not send the robot through an entire room or several areas when a short move
to a nearby entrance/boundary will return it to the intended approach.

The temporary instruction MUST contain all three, in this order:
1. Concrete direction relative to the ACTUAL current front: forward, left/right,
   ahead-left/ahead-right, or another direction supported by the current views.
2. The nearby visible instruction-relevant area/boundary to approach in that
   direction. Identify it sufficiently to distinguish the intended route.
3. An explicit, visually recognizable LOCAL endpoint where this correction ends
   and StreamVLN should issue its local STOP; the ORIGINAL global task then resumes.
Keep the destination and the travel short. Prefer one brief executable sentence,
roughly one approach and one ending condition, without explanatory paragraphs.
A generic pattern is 'Move [current-relative direction] toward [nearby visible
instruction-related area], and stop [at its nearest useful boundary].' This is
a structure, not a fixed target or direction: fill it only from this episode's
instruction and current images. Do not copy placeholders or invent a landmark.

The heading selected before alignment has ALREADY been reached by real turns.
It is NOW FRONT. A former left/back view may now require 'forward'; do not repeat
the recorded turn or use direction words relative to the pre-alignment camera.
If no alignment occurred, use the unchanged actual current frame. Do not confuse
turning toward a room with entering it. Exclude confirmed completed destinations
as new goals, while allowing their forward exit when required for remaining work.
Preserve exact global at/in/beside/before/through relations: a boundary goal must
not become entry into another room or a more distant destination. Do not merely
replace words in a global clause; specify a grounded direction and short ending.
No nested correction is possible. Avoid long, vague or exploratory temporary tasks.
Return {"decision":"instruction","instruction":"one concise directional local task in English","reason":"current issue, selected nearby area and short endpoint"}.
Do not make a separate global STOP judgment; that authority belongs to the shared
position/route/STOP review. 'Stop at ...' inside the local instruction is the
ending condition for this temporary task, not a global STOP verdict.
""".strip()
POINT_PROMPT = INSTRUCTION_EVIDENCE + "\n" + MOTION + "\n" + """
Choose ONE NEARBY GROUND POINT in the ACTUAL CURRENT FRONT image for a short local
recovery. The original global instruction remains unchanged. Use the completed
progress, frozen original endpoint clauses, current views and recorded alignment.
The selected old direction is already the NEW FRONT. Do not repeat that turn.
The point indicates the nearby AREA to move toward, not a distant final goal.

The pixel MUST lie on directly visible, traversable floor at the robot's current
floor level. Choose a modest short advance, within the supplied maximum ground
distance, with a clear straight approach and enough robot clearance. Do not choose
a doorway silhouette, wall, furniture surface, threshold above the floor, hidden
floor, reflection, obstacle, drop or distant target merely because it is salient.
No new target may be invented. Preserve the exact original stopping relation and
do not move deeper into an area when the instruction only requires its boundary.
The current image and visible floor geometry determine the pixel; there is no
fixed preferred coordinate, centerline or scene-specific point.

Return ONE [x,y] coordinate, normalized to the current FRONT image: (0,0) top left,
(1,1) bottom right. Both coordinates must lie strictly between 0 and 1. Return
numbers, not pixel units, strings, a polyline, a list of points or an empty value.
The controller depth-projects this pixel once into a FIXED world-space point,
turns/advances toward it and reprojects that SAME point into each new camera view.
After real approach makes it leave the field of view, local correction ends and
the original global task resumes. Local correction is not monitored for deviation
and cannot spawn a nested correction task. The point is not reselected each step, and reaching a
small distance alone does not end correction. Rotation alone and occlusion are
not successful local completion. Choose a nearby floor point suitable for this
short approach; do not plan three points or a long route.

Return {"decision":"point","point":[x,y],"reason":"unfinished requirement, nearby visible floor and clear approach"}.
The shared position/route/STOP review already handled global arrival. Do not
make another STOP judgment here. Do not fabricate a point when geometry is
uncertain. If depth, height or distance validation fails, reassess this same
image and choose a grounded nearby floor pixel.
""".strip()
CHECKPOINT_PROMPT = INSTRUCTION_EVIDENCE + "\n" + """
Select ONE supplied candidate image just before the route began to go wrong, from
which a recovery can continue the unfinished instruction. The current image is
also supplied to show the deviation, but may not be an eligible candidate.
Use only the original instruction, explicit retained progress and supplied visual
history. No action ledger or hidden goal positions. Do not return to a completed
destination simply because it resembles an instruction noun. Prefer a useful
last correct decision/approach position. Candidates are limited to the CURRENT
16-action compression interval, including its boundary, not all prior history.
The recovery trigger is supplied explicitly: a positional/directional/order
route mismatch, a rejected early global STOP, or code-detected blocked motion.
Select a useful valid checkpoint for that trigger; do not diagnose a new error
or introduce a separate STOP decision.
Do not score every frame, build a STOP-frame pool or rewrite the instruction.
Return {"step":one_candidate_integer,"reason":"why this is the recovery checkpoint"}.
""".strip()
PLAN_SCOPE_PROMPT = ENDPOINT_MEANING + "\n" + """
Check ONLY whether the supplied local recovery plan preserves the ORIGINAL global
instruction and frozen endpoint clauses. No images or action history are supplied:
you cannot re-localize the robot or change the endpoint from a guessed scene.
The local task may be a short intermediate maneuver, but cannot add a new goal,
repeat confirmed work, change target identity, or strengthen/weaken the terminal
spatial relation. In particular, reaching a boundary is not permission to enter
an additional region beyond it. For an instruction, require ALL of: a concrete CURRENT-relative direction, a
nearby visible instruction-related region to approach, and an explicit local
stopping boundary. Reject a synonym-only global rewrite, a vague copied clause,
a repeated already-executed alignment, or unnecessary long/multi-region travel.
For a point, check its stated intended region and relation; the controller checks
its floor geometry separately. The literal global goal takes priority over a
contradictory progress description or a plan's unsupported interpretation.
Return {"accepted":true_or_false,"reason":"the specific preserved clause or scope violation"}.
Do not output a new instruction, a new point, a STOP verdict or a new location.
""".strip()


JOINT_REVIEW_PROMPT = INSTRUCTION_EVIDENCE + "\n" + PROGRESS_RULES + "\n" + VISUAL_CONTINUITY + "\n" + """
Produce THREE DISTINCT sections in ONE response to this one review request.
Resolve a SINGLE shared account of the CURRENT spatial state first. Both route
checking and STOP must use that SAME area, heading, stage and route history.
Do not have separate sections invent contradictory places or endpoint meanings.

A. POSITION REVIEW: CONTINUOUS location inference from paired history.
Use the original global instruction, retained actions and coarse summaries, the
HISTORICAL IMAGE/DESCRIPTION PAIRS, and the separately labeled ACTUAL CURRENT
observation. This is an update of the previous spatial interpretation, not a
fresh independent room description at every step.

Each historical message contains ONE past image and the position description
written for that SAME state and physical step. At most four frames are sampled
uniformly from the previous 16 valid historical states; the current frame is
separate. If fewer than four historical frames exist, all available frames are
supplied, including zero at the start. This is valid: do not request padding,
repeat frames or invent missing observations. A description marked not_available
means only that this exact image lacks a review; do not borrow another state's
text. An initial_scan description applies only to that initial paired image.

Read each pair as a unit, in time order. Use its description and image to
understand the region occupied THEN and any supported surrounding region: FRONT,
RIGHT, BACK or LEFT. Directions mentioned in a historical description belong to
that historical state's heading, not the current heading or fixed global axes.
A historical front image is one time sample, not a simultaneous panoramic scan.
Its description may include other directions established by earlier actual views.

Navigation is continuous. Interpret the current image through those previous
observations/descriptions and the intervening actions, rather than restarting
localization from scratch. Consider where the robot could now be relative to
those earlier places and which previously described region it may have entered.
A region previously ahead, to either side or behind can become the current
region after turning and moving. Do not restrict matching to historical FRONT.
Understand heading and physical movement separately: turning in place changes
what is visible but does not move the robot into another region. Translation
can carry it into a region previously seen in any direction. Action commands
alone do not prove movement; use visual transitions and boundaries as evidence.

Use these spatial relationships to understand continuity, but do NOT output
separate position-change or heading-change fields, a direction table, or a
step-by-step account of this reasoning. The output goal is a natural CURRENT
location description connected to the previous descriptions. In location,
describe where the robot is now, referring to a previously described region
when the match is supported, and mention relevant nearby regions in whichever
direction the evidence supports. In history_match, briefly cite the supporting
sampled state(s) and visual correspondence, or say what remains uncertain.
For example, if supported, say that the robot is now inside the passage seen
to the right at an earlier state; do not keep describing that same passage as
an unreached destination merely because it was previously outside the robot.
This is a general continuity rule for any region, not a prescribed route.

A former description 'leading to [region]' does not mean that region must remain
forever ahead. When the retained observations and actual motion support entry,
update the description to being in that region. Do not invent another still-ahead
copy of it. Conversely, proximity, a turn, a forward command, or a landmark
disappearing alone does not establish entry. Match visible layout/boundaries,
the connecting transitions and the instruction's spatial meaning. Losing sight
of a landmark can be compatible with moving into or past its region; lack of
current visibility alone does not erase a grounded historical correspondence.

Normal navigation supplies only current FRONT RGB. You may use prior actual
views and supported movement to understand other directions, but do not invent
unseen regions or a complete map. Historical time samples are not current side
or rear images. Only images explicitly labeled ACTUAL CURRENT right/back/left
provide current observations in those directions. An old nearby region need
not remain nearby after movement; preserve uncertainty when correspondence is
unsupported. No full four-direction inventory is required in each description.

If paired descriptions are supplied, do not claim that no prior review exists.
If the robot is still in the same place, preserve that location across a view
change unless evidence supports revising it. If it has entered a previously
observed region, explicitly connect the new location to that earlier region.
Continuity is evidence-based updating, not copying the old text unchanged.
Being in the same region does not mean two images are identical or that no motion
occurred. Repeated architecture can remain visible during movement. Do not assert
zero displacement merely because old descriptions used similar wording. Read the
actual sampled images and action record; do not call every past action FORWARD
or invent action counts. A historical description is a hypothesis, not a visual
measurement. Current visual evidence can revise it.

Old descriptions and summaries can be wrong. Detect their internal conflicts
before propagating them: 'already passed a region' cannot also mean 'not yet
passed that same region' in the same current account. Resolve using paired
images/actions and state which earlier interpretation is revised; when evidence
is insufficient, preserve uncertainty instead of copying both claims. Do not
copy old descriptions or the output example verbatim as a substitute for current
inference. All current fields (stage, completed, location, history_match, reason)
must express ONE consistent spatial interpretation.
Update stage and completed FROM the updated location and retained route evidence;
do not copy an old initial-stage label or empty completion list after establishing
that the robot has traversed those stages. If the route is uncertain, state that
uncertainty instead of claiming both route completion and no completed stages.
In stage describe the current instruction phase and any genuinely unfinished
spatial requirement. Being inside an intermediate room is not the same as being
at the GLOBAL FINAL stopping location. Terminal waiting itself is not unfinished
spatial work. B and C must use this SAME current account. Do not output an extra
before/within/beyond flag or a second remaining-task checklist to decide STOP;
the single explicit conclusion in C determines whether stopping here is valid.
Only the retained prefix is valid after rollback; never reconstruct discarded
future observations or descriptions from a different physical-step ID.

B. ROUTE CHECK: only WRONG PLACE, WRONG DIRECTION or WRONG REGION ORDER.
Use the position/heading/stage already established in A. Compare them with the
places, directions and ordering specified by the original instruction. Examples
of the GENERAL distinctions: taking one visually identified passage instead of
the instruction's different passage; heading along the right branch when the
applicable instruction requires the left branch; actually being in region B
before mandatory preceding region A was traversed. The last case is an ordering
error even if B is named in the instruction. These are generic relations, not
fixed objects, regions or scene rules. For order errors, require evidence that
a prerequisite was skipped: merely seeing a later region, an unobserved gap or
uncertain earlier progress does NOT establish that it was skipped.
Anchor left/right to the instruction's relevant approach and stage, accounting
for already executed reorientation. A camera-relative left/right change alone
is not a wrong direction. A slight alignment turn is not an incorrect branch.
Do not apply a later-stage requirement before its prerequisite is reached.

Do NOT check whether each action satisfies a task, whether the agent has already
waited, issued STOP, finished a temporal behavior or moved for long enough.
Remaining work by itself is not off-route. Passing a valid stopping region on
an otherwise correct path, continuing where it should stop, or failure to output
STOP belongs EXCLUSIVELY to C, not a route deviation by itself. Missing targets,
slow progress, collision or oscillation alone are not semantic route mismatches.
Report true only for positive evidence of wrong_place, wrong_direction or
wrong_order. Otherwise false/none. Motion blockage is a separate controller
trigger. No recovery-frame selection, action plan or instruction generation here.
When route_check_enabled=false, do NOT evaluate deviation: set route_check.off_route
to false, kind to disabled, and reason to 'Route checking is disabled in this mode.'
Keep this member inside the full response. This applies during correction,
rollback review and confirmation, preserving non-nested correction.

C. STOP PROPOSAL: one conclusion from the SAME location and route in A.
Ask: 'If STOP were executed at the ACTUAL CURRENT location NOW, would the original
global instruction be satisfied?' Check the final place, its specified spatial
relation and the required preceding route. A local correction endpoint is not the
global endpoint. Being somewhere in a room is insufficient when the instruction
requires a particular boundary or relation. Do not invent a deeper entry, a new
destination or an exact distance not required by the instruction.

Choose exactly ONE decision, and give the evidence for that same decision:
- stop: the evidence establishes the required final spatial position and route;
  STOP here would fulfill the instruction. Explain the satisfied spatial relation
  and the supported preceding progress. No further spatial task must remain.
- continue: the evidence positively establishes that STOP here would be wrong,
  for example a clearly identified necessary region is still ahead, the robot has
  passed the required stopping region, or a required part of the route was skipped.
  State the concrete unfulfilled spatial requirement and the observations proving
  it. This means a native global StreamVLN STOP must also be rejected at this pose.
- uncertain: the evidence cannot establish either of the above. Explain exactly
  what is unknown. 'Arrival has not been confirmed' or 'the target is not visible'
  alone belongs here, not automatically in continue. Do not assert definitely
  unfinished work when the evidence only leaves completion uncertain.

There are not separate standards for a governor proposal and a native global
STOP. Do not output an independent reject_vln_stop choice: the controller derives
it from the SAME decision (stop=false, continue=true, uncertain=false). Continue
requires comprehensive positive evidence of an incorrect endpoint or unfulfilled
mandatory route; uncertainty remains conservative and does not veto native STOP.
Your reason must agree with your decision. A reason that clearly establishes
'not at the required endpoint; movement is still necessary' cannot accompany stop
or uncertain. A reason establishing 'at the correct endpoint; no spatial work
remains' cannot accompany continue. Merely saying 'must continue' is not evidence.

Terminal wait/stop/stay is fulfilled BY issuing STOP at the correct final place.
It is NOT an additional spatial prerequisite and needs no prior stationary period.
The robot may still be moving and StreamVLN may not have issued STOP: catching
precisely that late-stopping behavior is the purpose of this proposal. If the
required route and final spatial relation are fulfilled, choose stop NOW, even
if a historical description still says 'wait there'. Never choose continue just
because the waiting/STOP action has not happened. If more movement is necessary,
name that movement's unmet spatial purpose, not the unexecuted waiting action.
This applies to arbitrary objects, regions and instruction phrasing.

Passing a valid stopping region on an otherwise correct path can yield continue
here without being a route error in B. The controller controls when proposals
are sampled or executed: do not weaken a stop conclusion because it will later
be returned to and confirmed. Uncertain does not prove departure from a stopping
region; it does not trigger a pending proposal's rollback. Only an explicit
continue does that. A native LOCAL StreamVLN STOP simply ends correction and
restores the parent task; the global stopping veto is never applied to it.

Mode controls authority, not goal meaning:
- normal: output all three sections. The controller uses route checking each
  navigation step; STOP proposals are sampled after four consecutive forward
  actions, at a same-direction turn-run end, and every action while pending.
- correction: A and C remain active on that STOP schedule; B is disabled.
- recovery / confirmation: actual four-view images are provided. A and C judge
  the ACTUAL returned/aligned pose, with B disabled. A positive STOP can be
  accepted immediately here. Otherwise the controller prepares one local task.
Initialization itself is descriptive only and does not call this review.
No separate completed-tasks decision or other task's STOP verdict overrides A/C.

OUTPUT CONTRACT (the parser uses these exact keys):
Return ONE JSON object with three members: position_review, route_check and
stop_proposal. No Markdown fences, leading commentary, separate JSON fragments
or extra objects. Use JSON true/false, not True/False, strings or placeholders.
Replace the example's contents with this episode's actual judgments; its default
uncertainty is illustrative, not a decision to copy on every step.

position_review.completed is an ARRAY OF STRINGS, one confirmed completed clause
per item. If no clause is confirmed complete, return [] and explain why in
position_review.reason; this is a valid answer, especially at the start. Unknown
progress must be explicitly described as unknown, not fabricated as complete.
Older records marked completion_status=not_reported contain no completion claim;
do not interpret their empty list as evidence that nothing has been accomplished.
Descriptive position fields (stage, location, history_match, reason) are TEXT.
location is the current place described continuously with the paired history;
history_match briefly states its evidential connection to earlier descriptions.
Do not add separate movement-change fields or require a four-direction table.
Explain unavailable evidence in words. If no earlier review or initial description
is actually supplied, say so and infer only from the available images/actions.
Do not leave explanations blank just because nothing is complete.
route_check.off_route is a JSON BOOLEAN. route_check.kind is ONE of none,
wrong_place, wrong_direction, wrong_order, disabled. stop_proposal.decision is
ONE of stop, continue, uncertain, with the precise meanings in C. The only
stop_proposal fields you generate are decision and reason; the controller adds
the matching reject_vln_stop flag. Do not emit enum lists joined by slashes or
vertical bars as values. No separate position enum or remaining_spatial list is
needed: explain progress in stage, and decide stopping exactly once in C.

EVERY section must include a nonempty reason, even for no completed tasks,
off_route=false, continue, uncertainty or disabled route checking. Give the
actual inference: relevant observations/actions, what they establish and what
remains uncertain. For false off-route explain why no positional/directional/order
conflict is established; do not leave reason empty. For continue identify the
positively established spatial mismatch or unfinished mandatory route. For
uncertain identify the missing evidence without pretending it proves non-arrival.
For stop explain how the original spatial endpoint is satisfied. Do not invent
evidence merely to fill a field. 'Insufficient evidence because ...' is valid.

Valid JSON structure for an UNCERTAIN situation (not a scene-specific example):
{"position_review":{"stage":"Current instruction stage is uncertain.",
"completed":[],
"location":"The available views do not establish the current instruction-relative place.",
"history_match":"No reliable correspondence to a historically described region is established.",
"reason":"The supplied evidence does not yet confirm a completed spatial clause or final arrival."},
"route_check":{"off_route":false,"kind":"none",
"reason":"Uncertain localization alone is not positive evidence of a wrong place, direction or order."},
"stop_proposal":{"decision":"uncertain",
"reason":"The images do not establish whether the current region is the required endpoint; there is also no positive evidence that a required spatial step remains."}}.
A skipped mandatory region is still unfinished route work even at the final
destination. True route error cannot coexist with fulfilled whole-task STOP:
resolve any contradiction against the actual evidence before answering. Do not plan recovery or issue motor
actions in this response. All examples and criteria apply to arbitrary scenes.
""".strip()

PLAN_INPUT_RULES = """
The shared review is the sole source of the supplied completed/current/remaining
assessment and global STOP decision. Use the unfinished spatial task described
in its stage and the confirmed completed clauses, plus actual current four views to generate the configured LOCAL correction. Do not independently
rejudge arrival, invent a different current region or decide global STOP here.
Do not repeat a completed clause. The recorded alignment was physically executed:
the selected old direction is now FRONT, and the new current front image is the
only ground-point canvas. Historical and other-direction images are context, not
pixel coordinates in that canvas. If the observations cannot support a safe plan,
do not invent a goal or fabricate a global STOP to bypass the planning task.
""".strip()

class StreamPolicy:
    """StreamVLN's RGB/depth/pose interface and actual-action history replay."""
    def __init__(self, evaluator):
        self.e = evaluator
        self.template = copy.deepcopy(evaluator.conversation)
        self.intrinsic = evaluator.get_intrinsic_matrix(evaluator.sim_sensors_config.rgb_sensor)

    def reset(self, first, instruction):
        self.e.conversation = copy.deepcopy(self.template)
        self.e.conversation[0]["value"] = self.e.conversation[0]["value"].replace("<instruction>.", instruction)
        self.e.model.reset_for_env(0)
        self.step = 0
        self.rgb, self.depth, self.poses, self.intrinsics, self.times, self.queue = [], [], [], [], [], []
        self.ids = self.past = None
        self.restoring = False
        x, y = first.observations["gps"]
        self.origin = np.array([x, -y])
        self.yaw = float(first.observations["compass"][0])
        self.height = first.height

    def append(self, frame):
        import torch
        from depth_camera_filtering import filter_depth
        e, obs = self.e, frame.observations
        self.times.append(self.step)
        depth = filter_depth(obs["depth"].reshape(obs["depth"].shape[:2]), blur_type=None)
        depth = (depth * (e._max_depth-e._min_depth) + e._min_depth) * 1000
        x, y = obs["gps"]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        xy = np.array([[c, s], [-s, c]]) @ (np.array([x, -y])-self.origin)
        pose = e.xyz_yaw_to_tf_matrix(np.array([*xy, e._camera_height+frame.height-self.height]),
                                      float(obs["compass"][0])-self.yaw)
        image = frame.image()
        pixels = e.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
        depth_image, shape = e.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16)), do_depth_scale=True)
        intrinsic = e.preprocess_instrinsic(self.intrinsic, image.size, shape)
        self.rgb.append(pixels)
        self.depth.append(torch.from_numpy(depth_image).float())
        self.poses.append(torch.from_numpy(pose) @ e.get_axis_align_matrix())
        self.intrinsics.append(torch.from_numpy(intrinsic).float())

    def inputs(self):
        import torch
        from utils.utils import dict_to_cuda, DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN
        fresh = self.ids is None
        if fresh:
            conversation = copy.deepcopy(self.e.conversation)
            conversation[0]["value"] = conversation[0]["value"].replace(DEFAULT_VIDEO_TOKEN+"\n", "")
            if self.step:
                conversation[0]["value"] += " These are your historical observations " + DEFAULT_MEMORY_TOKEN + "."
        else:
            conversation = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
        ids, _ = self.e.preprocess_qwen([conversation], self.e.tokenizer, True, add_system=fresh)
        if not fresh:
            ids = torch.cat([self.ids, ids.to(self.ids.device)], dim=1)
        streams = [self.rgb[-1:], self.depth[-1:], self.poses[-1:], self.intrinsics[-1:]]
        if self.step and (self.step % self.e.num_frames == 0 or fresh and self.restoring):
            stride = max(1, self.times[0] // self.e.num_history)
            sample = slice(0, self.times[0], stride)
            streams = [old[sample]+new for old, new in zip(
                (self.rgb, self.depth, self.poses, self.intrinsics), streams)]
        data = dict(zip(("images", "depths", "poses", "intrinsics"),
                        [torch.stack(items).unsqueeze(0) for items in streams]))
        data.update(inputs=ids, env_id=0, time_ids=[self.times], task_type=[0])
        data = dict_to_cuda(data, self.e.device)
        for key in ("images", "depths", "poses", "intrinsics"):
            data[key] = data[key].to(torch.bfloat16)
        return data

    def predict(self, frame):
        import torch
        self.append(frame)
        frame.query = not self.queue
        if frame.query:
            with torch.inference_mode():
                result = self.e.model.generate(**self.inputs(), do_sample=False, num_beams=1,
                    max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=self.past)
            self.ids, self.past = result.sequences, result.past_key_values
            text = self.e.tokenizer.batch_decode(self.ids, skip_special_tokens=False)[0].strip()
            self.queue = list(self.e.parse_actions(text) or [0])
        return int(self.queue.pop(0))

    def advance(self):
        self.step += 1
        if self.step % self.e.num_frames == 0:
            self.e.model.reset_for_env(0)
            self.ids = self.past = None
            self.times = []

    def replay_batch(self, data, actions):
        import torch
        model = self.e.model
        embeds = model.prepare_inputs_labels_for_multimodal(data["inputs"], None, None, None, None,
            data["images"], None, data["depths"], data["poses"], data["intrinsics"],
            data["time_ids"], data["task_type"])[4]
        cache = model.cache[0]
        if model.curr_t[0]:
            embeds = torch.cat([cache["inputs_embeds"], embeds], dim=1)
        cache["inputs_embeds"] = embeds
        model.curr_t[0] += 1
        length = (0 if self.past is None else self.past.get_seq_length()
                  if hasattr(self.past, "get_seq_length") else self.past[0][0].shape[-2])
        self.past = model.model(inputs_embeds=embeds[:, length:], past_key_values=self.past,
                                use_cache=True, return_dict=True).past_key_values
        self.ids = self.e.tokenizer("".join(TOKENS[a] for a in actions)+"<|im_end|>",
            add_special_tokens=False, return_tensors="pt")["input_ids"].to(embeds.device)
        if self.ids.shape[1] > 1:
            self.past = model.model(input_ids=self.ids[:, :-1], past_key_values=self.past,
                                    use_cache=True, return_dict=True).past_key_values

    def restore(self, history, instruction):
        import torch
        self.reset(history[0], instruction)
        self.restoring = True
        with torch.inference_mode():
            for i, frame in enumerate(history[:-1]):
                self.append(frame)
                if frame.query:
                    end = next((j for j in range(i+1, len(history)-1) if history[j].query), len(history)-1)
                    self.replay_batch(self.inputs(), [f.action for f in history[i:end]])
                self.advance()
        self.queue = []


class HistoryPlanner:
    """No hidden chat history. Each method supplies only that task's evidence."""
    def __init__(self, client, instruction, args):
        self.client, self.instruction, self.args = client, instruction, args
        self.summaries = []
        self.initial_location = None
        self.position_reviews = []  # Explicit memory of this task only, pruned on rollback.
        self.last_review_failure = None
        self.endpoint = [instruction]  # Fixed literal goal; no separate extraction call.
        self.log = lambda data: None

    @property
    def boundary(self):
        return self.summaries[-1]["end"] if self.summaries else 0

    def ask(self, prompt, data, images=(), tokens=None, validator=None, trace=None):
        messages = [{"role": "system", "content": prompt},
                    pack_message("ORIGINAL GLOBAL INSTRUCTION:\n"+self.instruction+
                        "\nFROZEN ORIGINAL ENDPOINT CLAUSES: "+json.dumps(self.endpoint, ensure_ascii=False)+"\n\n"+data)]
        messages += [pack_message(label, image) for label, image in images]
        for attempt in range(2):
            raw = None
            try:
                raw = self.client.generate(messages, tokens or self.args.init_max_new_tokens)
                result = json_object(raw)
                if not isinstance(result, dict):
                    raise ValueError("response must be one JSON object")
                if validator is not None:
                    result = validator(result)
            except (ValueError, TypeError) as exc:
                raw = getattr(exc,"raw_reply",raw)
                if trace:
                    trace(attempt=attempt+1, raw=raw, validation="invalid", error=str(exc))
                if attempt:
                    raise
                if raw is not None:
                    messages.append({"role": "assistant", "content": raw})
                messages.append(pack_message("Correct ONLY the current task's response above. "
                    "Keep the same evidence and criteria; do not invent data to fix format. Error: "+str(exc)))
            else:
                if trace:
                    trace(attempt=attempt+1, raw=raw, validation="valid")
                return result

    def context(self, include_initial=True):
        text = "Retained task-progress summaries (fallible):\n"+json.dumps(self.summaries,ensure_ascii=False)
        if include_initial and self.initial_location is not None:
            text += "\nInitial four-view description at state 0, NOT current: "+json.dumps(self.initial_location,ensure_ascii=False)
            text += "\nAt state 0 the robot remained facing FRONT; this scan caused no motion."
        return text

    def images(self, history, begin=None, end=None):
        end = len(history)-1 if end is None else end
        begin = min(self.boundary, end) if begin is None else begin
        return [("State %d, FRONT%s" % (i, "; ACTUAL returned pose" if history[i].phase == "rollback" else ""),
                 history[i].image()) for i in range(begin, end+1)]

    def actions(self, history, end=None):
        end = len(history)-1 if end is None else end
        runs = []
        for i in range(end):
            a = history[i].action
            if runs and runs[-1][2] == a:
                runs[-1][1] = i+1
            else:
                runs.append([i, i+1, a])
        return "ACTUAL ACTIONS (state -> state):\n" + "\n".join(
            "%d -> %d: %s, %d actions" % (a,b,ACTIONS[c],b-a) for a,b,c in runs)

    def history_images(self, history):
        end = len(history)-1
        begin = max(0, end-WINDOW)
        anchors = sorted({step for row in self.summaries for step in row.get("anchor_steps", [])
                          if type(step) is int and 0 <= step < begin})[-6:]
        # Bound image count: recent observations plus up to six historical anchors.
        return [("State %d, HISTORICAL FRONT anchor; not the current view" % step,
                 history[step].image()) for step in anchors] + self.images(history, begin, end)

    def review_history_pairs(self, history):
        end = len(history)-1
        begin = max(0,end-WINDOW)
        ids = np.linspace(begin,end-1,min(4,end-begin),dtype=int).tolist() if end>begin else []
        reviews = {(r["state"],r["observed_at"]):r for r in self.position_reviews}
        images,metadata = [],[]
        for step in ids:
            frame = history[step]
            row = reviews.get((step,frame.physical_step))
            if row is not None:
                description = {k:row.get(k) for k in ("stage","completed","completion_status","location","history_match","reason")}
                source = "position_review"
            elif step==0 and frame.physical_step==0 and self.initial_location is not None:
                description = self.initial_location
                source = "initial_scan"
            else:
                description = "No position review exists for THIS exact frame. Read its image; do not borrow another frame's description."
                source = "not_available"
            label = ("HISTORICAL PAIR: state %d, physical step %d, FRONT image. " % (step,frame.physical_step))
            label += "The following description belongs ONLY to this paired historical image; it is fallible, not the current location.\n"
            label += "Any directional words in this historical description use THIS state's OLD heading, not the current heading. "
            label += "The paired image itself is FRONT RGB, not a four-view panorama.\n"
            label += "Description source: "+source+"\n"+json.dumps(description,ensure_ascii=False)
            images.append((label,frame.image()))
            metadata.append(dict(state=step,physical_step=frame.physical_step,description_source=source))
        return images,metadata

    def summarize(self, history):
        begin, end = self.boundary, self.boundary+WINDOW
        data = "Earlier progress:\n"+json.dumps(self.summaries, ensure_ascii=False)
        data += "\nNew interval %d..%d. Do not read beyond state %d.\n" % (begin,end,end)
        data += self.actions(history, end)
        result = self.ask(SUMMARY_PROMPT, data, self.images(history, begin, end))
        proposed = result.get("anchor_steps", [])
        anchors = []
        if isinstance(proposed, list):
            for value in proposed:
                if isinstance(value, str) and value.strip().isdigit():
                    value = int(value)
                if type(value) is int and begin <= value <= end and value not in anchors:
                    anchors.append(value)
        fallback = len(anchors) < 2
        for value in (begin, end):
            if len(anchors) < 2 and value not in anchors:
                anchors.append(value)
        return dict(text=result["summary"], grounding=result.get("grounding", ""),
                    anchor_steps=sorted(anchors[:2]), anchor_source="boundary_fallback" if fallback else "vlm")

    def locate(self, views, initial=False):
        return self.ask(LOCATION_PROMPT, "INITIALIZATION ONLY; navigation has not begun." if initial else
                        "Describe the actual current recovery position only.", list(views.items()))

    def direction(self, views, location, progress):
        return self.ask(DIRECTION_PROMPT, "Current location: "+json.dumps(location, ensure_ascii=False)+
                        "\nCurrent progress: "+json.dumps(progress, ensure_ascii=False),
                        list(views.items()))

    def plan(self, current, progress, location, alignment, history, views, project=None, trace=None):
        prompt = (REWRITE_PROMPT if CORRECTION_MODE == "instruction" else POINT_PROMPT) + "\n" + PLAN_INPUT_RULES
        data = self.context()+"\nProgress: "+json.dumps(progress, ensure_ascii=False)
        data += "\n"+self.actions(history)
        data += "\nLocation with PRE-ALIGNMENT labels: "+json.dumps(location, ensure_ascii=False)
        data += "\nAlignment already executed; selected old heading is now FRONT: "+json.dumps(alignment)
        data += "\nMaximum nearby ground distance: %.2f m." % self.args.point_max_distance
        images = self.history_images(history) + [("ACTUAL CURRENT "+name, image)
                                                 for name, image in views.items() if name != "front"]
        images += [("ACTUAL CURRENT FRONT: ONLY drawing canvas, after alignment", current.image())]
        def validate(value):
            validate_plan_reply(value)
            if CORRECTION_MODE == "point":
                value["target"] = project(value["point"])
            audit = self.ask(PLAN_SCOPE_PROMPT, "Progress: "+json.dumps(progress, ensure_ascii=False)+
                             "\nAlready executed alignment: "+json.dumps(alignment)+
                             "\nProposed plan: "+json.dumps(value, ensure_ascii=False))
            self.log(dict(phase="plan_scope", accepted=audit.get("accepted"), reason=audit.get("reason", "")))
            if audit.get("accepted") is not True:
                raise ValueError("local plan changes or fails to preserve the original task: "+str(audit.get("reason")))
            return value
        return self.ask(prompt, data, images, validator=validate, trace=trace)

    def checkpoint(self, history, reason="off_route"):
        candidates = list(range(self.boundary, len(history)-1))
        if not candidates:
            return None
        result = self.ask(CHECKPOINT_PROMPT, self.context()+"\nRecovery trigger: "+reason+"\nEligible state IDs: "+str(candidates),
                          self.images(history))
        step = result.get("step")
        return step if type(step) is int and step in candidates else None

    def review(self, history, mode="normal", views=None, alignment=None, proposal=None):
        route_enabled = mode == "normal"
        images,samples = self.review_history_pairs(history)
        data = self.context(include_initial=False)+"\n"+self.actions(history)
        data += "\nReview mode: %s; route_check_enabled=%s." % (mode,str(route_enabled).lower())
        data += "\nCURRENT state: %d, physical step: %d." % (len(history)-1,history[-1].physical_step)
        data += "\nSampled historical pairs from the previous at-most-16 valid states: "+json.dumps(samples)
        data += "\nRead each historical image with its own location description. Use the intervening motion to understand continuity with previously described regions in ANY direction; express the updated place naturally, without separate movement-change fields. The CURRENT FRONT image is last."
        data += "\nAvailable ACTUAL CURRENT image directions: "+json.dumps(list(views) if views is not None else ["front"])
        data += "; unsupported side/rear regions must remain unknown, not fabricated."
        data += "\nSummaries are coarse earlier context, not descriptions of the current image. Gaps in the samples do not prove a crossing."
        if alignment is not None:
            data += "\nAlready EXECUTED alignment; selected old heading is now FRONT: "+json.dumps(alignment)
        if proposal is not None:
            images.append(("HISTORICAL middle STOP proposal, physical step %d; not proof of actual return. Its own proposal reason: %s" %
                           (proposal[1].physical_step,proposal[2]),proposal[1].image()))
        if views is not None:
            images += [("ACTUAL CURRENT "+name+" at the SAME current position; this scan causes no motion",image)
                       for name,image in views.items() if name!="front"]
        current = views["front"] if views is not None else history[-1].image()
        images.append(("ACTUAL CURRENT FRONT: state %d, physical step %d. Infer THIS location from the earlier image/description pairs." %
                       (len(history)-1,history[-1].physical_step),current))
        self.last_review_failure = None
        notes = []
        def text_value(value):
            if isinstance(value,str):
                return value.strip()
            if isinstance(value,list):
                return "; ".join(filter(None,(text_value(x) for x in value)))
            if isinstance(value,dict) and value:
                return json.dumps(value,ensure_ascii=False)
            return ""
        def boolean(value,field):
            if type(value) is bool:
                return value
            if isinstance(value,str) and value.strip().lower() in ("true","false"):
                notes.append(field+": converted explicit true/false text to boolean")
                return value.strip().lower()=="true"
            raise ValueError("%s must be JSON true/false; received %r" % (field,value))
        def validate(value):
            notes.clear()
            pos,route,stop = (value.get(k) for k in ("position_review","route_check","stop_proposal"))
            if not all(isinstance(x,dict) for x in (pos,route,stop)):
                raise ValueError("return ONE object with position_review, route_check and stop_proposal objects")
            # Descriptions may be prose or a list. They are not control flags.
            completed = pos.get("completed")
            if isinstance(completed,list):
                clauses = [text_value(x) for x in completed]
                pos["completed"] = [x for x in clauses if x]
                pos["completion_status"] = "reported" if pos["completed"] else "not_reported" if completed else "none_confirmed"
                if any(not x for x in clauses):
                    notes.append("completed contains unspecified entries; no completion inferred from them")
            elif text_value(completed):
                pos["completed"] = [text_value(completed)]
                pos["completion_status"] = "reported"
            else:
                pos["completed"],pos["completion_status"] = [],"not_reported"
                notes.append("completed is missing/unspecified; this does NOT mean no tasks were completed")
            for key in ("stage","location","history_match","reason"):
                pos[key] = text_value(pos.get(key))
            if not pos["reason"]:
                pos["reason"] = pos["history_match"] or pos["location"] or pos["stage"]
                notes.append("no separate position reason; retained the model's available spatial explanation")
            for key in ("stage","location","history_match","reason"):
                if not pos[key]:
                    pos[key] = "Not supplied by VLM; no conclusion inferred for this field."
                    notes.append("position_review."+key+" was unspecified")
            stop["decision"] = str(stop.get("decision","")).strip().lower()
            if stop["decision"] not in ("stop","continue","uncertain"):
                raise ValueError("stop_proposal.decision must be stop/continue/uncertain; received %r" % stop["decision"])
            rejected = stop["decision"]=="continue"
            if "reject_vln_stop" in stop and stop["reject_vln_stop"] is not rejected:
                notes.append("reject_vln_stop derived from the single STOP decision; conflicting separate flag ignored")
            stop["reject_vln_stop"] = rejected
            if not route_enabled:
                route = value["route_check"] = dict(off_route=False,kind="disabled",reason="Route checking is disabled in this mode.")
            else:
                route["off_route"] = boolean(route.get("off_route"),"route_check.off_route")
                route["kind"] = str(route.get("kind","none" if not route["off_route"] else "")).strip().lower()
                if (route["kind"] not in ("none","wrong_place","wrong_direction","wrong_order")
                        or route["off_route"] != (route["kind"]!="none")):
                    raise ValueError("route kind must match off_route: false=none, true=wrong_place/wrong_direction/wrong_order")
                route["reason"] = text_value(route.get("reason"))
                if not route["reason"]:
                    route["reason"] = "VLM supplied no separate route rationale. Shared spatial explanation: "+pos["reason"]
                    notes.append("route rationale was not supplied separately")
            stop["reason"] = text_value(stop.get("reason"))
            if not stop["reason"]:
                stop["reason"] = "VLM supplied no separate STOP rationale. Shared spatial explanation: "+pos["reason"]
                notes.append("STOP rationale was not supplied separately")
            if stop["decision"]=="stop" and route["off_route"]:
                raise ValueError("shared review cannot both fulfill the whole task and report an incompatible route; resolve the contradiction from the evidence")
            # The VLM makes the semantic STOP judgment once. Description/checklist
            # wording must never silently turn its STOP into CONTINUE or vice versa.
            value = dict(
                position_review={k:pos[k] for k in ("stage","completed","completion_status","location","history_match","reason")},
                route_check={k:route[k] for k in ("off_route","kind","reason")},
                stop_proposal={k:stop[k] for k in ("decision","reject_vln_stop","reason")})
            if notes:
                value["reply_notes"] = list(notes)
            return value
        def trace(attempt,raw,validation,error=None):
            if validation=="invalid":
                try:
                    parsed = json_object(raw or "")
                except (ValueError,TypeError):
                    parsed = None
                self.last_review_failure = dict(error=error,raw_reply=raw,unaccepted_reply=parsed)
                self.log(dict(phase="governor_reply_rejected",physical_step=history[-1].physical_step,
                    effective_step=len(history)-1,review_mode=mode,history_sample=samples,attempt=attempt,**self.last_review_failure))
            else:
                self.last_review_failure = None
                if notes:
                    self.log(dict(phase="governor_reply_normalized",physical_step=history[-1].physical_step,
                        effective_step=len(history)-1,review_mode=mode,notes=list(notes),raw_reply=raw))
        # One generation for all three sections; only invalid control fields need a repair.
        result = self.ask(JOINT_REVIEW_PROMPT,data,images,tokens=self.args.review_max_new_tokens,
                          validator=validate,trace=trace)
        result["history_sample"] = samples
        self.position_reviews.append(dict(state=len(history)-1,observed_at=history[-1].physical_step,
                                          source=mode,**result["position_review"]))
        return result

@dataclass
class Task:
    instruction: str
    start: int = 0
    checkpoint: int = 0
    id: int = 0
    mode: str = "policy"
    target: object = None
    plan: dict = field(default_factory=dict)


class LegacyEpisode:
    def __init__(self, env, policy, planner, args, name, folder, log, video):
        self.env, self.policy, self.vlm, self.args = env, policy, planner, args
        self.name, self.folder, self.log, self.video = name, folder, log, video
        self.steps = self.deviations = self.corrections = self.rewrites = self.depth = 0
        self.stop_returns = self.stop_rejections = self.serial = self.point_runs = self.point_completed = 0
        self.run_action, self.forward_run, self.blocked = None, 0, 0
        self.last_stop = self.last_recovery = -1
        self.retry_at = 0
        self.review_cache = None
        self.blockage_recoveries = 0
        self.proposals, self.pictures = [], []
        self.termination = ""
        self.history = [Frame.capture(env.reset(), env.sim, 0, "initial")]
        self.current = self.history[0]
        self.tasks = [Task(planner.instruction)]

    @property
    def done(self):
        return bool(self.termination) or self.env.episode_over or self.steps >= self.args.max_steps

    def event(self, phase, **data):
        self.log(dict(physical_step=self.steps, effective_step=len(self.history)-1,
                      task_depth=len(self.tasks)-1, task_id=self.tasks[-1].id, phase=phase, **data))

    def call(self, stage, operation):
        try:
            return operation()
        except (ValueError, KeyError, TypeError) as exc:
            self.event("vlm_invalid", task=stage, reason=str(exc)[:600])
            return None

    def point_status(self, task):
        uv, visible = reproject_point(self.current,self.env.sim,self.policy.e,task.target)
        origin = self.history[task.start].position
        direction = (task.target-origin)[[0,2]]
        progress = float(np.dot((self.current.position-origin)[[0,2]],direction)/max(np.linalg.norm(direction),1e-6))
        advanced = (len(self.history)>1 and self.history[-2].action == 1 and
                    np.linalg.norm(self.current.position-self.history[-2].position) > .02)
        return dict(point=uv,in_view=visible,range_m=float(np.linalg.norm((task.target-self.current.position)[[0,2]])),
                    progress_m=progress,complete=bool(not visible and advanced and progress>.05))

    def move(self, action, phase, retain=True, controller=False):
        if self.done:
            return False
        before = self.current.position.copy()
        if retain:
            self.current.action = action
            if controller:
                self.current.query = True
        obs = self.env.step({"action": "governor_move_backward"} if action == 4 else action)
        self.steps += 1
        self.current = Frame.capture(obs, self.env.sim, self.steps, phase)
        if retain:
            self.history.append(self.current)
        if phase == "normal":
            self.blocked = self.blocked+1 if action == 1 and np.linalg.norm(self.current.position-before) < .10 else 0
        video_image = obs["rgb"]
        if phase == "correction" and self.tasks[-1].mode == "point":
            status = self.point_status(self.tasks[-1])
            self.event("point_tracking", **status)
            if status["in_view"]:
                video_image = np.asarray(draw_point(self.current,status["point"]))
        self.video.append_data(video_image)
        self.event(phase, action=ACTIONS[action], instruction=self.tasks[-1].instruction)
        return True

    def compress(self):
        while self.vlm.boundary+WINDOW <= len(self.history)-1:
            begin, end = self.vlm.boundary, self.vlm.boundary+WINDOW
            summary = self.call("summary", lambda: self.vlm.summarize(self.history))
            row = dict(start=begin, end=end, **(summary or dict(
                text="Task progress in this interval is unverified.", grounding="", anchor_steps=[begin,end],
                anchor_source="boundary_fallback")))
            self.vlm.summaries.append(row)
            self.event("history_compressed", **row)

    def activate(self, replay):
        self.run_action, self.forward_run, self.blocked = None, 0, 0
        self.compress()
        task = self.tasks[-1]
        if task.mode != "point":
            if replay:
                self.policy.restore(self.history[task.start:], task.instruction)
            else:
                self.policy.reset(self.current, task.instruction)

    def rollback(self, target, reason):
        saved = self.history[target]
        sequence = []
        for a, b in reversed(list(zip(self.history[target:], self.history[target+1:]))):
            if a.action == 1 and np.linalg.norm(b.position-a.position) <= 1e-5:
                continue
            sequence.append({1: 4, 2: 3, 3: 2}[a.action])
        self.proposals.clear()
        self.run_action, self.forward_run, self.blocked = None, 0, 0
        self.event("rollback_start", target=target, target_video_frame=saved.physical_step, reason=reason)
        for action in sequence:
            if not self.move(action, "rollback", retain=False):
                return False
        if self.done:
            return False
        self.history = self.history[:target] + [self.current]
        self.current.action, self.current.query, self.current.phase = None, True, "rollback"
        while len(self.tasks) > 1 and self.tasks[-1].start >= target:
            self.event("task_discarded", reason="rollback_to_or_before_task_start")
            self.tasks.pop()
        import quaternion
        position_error = float(np.linalg.norm(self.current.position-saved.position))
        angle_error = float(np.rad2deg(quaternion.rotation_intrinsic_distance(self.current.rotation,saved.rotation)))
        same_pose = position_error<.05 and angle_error<1.
        self.vlm.summaries = [x for x in self.vlm.summaries if x["end"]<target or same_pose and x["end"]==target]
        self.review_cache = None
        self.vlm.position_reviews = [r for r in self.vlm.position_reviews
                                    if r["state"]<target or same_pose and r["state"]==target]
        self.event("rollback_end", target=target,
                   position_error_m=round(position_error,4), rotation_error_deg=round(angle_error,3))
        return True

    def joint_review(self, mode=None, views=None, alignment=None, proposal=None):
        mode = mode or ("normal" if len(self.tasks)==1 else "correction")
        key = (self.steps,mode,views is not None)
        if self.review_cache is not None and self.review_cache[0]==key:
            return self.review_cache[1]
        review = self.call("governor_review",lambda:self.vlm.review(
            self.history,mode=mode,views=views,alignment=alignment,proposal=proposal))
        self.review_cache = (key,review)
        if review is None:
            failure = self.vlm.last_review_failure or {}
            self.event("governor_review",review_mode=mode,valid_reply=False,
                reason=failure.get("error","No usable review was returned."),
                unaccepted_reply=failure.get("unaccepted_reply"),raw_reply_event="governor_reply_rejected")
        else:
            self.event("governor_review",review_mode=mode,valid_reply=True,**review)
        return review

    def recovery_stop(self, review):
        proposal = review["stop_proposal"]
        if proposal["decision"] != "stop":
            return False
        self.event("recovery_decision", **proposal)
        self.proposals.clear()
        if self.move(0,"governor_stop",retain=False):
            self.termination = "governor_stop"
        return True

    def prepare(self, checkpoint, views=None, review=None):
        # A failed STOP recheck may request a new local task; replace, never nest.
        if len(self.tasks)>1:
            self.event("correction_replaced", reason="stop_review_requires_new_recovery")
            self.tasks = self.tasks[:1]
        self.corrections += 1
        def failed(reason):
            self.retry_at = self.steps+4
            self.event("correction_unavailable", reason=reason)
            if not self.done:
                self.activate(replay=True)
            return False
        views = four_views(self.env.sim) if views is None else views
        review = review or self.joint_review("recovery",views)
        if review is None:
            return failed("shared_review")
        progress = review["position_review"]
        location = dict(area=progress["location"])
        self.event("completed_tasks",history_end=len(self.history)-1,description=progress,
                   decision=review["stop_proposal"]["decision"],source="shared_review")
        if self.recovery_stop(review):
            return True
        choice = self.call("direction", lambda: self.vlm.direction(views, location, progress))
        if not choice or choice.get("direction") not in (*DIRECTIONS, "none"):
            return failed("direction")
        self.event("direction_selected", **choice)
        direction = "front" if choice["direction"] == "none" else choice["direction"]
        turns = TURNS[direction]
        alignment = dict(selected_direction=choice["direction"], current_front_was=direction,
                         reference_state=checkpoint, executed_actions=[ACTIONS[2 if turns > 0 else 3]]*abs(turns),
                         reason=choice.get("reason", ""))
        for _ in range(abs(turns)):
            if not self.move(2 if turns > 0 else 3, "alignment", controller=True):
                return False
        if self.done:
            return False
        self.event("alignment_complete", **alignment)
        if turns:
            views = four_views(self.env.sim)
            review = self.joint_review("recovery",views,alignment=alignment)
            if review is None:
                return failed("aligned_shared_review")
            progress = review["position_review"]
            self.event("completed_after_alignment",description=progress,source="shared_review")
            if self.recovery_stop(review):
                return True
        self.serial += 1
        point_attempt = 0
        def project(point):
            nonlocal point_attempt
            point_attempt += 1
            filename = "%s_step_%04d_effective_%04d_correction_%03d_attempt_%d.png" % (
                self.name,self.steps,len(self.history)-1,self.serial,point_attempt)
            gallery = Path(self.args.output_path)/"correction_images"
            draw_point(self.current,point,gallery/filename)
            self.pictures.append("correction_images/"+filename)
            self.event("point_overlay",image=self.pictures[-1],point=point,rollback_target=checkpoint)
            target = project_ground_point(self.current,self.env.sim,self.policy.e,point,self.args.point_max_distance)
            uv, visible = reproject_point(self.current,self.env.sim,self.policy.e,target)
            if not visible:
                raise ValueError("projected point is outside the current view; choose an interior ground pixel")
            self.event("point_target",world_point=target.tolist(),reprojected_point=uv,
                       ground_distance_m=float(np.linalg.norm((target-self.current.position)[[0,2]])))
            return target.tolist()
        plan = self.call("recovery_plan", lambda: self.vlm.plan(self.current,progress,location,alignment,
            self.history,views,project=project,
            trace=lambda **data: self.event("recovery_plan_reply",mode=CORRECTION_MODE,**data)))
        if plan is None:
            return failed("plan")
        task = Task(self.vlm.instruction,start=len(self.history)-1,checkpoint=checkpoint,
                    id=self.serial,mode=CORRECTION_MODE,plan=plan)
        if CORRECTION_MODE == "instruction":
            task.instruction = plan["instruction"].strip()
            self.rewrites += 1
        else:
            task.target = np.asarray(plan["target"],dtype=float)
            self.point_runs += 1
        self.tasks.append(task)
        self.depth = max(self.depth, len(self.tasks)-1)
        self.activate(replay=False)
        self.event("correction_start", instruction=task.instruction, mode=CORRECTION_MODE)
        return True

    def recover(self, early_stop=False, blocked=False):
        if len(self.tasks)>1:
            return  # Route recovery is allowed only during the original global task.
        reason = "early_stop" if early_stop else "blocked_motion" if blocked else "off_route"
        # Preserve a pending stopping region before any ordinary recovery can clear it.
        if self.proposals:
            return self.return_to_proposal(reason+"_with_stop_proposal")
        if not early_stop and (self.steps < self.retry_at or self.steps == self.last_recovery):
            return
        if early_stop and self.steps == self.last_recovery:
            self.termination = "rejected_stop_without_new_motion"
            return
        self.last_recovery = self.steps
        self.deviations += int(not early_stop and not blocked)
        self.blockage_recoveries += int(blocked)
        self.stop_rejections += int(early_stop)
        self.proposals.clear()
        target = len(self.history)-1 if early_stop else self.call("checkpoint", lambda: self.vlm.checkpoint(self.history,reason))
        if target is None:
            self.retry_at = self.steps+4
            self.event("correction_unavailable", reason="checkpoint")
            return
        if not self.rollback(target, reason):
            return
        prepared = self.prepare(target)
        self.last_recovery = self.steps
        if early_stop and not prepared and not self.done:
            self.termination = "rejected_stop_recovery_failed"

    def check_stop(self, trigger, review=None):
        if self.done or self.last_stop == self.steps:
            return False
        self.last_stop = self.steps
        review = review or self.joint_review()
        if review is None:
            return False  # Invalid is not CONTINUE; keep the proposal run unchanged.
        proposal = review["stop_proposal"]
        self.event("stop_proposal",trigger=trigger,**proposal)
        if proposal["decision"]=="stop":
            self.proposals.append((len(self.history)-1,self.current,proposal["reason"]))
            return False
        if proposal["decision"]!="continue" or not self.proposals:
            return False  # Uncertainty is not evidence of leaving the STOP region.
        return self.return_to_proposal("stop_to_continue")

    def return_to_proposal(self, trigger):
        proposal = self.proposals[len(self.proposals)//2]
        self.proposals.clear()
        self.stop_returns += 1
        self.event("stop_return",target=proposal[0],target_video_frame=proposal[1].physical_step,trigger=trigger)
        if not self.rollback(proposal[0],trigger):
            return True
        views = four_views(self.env.sim)
        review = self.joint_review("confirmation",views,proposal=proposal)
        if review and review["stop_proposal"]["decision"]=="stop":
            self.recovery_stop(review)
        else:
            # Reuse this shared judgment; no second completed-tasks STOP interpretation.
            self.prepare(proposal[0],views,review=review)
        return True  # Discard any action predicted before rollback.

    def finish_local(self):
        task = self.tasks[-1]
        self.point_completed += int(task.mode == "point")
        self.event("correction_end",reason="point_left_view" if task.mode == "point" else "local_policy_stop")
        self.tasks.pop()
        self.activate(replay=True)
        self.event("history_fused",checkpoint=task.checkpoint,
                   appended_actions=len(self.history)-1-task.checkpoint,instruction=self.tasks[-1].instruction)

    def next_action(self):
        task = self.tasks[-1]
        if task.mode != "point":
            return self.policy.predict(self.current)
        if self.point_status(task)["complete"]:
            return 0  # Local completion only, not a simulator STOP.
        return point_action(self.current,task.target)

    def run(self):
        self.event("episode_start", instruction=self.vlm.instruction)
        self.video.append_data(self.current.observations["rgb"])
        views = four_views(self.env.sim)
        self.vlm.initial_location = self.call("initial_location", lambda: self.vlm.locate(views, initial=True))
        self.event("initial_location", description=self.vlm.initial_location, current_heading="front", motion_steps=0)
        self.policy.reset(self.current, self.vlm.instruction)
        while not self.done:
            task = self.tasks[-1]
            action = self.next_action()
            if self.run_action in (2, 3) and action != self.run_action:
                self.run_action, self.forward_run = None, 0
                if self.check_stop("turn_run_ended"):
                    continue
            if action == 0:
                if len(self.tasks) > 1:
                    self.finish_local()
                else:
                    review = self.joint_review()
                    rejected = bool(review and review["stop_proposal"]["reject_vln_stop"])
                    self.event("global_stop_review",rejected=rejected,source="shared_review",
                        decision=review["stop_proposal"]["decision"] if review else "unavailable",
                        reason=review["stop_proposal"]["reason"] if review else "No usable review; native STOP is not vetoed.")
                    if rejected:
                        self.recover(early_stop=True)
                    else:
                        self.move(0, "global_stop", retain=False)
                        self.termination = "global_stop"
                continue
            self.move(action, "normal" if len(self.tasks) == 1 else "correction", controller=task.mode == "point")
            if task.mode != "point":
                self.policy.advance()
            if self.done:
                break
            self.forward_run = self.forward_run+1 if action == self.run_action == 1 else int(action == 1)
            self.run_action = action
            # Normal navigation gets ONE shared review every step. Local correction
            # requests it only at STOP sampling times, with route checking disabled.
            review = self.joint_review() if len(self.tasks)==1 else None
            off_route = bool(review and review["route_check"]["off_route"])
            blocked = len(self.tasks)==1 and self.blocked>=4
            if off_route:
                print("[governor] off_route step=%d: %s" % (self.steps,review["route_check"]["reason"]),flush=True)
            if blocked:
                self.event("motion_blocked",reason="Four consecutive FORWARD actions each moved less than 0.10 m.")
            if self.proposals or action==1 and self.forward_run%4==0:
                if self.check_stop("pending_proposal" if self.proposals else "four_forward_actions",review):
                    continue
            if off_route or blocked:
                self.recover(blocked=blocked and not off_route)
            elif task.mode=="point" and self.point_status(task)["complete"]:
                self.finish_local()
            else:
                self.compress()
        self.termination = self.termination or "max_steps"
        self.event("episode_end", reason=self.termination)
        return dict(steps=self.steps, effective_steps=len(self.history)-1, num_rewrites=self.rewrites,
                    corrections=self.corrections, deviations=self.deviations, blockage_recoveries=self.blockage_recoveries, stop_returns=self.stop_returns,
                    stop_rejections=self.stop_rejections, max_task_depth=self.depth,
                    point_runs=self.point_runs, point_completed=self.point_completed,
                    correction_images=self.pictures, final_instruction=self.tasks[-1].instruction,
                    termination_reason=self.termination, summaries=self.vlm.summaries,
                    intervened=bool(self.deviations or self.blockage_recoveries or self.stop_returns or self.stop_rejections or
                                    self.termination == "governor_stop"))

RESULT_FIELDS = ("scene_id", "episode_id", "success", "spl", "os", "ne", "steps",
                 "episode_instruction", "final_instruction", "num_rewrites")
CATEGORIES = ("intervened/success_to_success", "intervened/success_to_failure",
              "intervened/failure_to_success", "intervened/failure_to_failure",
              "unintervened/success", "unintervened/failure")


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    decoder, offset, rows = json.JSONDecoder(), 0, []
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        value, offset = decoder.raw_decode(text, offset)
        if isinstance(value, dict) and "scene_id" not in value:
            value = value.get("episodes", value.get("results", value))
        rows.extend(value if isinstance(value, list) else [value])
    return rows


def identity(row):
    scene = Path(str(row["scene_id"]).replace("\\", "/"))
    scene = scene.parent.name if scene.suffix == ".glb" else scene.name
    instruction = row.get("episode_instruction", row.get("instruction", ""))
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "")
    return scene, str(row["episode_id"]), " ".join(instruction.split())


def trajectory_name(row):
    scene, episode, text = identity(row)
    return re.sub(r"[^\w.-]", "_", scene+"_"+episode)+"_"+hashlib.sha256(text.encode()).hexdigest()[:12]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


class Records:
    def __init__(self, args):
        self.root = Path(args.output_path)
        self.root.mkdir(parents=True, exist_ok=True)
        if CORRECTION_MODE != 'topdown_map':
            (self.root/"correction_images").mkdir(exist_ok=True)
        config = dict(version=VERSION, mode=CORRECTION_MODE, settings={k:v for k,v in vars(args).items()
                      if k not in ("output_path", "max_episodes", "local_vlm_gpu")})
        config_path = self.root/"run_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Use a new output folder for this version/mode/settings")
        elif (self.root/"result.json").exists():
            raise ValueError("Use a new output folder; existing results have no matching run config")
        write_json(config_path, config)
        self.baseline, self.rows = {}, {}
        raw = read_rows(args.baseline_result_path)
        for row in raw:
            if isinstance(row, dict) and "scene_id" in row and "episode_id" in row:
                self.baseline.setdefault(identity(row)[:2], []).append(row)
        print("[governor] baseline: loaded %d episodes" % sum(map(len, self.baseline.values())), flush=True)
        for category in CATEGORIES:
            folder = self.root/category
            folder.mkdir(parents=True, exist_ok=True)
            for path in folder.glob("*/details.json"):
                row = json.loads(path.read_text(encoding="utf-8"))
                self.rows[identity(row)] = row

    def before(self, row):
        key = identity(row)
        candidates = self.baseline.get(key[:2], [])
        matching = [item for item in candidates if identity(item)[2] == key[2]]
        if not matching:
            matching = [item for item in candidates if not identity(item)[2]]
        if len(matching) != 1:
            raise ValueError("Missing/ambiguous baseline episode: "+str(key))
        return matching[0]

    def save(self, row, staging):
        baseline = self.before(row)
        before, after = baseline["success"] == 1, row["success"] == 1
        category = ("intervened/"+("success" if before else "failure")+"_to_"+("success" if after else "failure")
                    if row["intervened"] else "unintervened/"+("success" if after else "failure"))
        row.update(category=category, baseline={k:baseline.get(k) for k in ("success", "spl", "os", "ne")},
                   version=VERSION, correction_mode=CORRECTION_MODE)
        folder = self.root/category/trajectory_name(row)
        folder.mkdir(exist_ok=True)
        for path in staging.iterdir():
            if path.is_file():
                path.replace(folder/path.name)
        staging.rmdir()
        write_json(folder/"result.json", {k:row[k] for k in RESULT_FIELDS})
        write_json(folder/"details.json", row)
        self.rows[identity(row)] = row
        values = list(self.rows.values())
        summary = dict(type="summary", episodes=len(values), SR=sum(r["success"] for r in values)/len(values),
                       category_counts={c:sum(r["category"] == c for r in values) for c in CATEGORIES})
        temporary = self.root/"result.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for value in [*({k:r[k] for k in RESULT_FIELDS} for r in values), summary]:
                stream.write(json.dumps(value, ensure_ascii=False)+"\n")
        temporary.replace(self.root/"result.json")
        return category


def base_evaluate(evaluator, client, args, records):
    import imageio.v2 as imageio
    import torch
    from habitat.utils.visualizations import maps
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        env = evaluator.config_env()
    policy = StreamPolicy(evaluator)
    combined = (records.root/"trajectory_log.jsonl").open("a", encoding="utf-8")
    try:
        episodes = sorted(env.episodes, key=lambda e:e.scene_id)
        if args.max_episodes > 0:
            episodes = episodes[:args.max_episodes]
        for episode in episodes:
            row = dict(scene_id=Path(episode.scene_id).parent.name, episode_id=episode.episode_id,
                       episode_instruction=episode.instruction.instruction_text)
            records.before(row)
            if identity(row) in records.rows:
                continue
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            env.seed(args.seed)
            env.current_episode = episode
            name = trajectory_name(row)
            staging = records.root/"records"/name
            staging.mkdir(parents=True, exist_ok=True)
            with (staging/"steps.jsonl").open("w", encoding="utf-8") as log:
                def event(data):
                    log.write(json.dumps(data, ensure_ascii=False)+"\n")
                    combined.write(json.dumps({"scene_id":row["scene_id"], "episode_id":row["episode_id"], **data},
                                              ensure_ascii=False)+"\n")
                    log.flush()
                    combined.flush()
                video = imageio.get_writer(str(staging/"trajectory.mp4"), fps=6, quality=9)
                planner = Planner(client, row["episode_instruction"], args)
                planner.log = event
                try:
                    stats = Episode(env, policy, planner, args, name, staging, event, video).run()
                except Exception as exc:
                    event({"phase":"episode_error", "error":type(exc).__name__+": "+str(exc)})
                    raise
                finally:
                    video.close()
                    metrics = env.get_metrics()
                    topdown = metrics.get("top_down_map")
                    if topdown is not None:
                        Image.fromarray(maps.colorize_draw_agent_and_fit_to_height(topdown, 512)).save(staging/"topdown.png")
            row.update(stats, success=float(metrics["success"]), spl=float(metrics["spl"]),
                       os=float(metrics["oracle_success"]), ne=float(metrics["distance_to_goal"]))
            category = records.save(row, staging)
            print("[governor] %s steps: %d success: %s SPL: %s category: %s" %
                  (name, row["steps"], row["success"], row["spl"], category), flush=True)
    finally:
        combined.close()
        env.close()


def base_arguments(vlm_backend='local'):
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(save_video=True, dialogue_history_turns=0)
    p.add_argument("--model_path", default=str(ROOT/"StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"))
    p.add_argument("--habitat_config_path", default=str(ROOT/"config/vln_r2r.yaml"))
    p.add_argument("--habitat_root", default="/mnt/disk8t1/zwm/Progress-Aware-VLN/habitat-lab")
    p.add_argument("--eval_split", default="val_unseen")
    p.add_argument("--output_path", default=str(ROOT/"results"/("governor_remake_"+CORRECTION_MODE)))
    p.add_argument("--baseline_result_path", default=
                   "/mnt/disk8t1/zwm/StreamVLN/results_base/val_unseen/streamvln_baseline_full/result.json")
    p.add_argument("--max_episodes", type=int, default=-1)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--point_max_distance", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_history", type=int, default=8)
    p.add_argument("--num_future_steps", type=int, default=4)
    p.add_argument("--model_max_length", type=int, default=4096)
    p.add_argument("--local_vlm_python", default="/home/ubuntun/miniconda3/envs/vlnpa/bin/python")
    p.add_argument("--local_vlm_model_path", default="/mnt/disk8t1/zwm/Progress-Aware-VLN/Weights/Qwen3-VL-8B-Instruct")
    p.add_argument("--local_vlm_gpu", default="4")
    p.add_argument("--local_vlm_attn_implementation", default="flash_attention_2")
    p.add_argument("--local_vlm_max_pixels", type=int, default=802816)
    p.add_argument("--local_vlm_startup_timeout", type=float, default=600)
    p.add_argument("--local_vlm_request_timeout", type=float, default=300)
    p.add_argument("--init_max_new_tokens", type=int, default=1536)
    p.add_argument("--review_max_new_tokens", "--monitor_max_new_tokens", type=int, default=1536)
    args = p.parse_args()
    if min(args.max_steps, args.num_frames, args.num_history, args.num_future_steps) <= 0:
        p.error("step and history sizes must be positive")
    if args.point_max_distance <= .15:
        p.error("--point_max_distance must exceed 0.15 meters")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdecimal() or (vlm_backend=='local' and
            (not args.local_vlm_gpu.isdecimal() or int(visible) == int(args.local_vlm_gpu))):
        p.error("set different single physical GPUs for CUDA_VISIBLE_DEVICES and --local_vlm_gpu")
    return args


def base_main():
    if sys.argv[1:2] == ["--vlm-worker"]:
        return vlm_worker(sys.argv[2:])
    args = arguments()
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT/"streamvln"), str(ROOT)]
    # The confirmed local 0.2.4 source fixes the stale editable-install path.
    # This affects only the main process, not the VLM worker's Python environment.
    if importlib.util.find_spec("habitat") is None or importlib.util.find_spec("habitat_baselines") is None:
        source = Path(args.habitat_root)
        sys.path[:0] = [str(source/"habitat-lab"), str(source/"habitat-baselines")]
    records = Records(args)
    import torch
    import habitat
    import transformers
    from streamvln_eval import VLNEvaluator
    from model.stream_video_vln import StreamVLNForCausalLM
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.model_max_length,
                                                           padding_side="right")
    model = StreamVLNForCausalLM.from_pretrained(args.model_path,
        config=transformers.AutoConfig.from_pretrained(args.model_path), attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to("cuda:0").eval()
    model.reset(1)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        evaluator = VLNEvaluator(config_path=args.habitat_config_path, split=args.eval_split, env_num=1,
            output_path=args.output_path, model=model, tokenizer=tokenizer, epoch=0, args=args)
    with habitat.config.read_write(evaluator.config):
        evaluator.config.habitat.environment.max_episode_steps = args.max_steps
    sim = evaluator.config.habitat.simulator
    if sim.turn_angle != 15 or sim.forward_step_size != 0.25:
        raise ValueError("prompts require 15-degree turns and 0.25-meter forward actions")
    register_backward(evaluator.config)
    client = LocalVLM(args)
    try:
        print("[governor]", VERSION, "mode:", CORRECTION_MODE, flush=True)
        evaluate(evaluator, client, args, records)
    finally:
        client.close()
    return 0


# --- geometry.py ---
"""Metric depth projection, grid helpers and A*."""
import heapq
import math
from collections import deque
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
@dataclass
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    position: np.ndarray
    hfov: float
    min_depth: float
    max_depth: float
    normalized: bool
    step: int
    view: str = "front"

    def project(self, stride=4):
        depth = np.asarray(self.depth).squeeze().astype(np.float32)
        if self.normalized:
            depth = depth * (self.max_depth-self.min_depth) + self.min_depth
        h, w = depth.shape
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        d = depth[v, u]
        valid = np.isfinite(d) & (d > max(.02, self.min_depth)) & (d < self.max_depth-1e-4)
        u, v, d = u[valid], v[valid], d[valid]
        f = w / (2*np.tan(np.deg2rad(self.hfov)/2))
        # Habitat camera: +X right, +Y up, forward -Z; depth is axial Z.
        local = np.column_stack(((u-(w-1)/2)*d/f, -(v-(h-1)/2)*d/f, -d))
        return local @ self.camera_rotation.T + self.camera_position, v, u


def disk(radius):
    return [(dx, dz) for dx in range(-radius, radius+1)
            for dz in range(-radius, radius+1) if dx*dx+dz*dz <= radius*radius]


def components(mask):
    seen = np.zeros(mask.shape, bool)
    groups = []
    h, w = mask.shape
    for z, x in zip(*np.nonzero(mask)):
        if seen[z, x]:
            continue
        seen[z, x] = True
        todo, group = deque([(int(x), int(z))]), []
        while todo:
            a, b = todo.popleft()
            group.append((a, b))
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c, d = a+dx, b+dz
                if 0 <= c < w and 0 <= d < h and mask[d, c] and not seen[d, c]:
                    seen[d, c] = True
                    todo.append((c, d))
        groups.append(group)
    return groups


def inflate(mask, cells):
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dx, dz in disk(cells):
        x0, x1, z0, z1 = max(0, dx), min(w, w+dx), max(0, dz), min(h, h+dz)
        if x0 < x1 and z0 < z1:
            out[z0:z1, x0:x1] |= mask[z0-dz:z1-dz, x0-dx:x1-dx]
    return out


def astar(free, start, goal):
    h, w = free.shape
    def valid(p):
        return 0 <= p[0] < w and 0 <= p[1] < h and bool(free[p[1], p[0]])
    if not valid(start) or not valid(goal):
        return []
    queue = [(math.dist(start, goal), 0., start)]
    costs, parent = {start: 0.}, {}
    while queue:
        _, cost, p = heapq.heappop(queue)
        if cost > costs[p]:
            continue
        if p == goal:
            path = [p]
            while p in parent:
                p = parent[p]
                path.append(p)
            return path[::-1]
        for dx, dz in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            q = (p[0]+dx, p[1]+dz)
            if not valid(q):
                continue
            if dx and dz and (not valid((p[0]+dx,p[1])) or not valid((p[0],p[1]+dz))):
                continue  # Never cut an obstacle corner.
            nc = cost+math.hypot(dx,dz)
            if nc < costs.get(q, float('inf')):
                costs[q], parent[q] = nc, p
                heapq.heappush(queue,(nc+math.dist(q,goal),nc,q))
    return []


# --- rose2.py ---
"""ROSE2 adaptation with doorway and short-passage refinements; GPL-3.0."""
import heapq
import math
from collections import deque
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
ROSE2_LICENSE = '                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n\n Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>\n Everyone is permitted to copy and distribute verbatim copies\n of this license document, but changing it is not allowed.\n\n                            Preamble\n\n  The GNU General Public License is a free, copyleft license for\nsoftware and other kinds of works.\n\n  The licenses for most software and other practical works are designed\nto take away your freedom to share and change the works.  By contrast,\nthe GNU General Public License is intended to guarantee your freedom to\nshare and change all versions of a program--to make sure it remains free\nsoftware for all its users.  We, the Free Software Foundation, use the\nGNU General Public License for most of our software; it applies also to\nany other work released this way by its authors.  You can apply it to\nyour programs, too.\n\n  When we speak of free software, we are referring to freedom, not\nprice.  Our General Public Licenses are designed to make sure that you\nhave the freedom to distribute copies of free software (and charge for\nthem if you wish), that you receive source code or can get it if you\nwant it, that you can change the software or use pieces of it in new\nfree programs, and that you know you can do these things.\n\n  To protect your rights, we need to prevent others from denying you\nthese rights or asking you to surrender the rights.  Therefore, you have\ncertain responsibilities if you distribute copies of the software, or if\nyou modify it: responsibilities to respect the freedom of others.\n\n  For example, if you distribute copies of such a program, whether\ngratis or for a fee, you must pass on to the recipients the same\nfreedoms that you received.  You must make sure that they, too, receive\nor can get the source code.  And you must show them these terms so they\nknow their rights.\n\n  Developers that use the GNU GPL protect your rights with two steps:\n(1) assert copyright on the software, and (2) offer you this License\ngiving you legal permission to copy, distribute and/or modify it.\n\n  For the developers\' and authors\' protection, the GPL clearly explains\nthat there is no warranty for this free software.  For both users\' and\nauthors\' sake, the GPL requires that modified versions be marked as\nchanged, so that their problems will not be attributed erroneously to\nauthors of previous versions.\n\n  Some devices are designed to deny users access to install or run\nmodified versions of the software inside them, although the manufacturer\ncan do so.  This is fundamentally incompatible with the aim of\nprotecting users\' freedom to change the software.  The systematic\npattern of such abuse occurs in the area of products for individuals to\nuse, which is precisely where it is most unacceptable.  Therefore, we\nhave designed this version of the GPL to prohibit the practice for those\nproducts.  If such problems arise substantially in other domains, we\nstand ready to extend this provision to those domains in future versions\nof the GPL, as needed to protect the freedom of users.\n\n  Finally, every program is threatened constantly by software patents.\nStates should not allow patents to restrict development and use of\nsoftware on general-purpose computers, but in those that do, we wish to\navoid the special danger that patents applied to a free program could\nmake it effectively proprietary.  To prevent this, the GPL assures that\npatents cannot be used to render the program non-free.\n\n  The precise terms and conditions for copying, distribution and\nmodification follow.\n\n                       TERMS AND CONDITIONS\n\n  0. Definitions.\n\n  "This License" refers to version 3 of the GNU General Public License.\n\n  "Copyright" also means copyright-like laws that apply to other kinds of\nworks, such as semiconductor masks.\n\n  "The Program" refers to any copyrightable work licensed under this\nLicense.  Each licensee is addressed as "you".  "Licensees" and\n"recipients" may be individuals or organizations.\n\n  To "modify" a work means to copy from or adapt all or part of the work\nin a fashion requiring copyright permission, other than the making of an\nexact copy.  The resulting work is called a "modified version" of the\nearlier work or a work "based on" the earlier work.\n\n  A "covered work" means either the unmodified Program or a work based\non the Program.\n\n  To "propagate" a work means to do anything with it that, without\npermission, would make you directly or secondarily liable for\ninfringement under applicable copyright law, except executing it on a\ncomputer or modifying a private copy.  Propagation includes copying,\ndistribution (with or without modification), making available to the\npublic, and in some countries other activities as well.\n\n  To "convey" a work means any kind of propagation that enables other\nparties to make or receive copies.  Mere interaction with a user through\na computer network, with no transfer of a copy, is not conveying.\n\n  An interactive user interface displays "Appropriate Legal Notices"\nto the extent that it includes a convenient and prominently visible\nfeature that (1) displays an appropriate copyright notice, and (2)\ntells the user that there is no warranty for the work (except to the\nextent that warranties are provided), that licensees may convey the\nwork under this License, and how to view a copy of this License.  If\nthe interface presents a list of user commands or options, such as a\nmenu, a prominent item in the list meets this criterion.\n\n  1. Source Code.\n\n  The "source code" for a work means the preferred form of the work\nfor making modifications to it.  "Object code" means any non-source\nform of a work.\n\n  A "Standard Interface" means an interface that either is an official\nstandard defined by a recognized standards body, or, in the case of\ninterfaces specified for a particular programming language, one that\nis widely used among developers working in that language.\n\n  The "System Libraries" of an executable work include anything, other\nthan the work as a whole, that (a) is included in the normal form of\npackaging a Major Component, but which is not part of that Major\nComponent, and (b) serves only to enable use of the work with that\nMajor Component, or to implement a Standard Interface for which an\nimplementation is available to the public in source code form.  A\n"Major Component", in this context, means a major essential component\n(kernel, window system, and so on) of the specific operating system\n(if any) on which the executable work runs, or a compiler used to\nproduce the work, or an object code interpreter used to run it.\n\n  The "Corresponding Source" for a work in object code form means all\nthe source code needed to generate, install, and (for an executable\nwork) run the object code and to modify the work, including scripts to\ncontrol those activities.  However, it does not include the work\'s\nSystem Libraries, or general-purpose tools or generally available free\nprograms which are used unmodified in performing those activities but\nwhich are not part of the work.  For example, Corresponding Source\nincludes interface definition files associated with source files for\nthe work, and the source code for shared libraries and dynamically\nlinked subprograms that the work is specifically designed to require,\nsuch as by intimate data communication or control flow between those\nsubprograms and other parts of the work.\n\n  The Corresponding Source need not include anything that users\ncan regenerate automatically from other parts of the Corresponding\nSource.\n\n  The Corresponding Source for a work in source code form is that\nsame work.\n\n  2. Basic Permissions.\n\n  All rights granted under this License are granted for the term of\ncopyright on the Program, and are irrevocable provided the stated\nconditions are met.  This License explicitly affirms your unlimited\npermission to run the unmodified Program.  The output from running a\ncovered work is covered by this License only if the output, given its\ncontent, constitutes a covered work.  This License acknowledges your\nrights of fair use or other equivalent, as provided by copyright law.\n\n  You may make, run and propagate covered works that you do not\nconvey, without conditions so long as your license otherwise remains\nin force.  You may convey covered works to others for the sole purpose\nof having them make modifications exclusively for you, or provide you\nwith facilities for running those works, provided that you comply with\nthe terms of this License in conveying all material for which you do\nnot control copyright.  Those thus making or running the covered works\nfor you must do so exclusively on your behalf, under your direction\nand control, on terms that prohibit them from making any copies of\nyour copyrighted material outside their relationship with you.\n\n  Conveying under any other circumstances is permitted solely under\nthe conditions stated below.  Sublicensing is not allowed; section 10\nmakes it unnecessary.\n\n  3. Protecting Users\' Legal Rights From Anti-Circumvention Law.\n\n  No covered work shall be deemed part of an effective technological\nmeasure under any applicable law fulfilling obligations under article\n11 of the WIPO copyright treaty adopted on 20 December 1996, or\nsimilar laws prohibiting or restricting circumvention of such\nmeasures.\n\n  When you convey a covered work, you waive any legal power to forbid\ncircumvention of technological measures to the extent such circumvention\nis effected by exercising rights under this License with respect to\nthe covered work, and you disclaim any intention to limit operation or\nmodification of the work as a means of enforcing, against the work\'s\nusers, your or third parties\' legal rights to forbid circumvention of\ntechnological measures.\n\n  4. Conveying Verbatim Copies.\n\n  You may convey verbatim copies of the Program\'s source code as you\nreceive it, in any medium, provided that you conspicuously and\nappropriately publish on each copy an appropriate copyright notice;\nkeep intact all notices stating that this License and any\nnon-permissive terms added in accord with section 7 apply to the code;\nkeep intact all notices of the absence of any warranty; and give all\nrecipients a copy of this License along with the Program.\n\n  You may charge any price or no price for each copy that you convey,\nand you may offer support or warranty protection for a fee.\n\n  5. Conveying Modified Source Versions.\n\n  You may convey a work based on the Program, or the modifications to\nproduce it from the Program, in the form of source code under the\nterms of section 4, provided that you also meet all of these conditions:\n\n    a) The work must carry prominent notices stating that you modified\n    it, and giving a relevant date.\n\n    b) The work must carry prominent notices stating that it is\n    released under this License and any conditions added under section\n    7.  This requirement modifies the requirement in section 4 to\n    "keep intact all notices".\n\n    c) You must license the entire work, as a whole, under this\n    License to anyone who comes into possession of a copy.  This\n    License will therefore apply, along with any applicable section 7\n    additional terms, to the whole of the work, and all its parts,\n    regardless of how they are packaged.  This License gives no\n    permission to license the work in any other way, but it does not\n    invalidate such permission if you have separately received it.\n\n    d) If the work has interactive user interfaces, each must display\n    Appropriate Legal Notices; however, if the Program has interactive\n    interfaces that do not display Appropriate Legal Notices, your\n    work need not make them do so.\n\n  A compilation of a covered work with other separate and independent\nworks, which are not by their nature extensions of the covered work,\nand which are not combined with it such as to form a larger program,\nin or on a volume of a storage or distribution medium, is called an\n"aggregate" if the compilation and its resulting copyright are not\nused to limit the access or legal rights of the compilation\'s users\nbeyond what the individual works permit.  Inclusion of a covered work\nin an aggregate does not cause this License to apply to the other\nparts of the aggregate.\n\n  6. Conveying Non-Source Forms.\n\n  You may convey a covered work in object code form under the terms\nof sections 4 and 5, provided that you also convey the\nmachine-readable Corresponding Source under the terms of this License,\nin one of these ways:\n\n    a) Convey the object code in, or embodied in, a physical product\n    (including a physical distribution medium), accompanied by the\n    Corresponding Source fixed on a durable physical medium\n    customarily used for software interchange.\n\n    b) Convey the object code in, or embodied in, a physical product\n    (including a physical distribution medium), accompanied by a\n    written offer, valid for at least three years and valid for as\n    long as you offer spare parts or customer support for that product\n    model, to give anyone who possesses the object code either (1) a\n    copy of the Corresponding Source for all the software in the\n    product that is covered by this License, on a durable physical\n    medium customarily used for software interchange, for a price no\n    more than your reasonable cost of physically performing this\n    conveying of source, or (2) access to copy the\n    Corresponding Source from a network server at no charge.\n\n    c) Convey individual copies of the object code with a copy of the\n    written offer to provide the Corresponding Source.  This\n    alternative is allowed only occasionally and noncommercially, and\n    only if you received the object code with such an offer, in accord\n    with subsection 6b.\n\n    d) Convey the object code by offering access from a designated\n    place (gratis or for a charge), and offer equivalent access to the\n    Corresponding Source in the same way through the same place at no\n    further charge.  You need not require recipients to copy the\n    Corresponding Source along with the object code.  If the place to\n    copy the object code is a network server, the Corresponding Source\n    may be on a different server (operated by you or a third party)\n    that supports equivalent copying facilities, provided you maintain\n    clear directions next to the object code saying where to find the\n    Corresponding Source.  Regardless of what server hosts the\n    Corresponding Source, you remain obligated to ensure that it is\n    available for as long as needed to satisfy these requirements.\n\n    e) Convey the object code using peer-to-peer transmission, provided\n    you inform other peers where the object code and Corresponding\n    Source of the work are being offered to the general public at no\n    charge under subsection 6d.\n\n  A separable portion of the object code, whose source code is excluded\nfrom the Corresponding Source as a System Library, need not be\nincluded in conveying the object code work.\n\n  A "User Product" is either (1) a "consumer product", which means any\ntangible personal property which is normally used for personal, family,\nor household purposes, or (2) anything designed or sold for incorporation\ninto a dwelling.  In determining whether a product is a consumer product,\ndoubtful cases shall be resolved in favor of coverage.  For a particular\nproduct received by a particular user, "normally used" refers to a\ntypical or common use of that class of product, regardless of the status\nof the particular user or of the way in which the particular user\nactually uses, or expects or is expected to use, the product.  A product\nis a consumer product regardless of whether the product has substantial\ncommercial, industrial or non-consumer uses, unless such uses represent\nthe only significant mode of use of the product.\n\n  "Installation Information" for a User Product means any methods,\nprocedures, authorization keys, or other information required to install\nand execute modified versions of a covered work in that User Product from\na modified version of its Corresponding Source.  The information must\nsuffice to ensure that the continued functioning of the modified object\ncode is in no case prevented or interfered with solely because\nmodification has been made.\n\n  If you convey an object code work under this section in, or with, or\nspecifically for use in, a User Product, and the conveying occurs as\npart of a transaction in which the right of possession and use of the\nUser Product is transferred to the recipient in perpetuity or for a\nfixed term (regardless of how the transaction is characterized), the\nCorresponding Source conveyed under this section must be accompanied\nby the Installation Information.  But this requirement does not apply\nif neither you nor any third party retains the ability to install\nmodified object code on the User Product (for example, the work has\nbeen installed in ROM).\n\n  The requirement to provide Installation Information does not include a\nrequirement to continue to provide support service, warranty, or updates\nfor a work that has been modified or installed by the recipient, or for\nthe User Product in which it has been modified or installed.  Access to a\nnetwork may be denied when the modification itself materially and\nadversely affects the operation of the network or violates the rules and\nprotocols for communication across the network.\n\n  Corresponding Source conveyed, and Installation Information provided,\nin accord with this section must be in a format that is publicly\ndocumented (and with an implementation available to the public in\nsource code form), and must require no special password or key for\nunpacking, reading or copying.\n\n  7. Additional Terms.\n\n  "Additional permissions" are terms that supplement the terms of this\nLicense by making exceptions from one or more of its conditions.\nAdditional permissions that are applicable to the entire Program shall\nbe treated as though they were included in this License, to the extent\nthat they are valid under applicable law.  If additional permissions\napply only to part of the Program, that part may be used separately\nunder those permissions, but the entire Program remains governed by\nthis License without regard to the additional permissions.\n\n  When you convey a copy of a covered work, you may at your option\nremove any additional permissions from that copy, or from any part of\nit.  (Additional permissions may be written to require their own\nremoval in certain cases when you modify the work.)  You may place\nadditional permissions on material, added by you to a covered work,\nfor which you have or can give appropriate copyright permission.\n\n  Notwithstanding any other provision of this License, for material you\nadd to a covered work, you may (if authorized by the copyright holders of\nthat material) supplement the terms of this License with terms:\n\n    a) Disclaiming warranty or limiting liability differently from the\n    terms of sections 15 and 16 of this License; or\n\n    b) Requiring preservation of specified reasonable legal notices or\n    author attributions in that material or in the Appropriate Legal\n    Notices displayed by works containing it; or\n\n    c) Prohibiting misrepresentation of the origin of that material, or\n    requiring that modified versions of such material be marked in\n    reasonable ways as different from the original version; or\n\n    d) Limiting the use for publicity purposes of names of licensors or\n    authors of the material; or\n\n    e) Declining to grant rights under trademark law for use of some\n    trade names, trademarks, or service marks; or\n\n    f) Requiring indemnification of licensors and authors of that\n    material by anyone who conveys the material (or modified versions of\n    it) with contractual assumptions of liability to the recipient, for\n    any liability that these contractual assumptions directly impose on\n    those licensors and authors.\n\n  All other non-permissive additional terms are considered "further\nrestrictions" within the meaning of section 10.  If the Program as you\nreceived it, or any part of it, contains a notice stating that it is\ngoverned by this License along with a term that is a further\nrestriction, you may remove that term.  If a license document contains\na further restriction but permits relicensing or conveying under this\nLicense, you may add to a covered work material governed by the terms\nof that license document, provided that the further restriction does\nnot survive such relicensing or conveying.\n\n  If you add terms to a covered work in accord with this section, you\nmust place, in the relevant source files, a statement of the\nadditional terms that apply to those files, or a notice indicating\nwhere to find the applicable terms.\n\n  Additional terms, permissive or non-permissive, may be stated in the\nform of a separately written license, or stated as exceptions;\nthe above requirements apply either way.\n\n  8. Termination.\n\n  You may not propagate or modify a covered work except as expressly\nprovided under this License.  Any attempt otherwise to propagate or\nmodify it is void, and will automatically terminate your rights under\nthis License (including any patent licenses granted under the third\nparagraph of section 11).\n\n  However, if you cease all violation of this License, then your\nlicense from a particular copyright holder is reinstated (a)\nprovisionally, unless and until the copyright holder explicitly and\nfinally terminates your license, and (b) permanently, if the copyright\nholder fails to notify you of the violation by some reasonable means\nprior to 60 days after the cessation.\n\n  Moreover, your license from a particular copyright holder is\nreinstated permanently if the copyright holder notifies you of the\nviolation by some reasonable means, this is the first time you have\nreceived notice of violation of this License (for any work) from that\ncopyright holder, and you cure the violation prior to 30 days after\nyour receipt of the notice.\n\n  Termination of your rights under this section does not terminate the\nlicenses of parties who have received copies or rights from you under\nthis License.  If your rights have been terminated and not permanently\nreinstated, you do not qualify to receive new licenses for the same\nmaterial under section 10.\n\n  9. Acceptance Not Required for Having Copies.\n\n  You are not required to accept this License in order to receive or\nrun a copy of the Program.  Ancillary propagation of a covered work\noccurring solely as a consequence of using peer-to-peer transmission\nto receive a copy likewise does not require acceptance.  However,\nnothing other than this License grants you permission to propagate or\nmodify any covered work.  These actions infringe copyright if you do\nnot accept this License.  Therefore, by modifying or propagating a\ncovered work, you indicate your acceptance of this License to do so.\n\n  10. Automatic Licensing of Downstream Recipients.\n\n  Each time you convey a covered work, the recipient automatically\nreceives a license from the original licensors, to run, modify and\npropagate that work, subject to this License.  You are not responsible\nfor enforcing compliance by third parties with this License.\n\n  An "entity transaction" is a transaction transferring control of an\norganization, or substantially all assets of one, or subdividing an\norganization, or merging organizations.  If propagation of a covered\nwork results from an entity transaction, each party to that\ntransaction who receives a copy of the work also receives whatever\nlicenses to the work the party\'s predecessor in interest had or could\ngive under the previous paragraph, plus a right to possession of the\nCorresponding Source of the work from the predecessor in interest, if\nthe predecessor has it or can get it with reasonable efforts.\n\n  You may not impose any further restrictions on the exercise of the\nrights granted or affirmed under this License.  For example, you may\nnot impose a license fee, royalty, or other charge for exercise of\nrights granted under this License, and you may not initiate litigation\n(including a cross-claim or counterclaim in a lawsuit) alleging that\nany patent claim is infringed by making, using, selling, offering for\nsale, or importing the Program or any portion of it.\n\n  11. Patents.\n\n  A "contributor" is a copyright holder who authorizes use under this\nLicense of the Program or a work on which the Program is based.  The\nwork thus licensed is called the contributor\'s "contributor version".\n\n  A contributor\'s "essential patent claims" are all patent claims\nowned or controlled by the contributor, whether already acquired or\nhereafter acquired, that would be infringed by some manner, permitted\nby this License, of making, using, or selling its contributor version,\nbut do not include claims that would be infringed only as a\nconsequence of further modification of the contributor version.  For\npurposes of this definition, "control" includes the right to grant\npatent sublicenses in a manner consistent with the requirements of\nthis License.\n\n  Each contributor grants you a non-exclusive, worldwide, royalty-free\npatent license under the contributor\'s essential patent claims, to\nmake, use, sell, offer for sale, import and otherwise run, modify and\npropagate the contents of its contributor version.\n\n  In the following three paragraphs, a "patent license" is any express\nagreement or commitment, however denominated, not to enforce a patent\n(such as an express permission to practice a patent or covenant not to\nsue for patent infringement).  To "grant" such a patent license to a\nparty means to make such an agreement or commitment not to enforce a\npatent against the party.\n\n  If you convey a covered work, knowingly relying on a patent license,\nand the Corresponding Source of the work is not available for anyone\nto copy, free of charge and under the terms of this License, through a\npublicly available network server or other readily accessible means,\nthen you must either (1) cause the Corresponding Source to be so\navailable, or (2) arrange to deprive yourself of the benefit of the\npatent license for this particular work, or (3) arrange, in a manner\nconsistent with the requirements of this License, to extend the patent\nlicense to downstream recipients.  "Knowingly relying" means you have\nactual knowledge that, but for the patent license, your conveying the\ncovered work in a country, or your recipient\'s use of the covered work\nin a country, would infringe one or more identifiable patents in that\ncountry that you have reason to believe are valid.\n\n  If, pursuant to or in connection with a single transaction or\narrangement, you convey, or propagate by procuring conveyance of, a\ncovered work, and grant a patent license to some of the parties\nreceiving the covered work authorizing them to use, propagate, modify\nor convey a specific copy of the covered work, then the patent license\nyou grant is automatically extended to all recipients of the covered\nwork and works based on it.\n\n  A patent license is "discriminatory" if it does not include within\nthe scope of its coverage, prohibits the exercise of, or is\nconditioned on the non-exercise of one or more of the rights that are\nspecifically granted under this License.  You may not convey a covered\nwork if you are a party to an arrangement with a third party that is\nin the business of distributing software, under which you make payment\nto the third party based on the extent of your activity of conveying\nthe work, and under which the third party grants, to any of the\nparties who would receive the covered work from you, a discriminatory\npatent license (a) in connection with copies of the covered work\nconveyed by you (or copies made from those copies), or (b) primarily\nfor and in connection with specific products or compilations that\ncontain the covered work, unless you entered into that arrangement,\nor that patent license was granted, prior to 28 March 2007.\n\n  Nothing in this License shall be construed as excluding or limiting\nany implied license or other defenses to infringement that may\notherwise be available to you under applicable patent law.\n\n  12. No Surrender of Others\' Freedom.\n\n  If conditions are imposed on you (whether by court order, agreement or\notherwise) that contradict the conditions of this License, they do not\nexcuse you from the conditions of this License.  If you cannot convey a\ncovered work so as to satisfy simultaneously your obligations under this\nLicense and any other pertinent obligations, then as a consequence you may\nnot convey it at all.  For example, if you agree to terms that obligate you\nto collect a royalty for further conveying from those to whom you convey\nthe Program, the only way you could satisfy both those terms and this\nLicense would be to refrain entirely from conveying the Program.\n\n  13. Use with the GNU Affero General Public License.\n\n  Notwithstanding any other provision of this License, you have\npermission to link or combine any covered work with a work licensed\nunder version 3 of the GNU Affero General Public License into a single\ncombined work, and to convey the resulting work.  The terms of this\nLicense will continue to apply to the part which is the covered work,\nbut the special requirements of the GNU Affero General Public License,\nsection 13, concerning interaction through a network will apply to the\ncombination as such.\n\n  14. Revised Versions of this License.\n\n  The Free Software Foundation may publish revised and/or new versions of\nthe GNU General Public License from time to time.  Such new versions will\nbe similar in spirit to the present version, but may differ in detail to\naddress new problems or concerns.\n\n  Each version is given a distinguishing version number.  If the\nProgram specifies that a certain numbered version of the GNU General\nPublic License "or any later version" applies to it, you have the\noption of following the terms and conditions either of that numbered\nversion or of any later version published by the Free Software\nFoundation.  If the Program does not specify a version number of the\nGNU General Public License, you may choose any version ever published\nby the Free Software Foundation.\n\n  If the Program specifies that a proxy can decide which future\nversions of the GNU General Public License can be used, that proxy\'s\npublic statement of acceptance of a version permanently authorizes you\nto choose that version for the Program.\n\n  Later license versions may give you additional or different\npermissions.  However, no additional obligations are imposed on any\nauthor or copyright holder as a result of your choosing to follow a\nlater version.\n\n  15. Disclaimer of Warranty.\n\n  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY\nAPPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT\nHOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY\nOF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,\nTHE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR\nPURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM\nIS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF\nALL NECESSARY SERVICING, REPAIR OR CORRECTION.\n\n  16. Limitation of Liability.\n\n  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING\nWILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS\nTHE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY\nGENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE\nUSE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF\nDATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD\nPARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),\nEVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF\nSUCH DAMAGES.\n\n  17. Interpretation of Sections 15 and 16.\n\n  If the disclaimer of warranty and limitation of liability provided\nabove cannot be given local legal effect according to their terms,\nreviewing courts shall apply local law that most closely approximates\nan absolute waiver of all civil liability in connection with the\nProgram, unless a warranty or assumption of liability accompanies a\ncopy of the Program in return for a fee.\n\n                     END OF TERMS AND CONDITIONS\n\n            How to Apply These Terms to Your New Programs\n\n  If you develop a new program, and you want it to be of the greatest\npossible use to the public, the best way to achieve this is to make it\nfree software which everyone can redistribute and change under these terms.\n\n  To do so, attach the following notices to the program.  It is safest\nto attach them to the start of each source file to most effectively\nstate the exclusion of warranty; and each file should have at least\nthe "copyright" line and a pointer to where the full notice is found.\n\n    <one line to give the program\'s name and a brief idea of what it does.>\n    Copyright (C) <year>  <name of author>\n\n    This program is free software: you can redistribute it and/or modify\n    it under the terms of the GNU General Public License as published by\n    the Free Software Foundation, either version 3 of the License, or\n    (at your option) any later version.\n\n    This program is distributed in the hope that it will be useful,\n    but WITHOUT ANY WARRANTY; without even the implied warranty of\n    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n    GNU General Public License for more details.\n\n    You should have received a copy of the GNU General Public License\n    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\nAlso add information on how to contact you by electronic and paper mail.\n\n  If the program does terminal interaction, make it output a short\nnotice like this when it starts in an interactive mode:\n\n    <program>  Copyright (C) <year>  <name of author>\n    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w\'.\n    This is free software, and you are welcome to redistribute it\n    under certain conditions; type `show c\' for details.\n\nThe hypothetical commands `show w\' and `show c\' should show the appropriate\nparts of the General Public License.  Of course, your program\'s commands\nmight be different; for a GUI interface, you would use an "about box".\n\n  You should also get your employer (if you work as a programmer) or school,\nif any, to sign a "copyright disclaimer" for the program, if necessary.\nFor more information on this, and how to apply and follow the GNU GPL, see\n<https://www.gnu.org/licenses/>.\n\n  The GNU General Public License does not permit incorporating your program\ninto proprietary programs.  If your program is a subroutine library, you\nmay consider it more useful to permit linking proprietary applications with\nthe library.  If this is what you want to do, use the GNU Lesser General\nPublic License instead of this License.  But first, please read\n<https://www.gnu.org/licenses/why-not-lgpl.html>.\n'

# ROSE/ROSE2 headless adaptation, 2026-11-02.
# Derived from aislabunimi/ROSE2 (GPL-3.0): rose_v1_repo/fft_structure_extraction.py,
# util/layout.py, util/matrice.py, object/Segment.py and rose_v2_repo/minibatch.py.
# This adaptation uses polygonize instead of enumerating 3..7-sided cycles, and
# sparse graph components for the upstream DBSCAN eps=.85, min_samples=1 case.
# See embedded ROSE2_LICENSE and accompanying THIRD_PARTY_ROSE2.md.
# Structural pipeline restored from the user-supplied working reference 59981cdf.
class Rose2Partition:
    def __init__(self, resolution):
        self.res=resolution
        self.filter_level=.18
        self.line_support=.1
        self.edge_support=.1
        self.sigma=.125
        self.eps=.85
        self.spatial_merge_m=.25
        self.extended_merge_m=.35
        self.min_line_m=.6
        self.max_gap_m=.25
        self.max_lines=64
        self.max_faces=512
        self.passage_min_length=.5
        self.passage_max_length=3.5
        self.passage_max_width=2.2
        self.passage_min_side_area=1.5
        self.door_min_width=.5
        self.door_max_width=1.6
        self._libraries=None
        self._fft_sampling=None
        self._structure_cache=None
        self.confirmed_doors={}
        self.next_door=1

    def door_masks(self, grid):
        cuts=np.zeros_like(grid['free']);support=np.zeros_like(cuts)
        origin=grid['origin'];h,w=cuts.shape
        for door in self.confirmed_doors.get(grid.get('floor',0),[]):
            for field,array in (('boundary_cells',cuts),('support_cells',support)):
                cells=np.asarray(door[field],int)
                if not cells.size:continue
                local=cells-origin
                valid=(local[:,0]>=0)&(local[:,0]<w)&(local[:,1]>=0)&(local[:,1]<h)
                local=local[valid];array[local[:,1],local[:,0]]=True
        return cuts,support

    def apply_confirmed_doors(self, labels, grid, walls):
        """Replay retained geometric cuts; door pixels belong to either side, never a third region."""
        _,ndi,*_=self.libraries()
        result=labels.copy();cuts,support=self.door_masks(grid)
        known=grid.get('observed',grid['free']|grid['occupied'])
        barriers=walls|support|grid['occupied'];h,w=result.shape;next_id=int(result.max())+1;public=[]
        for door in self.confirmed_doors.get(grid.get('floor',0),[]):
            center=np.asarray(door['center_cell'],float)-grid['origin']
            if not(0<=center[0]<w and 0<=center[1]<h):continue
            normal=np.asarray(door['normal'],float);probes=[];sides=[]
            for sign in (-1,1):
                x,z=np.rint(center+sign*.45/self.res*normal).astype(int)
                probes.append((x,z));sides.append(int(result[z,x]) if 0<=x<w and 0<=z<h else 0)
            state='retained_boundary'
            if sides[0]>0 and sides[0]==sides[1]:
                source=sides[0];mask=result==source
                components_,number=ndi.label(mask&known&~barriers&~cuts)
                ends=[int(components_[z,x]) for x,z in probes]
                areas=np.bincount(components_[grid['free']].ravel(),minlength=number+1)*self.res**2
                if 0 not in ends and ends[0]!=ends[1] and min(areas[k] for k in ends)>=self.passage_min_side_area:
                    keep=max(ends,key=lambda k:areas[k]);other=ends[1] if keep==ends[0] else ends[0]
                    assigned=np.zeros_like(result)
                    assigned[components_==keep]=source;assigned[components_==other]=next_id
                    next_id+=1
                    seeds=assigned>0
                    _,indices=ndi.distance_transform_edt(~seeds,return_indices=True)
                    fringe=mask&~seeds
                    assigned[fringe]=assigned[tuple(indices[:,fringe])]
                    result[mask]=assigned[mask];state='retained_split'
            public.append(dict(id=door['id'],center_cell=door['center_cell'],width_m=door['width_m'],
                method='supported_entrance_gap' if door.get('kind')=='entrance' else 'supported_wall_gap',status=state,role='boundary_only',
                side_observed_floor_m2=door['side_observed_floor_m2']))
        return result,cuts&known,public

    def signature(self):
        return tuple(getattr(self,k) for k in ('res','filter_level','line_support','edge_support',
            'sigma','eps','spatial_merge_m','extended_merge_m','min_line_m','max_gap_m','max_lines','max_faces',
            'passage_min_length','passage_max_length','passage_max_width','passage_min_side_area',
            'door_min_width','door_max_width'))

    def libraries(self):
        if self._libraries is None:
            try:
                import cv2
                from scipy import ndimage
                from scipy.signal import find_peaks
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components
                from shapely.geometry import LineString, box
                from shapely.ops import polygonize, unary_union
            except ImportError as exc:
                raise RuntimeError('ROSE2 requires scipy, shapely and opencv-python-headless in the streamvln environment. '+str(exc)) from exc
            self._libraries=(cv2,ndimage,find_peaks,coo_matrix,connected_components,LineString,box,polygonize,unary_union)
        return self._libraries

    @staticmethod
    def angle_distance(a,b):
        return abs((a-b+np.pi/2)%np.pi-np.pi/2)

    def structure(self, occupied):
        """ROSE: polar FFT peaks -> opposite directions -> filtered structure."""
        _,ndi,find_peaks,*_=self.libraries()
        h,w=occupied.shape
        key=(occupied.shape,occupied.tobytes(),self.filter_level)
        if self._structure_cache is not None and self._structure_cache[0]==key:
            return self._structure_cache[1]
        n=max(h,w);n+=n%2
        image=np.zeros((n,n),float);image[:h,:w]=occupied
        spectrum=np.fft.fftshift(np.fft.fft2(image))
        amplitude=np.abs(spectrum);peak=float(amplitude.max())
        if peak==0:return np.zeros_like(occupied),[]
        # Vectorized version of helpers.topolar; paired frequency peaks select
        # structural orientations. DC affects every angle equally.
        if self._fft_sampling is None or self._fft_sampling[0]!=n:
            angles=np.linspace(0,2*np.pi,n,endpoint=False)
            radius=np.arange(n)*(.5*np.hypot(n,n)/n)
            rows=n/2+radius[:,None]*np.sin(angles)[None,:]
            cols=n/2+radius[:,None]*np.cos(angles)[None,:]
            yy,xx=np.mgrid[:n,:n];yy=yy-n/2;xx=xx-n/2
            self._fft_sampling=(n,angles,rows,cols,yy,xx)
        _,angles,rows,cols,yy,xx=self._fft_sampling
        polar=ndi.map_coordinates(amplitude/peak*255,[rows,cols],order=3,mode='constant')
        signal=polar.sum(axis=0)
        peaks,_=find_peaks(np.tile(signal,3),prominence=max(1e-9,float(np.ptp(signal))*.2))
        peaks=peaks[(peaks>=n)&(peaks<2*n)]-n
        selected=[]
        for index in sorted(peaks,key=lambda p:signal[p],reverse=True):
            angle=float(angles[index])
            opposite=(angle+np.pi)%(2*np.pi)
            paired=any(abs((float(angles[p])-opposite+np.pi)%(2*np.pi)-np.pi)<.1 and
                abs(signal[p]-signal[index])/max(1e-9,float(np.ptp(signal)))<.8 for p in peaks)
            if paired and all(self.angle_distance(angle,old)>.1 for old in selected):selected.append(angle%np.pi)
        # A scan with too little directional evidence is left unpartitioned.
        if len(selected)<2:
            result=(np.zeros_like(occupied),[]);self._structure_cache=(key,result)
            return result
        band=np.zeros((n,n),bool)
        for normal in selected:
            # Upstream par=50 with reference line length 20: 2.5 FFT pixels.
            band |= np.abs(-np.sin(normal)*xx+np.cos(normal)*yy)<=2.5
        score=np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum*band)))
        score*=image
        maximum=float(score.max())
        clean=(score[:h,:w]>=self.filter_level*maximum)&occupied if maximum else np.zeros_like(occupied)
        result=(clean,[(normal+np.pi/2)%np.pi for normal in selected])
        self._structure_cache=(key,result)
        return result

    @staticmethod
    def coverage(start,end,intervals):
        spans=sorted((max(start,a),min(end,b)) for a,b in intervals if b>start and a<end)
        total=0.;last=start
        for a,b in spans:
            if b>max(last,a):total+=b-max(last,a)
            last=max(last,b)
        return total/max(end-start,1e-9)

    def segments(self, clean, directions):
        cv2,*_=self.libraries()
        # Keyword arguments avoid the ambiguous positional Hough overload in upstream.
        lines=cv2.HoughLinesP(clean.astype(np.uint8)*255,1,np.pi/180,
            threshold=max(5,int(round(.8/self.res))),
            minLineLength=max(3,int(round(self.min_line_m/self.res))),
            maxLineGap=max(1,int(round(self.max_gap_m/self.res))))
        groups=[]
        if lines is None:return groups
        for raw in sorted(lines[:,0,:],key=lambda a:-float(np.hypot(a[2]-a[0],a[3]-a[1]))):
            a,b=np.asarray(raw[:2],float),np.asarray(raw[2:],float)
            theta=math.atan2(*(b-a)[::-1])%np.pi
            direction=min(directions,key=lambda t:self.angle_distance(t,theta))
            if self.angle_distance(direction,theta)>np.deg2rad(12):continue
            tangent=np.array([np.cos(direction),np.sin(direction)])
            normal=np.array([-tangent[1],tangent[0]])
            offset=float((a+b)@normal/2)
            interval=sorted((float(a@tangent),float(b@tangent)))
            match=next((g for g in groups if self.angle_distance(g['angle'],direction)<.05 and
                abs(g['offset']-offset)*self.res<=self.spatial_merge_m),None)
            if match is None:
                groups.append(dict(angle=direction,offset=offset,tangent=tangent,normal=normal,
                    intervals=[interval],length=interval[1]-interval[0]))
            else:
                length=interval[1]-interval[0];old=match['length']
                match['offset']=(match['offset']*old+offset*length)/(old+length)
                match['length']+=length;match['intervals'].append(interval)
        # Second spatial merge corresponds to upstream ExtendedSegment.merge_together.
        merged=[]
        for group in sorted(groups,key=lambda g:-g['length']):
            match=next((g for g in merged if self.angle_distance(g['angle'],group['angle'])<.05 and
                abs(g['offset']-group['offset'])*self.res<=self.extended_merge_m),None)
            if match is None:merged.append(group)
            else:match['intervals'].extend(group['intervals'])
        return merged

    def bounded_extensions(self, group, known, occupied, start, end, groups):
        """Extend measured intervals only through known cells up to the first wall.

        Grid-cell boundary events, rather than a fixed sampling stride, prevent
        skipping a one-cell wall. Disjoint supports never bridge an unknown run.
        The measured source intervals themselves are retained unchanged.
        """
        h,w=known.shape
        center=np.asarray(group['normal'],float)*group['offset']
        tangent=np.asarray(group['tangent'],float)
        intervals=[]
        for a,b in sorted(group['intervals']):
            a,b=max(start,float(a)),min(end,float(b))
            if b-a<=1e-9:continue
            if intervals and a<=intervals[-1][1]+1e-9:
                intervals[-1][1]=max(intervals[-1][1],b)
            else:intervals.append([a,b])
        stops=dict(unknown=0,wall=0,bounds=0)

        def wall_junction(u,v):
            # End at a supported wall intersection inside the first hit cell.
            # This lets polygonize join real walls without crossing their width.
            intersections=[]
            lo,hi=sorted((u,v))
            for other in groups:
                if other is group:continue
                normal=np.asarray(other['normal'],float)
                denominator=float(normal@tangent)
                if abs(denominator)<1e-9:continue
                t=float((other['offset']-normal@center)/denominator)
                if not lo-1e-8<=t<=hi+1e-8:continue
                along=float((center+t*tangent)@other['tangent'])
                if any(a-.5<=along<=b+.5 for a,b in other['intervals']):
                    intersections.append(t)
            return min(intersections,key=lambda t:abs(t-u)) if intersections else None

        def trace(anchor,limit):
            if abs(limit-anchor)<=1e-9:return anchor
            # Include every X/Z cell-edge crossing; coordinates refer to cell centers.
            events=[anchor,limit]
            lo,hi=sorted((anchor,limit))
            for axis,size in ((0,w),(1,h)):
                if abs(tangent[axis])<1e-12:continue
                values=(np.arange(size+1,dtype=float)-.5-center[axis])/tangent[axis]
                events.extend(values[(values>lo+1e-9)&(values<hi-1e-9)].tolist())
            events=sorted(set(events),reverse=limit<anchor)
            previous=None
            for index,(u,v) in enumerate(zip(events,events[1:])):
                if abs(v-u)<1e-9:continue
                point=center+(u+v)/2*tangent
                x,z=np.floor(point+.5).astype(int)
                if not(0<=x<w and 0<=z<h):
                    stops['bounds']+=1;return u
                # Supercover simultaneous corner crossings, including side cells
                # which the ray only touches at a corner of a thin wall.
                if previous is not None:
                    edge=center+u*tangent+.5
                    if np.all(np.abs(edge-np.rint(edge))<1e-7):
                        ex,ez=np.rint(edge).astype(int)
                        for cx,cz in ((ex-1,ez-1),(ex,ez-1),(ex-1,ez),(ex,ez)):
                            if (cx,cz) in (previous,(x,z)):continue
                            if not(0<=cx<w and 0<=cz<h):
                                stops['bounds']+=1;return u
                            if not known[cz,cx]:
                                stops['unknown']+=1;return u
                            if occupied[cz,cx]:
                                stops['wall']+=1;return u
                if not known[z,x]:
                    stops['unknown']+=1;return u
                if occupied[z,x]:
                    junction=wall_junction(u,v)
                    # The first cell may be the measured source wall endpoint.
                    # A different wall in that cell still terminates extension.
                    if index>0 or junction is not None:
                        stops['wall']+=1
                        return u if junction is None else junction
                previous=(x,z)
            stops['bounds']+=1
            return limit

        spans=[]
        for a,b in intervals:
            spans.append([min(a,trace(a,start)),max(b,trace(b,end))])
        # Combine only actual overlaps. An unknown gap remains a gap.
        merged=[]
        for a,b in sorted(spans):
            if merged and a<=merged[-1][1]+1e-9:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        return merged,stops

    def known_line_spans(self, center, tangent, start, end, known):
        """Clip any fitted line, including measured support, at unknown cells.

        Separate visible portions remain separate: an Hough interval spanning
        an unknown gap is not permission to connect the two sides of that gap.
        """
        if end-start<=1e-9:return []
        h,w=known.shape;events=[float(start),float(end)]
        for axis,size in ((0,w),(1,h)):
            if abs(tangent[axis])<1e-12:continue
            values=(np.arange(size+1,dtype=float)-.5-center[axis])/tangent[axis]
            events.extend(values[(values>start+1e-9)&(values<end-1e-9)].tolist())
        events=sorted(set(events));spans=[];previous=None
        for a,b in zip(events,events[1:]):
            if b-a<=1e-9:continue
            x,z=np.floor(center+(a+b)/2*tangent+.5).astype(int)
            if not(0<=x<w and 0<=z<h) or not known[z,x]:
                previous=None;continue
            join=bool(spans and abs(spans[-1][1]-a)<1e-8)
            edge=center+a*tangent+.5
            if join and previous is not None and np.all(np.abs(edge-np.rint(edge))<1e-7):
                ex,ez=np.rint(edge).astype(int)
                for cx,cz in ((ex-1,ez-1),(ex,ez-1),(ex-1,ez),(ex,ez)):
                    if (cx,cz) in (previous,(x,z)):continue
                    if not(0<=cx<w and 0<=cz<h) or not known[cz,cx]:join=False;break
            if join:spans[-1][1]=b
            else:spans.append([a,b])
            previous=(x,z)
        return spans

    def partition(self, grid):
        cv2,ndi,_,coo,connected,LineString,box,polygonize,union=self.libraries()
        occupied=grid['occupied'];known=grid.get('observed',grid['free']|occupied)
        labels=np.zeros(known.shape,np.int32);gates=np.zeros_like(known)
        info=dict(method='rose2',status='insufficient_structure',
            algorithm='ROSE FFT structure; ROSE2 weighted line arrangement and face clustering',
            filter_level=self.filter_level,line_support=self.line_support,edge_support=self.edge_support,
            sigma=self.sigma,dbscan_eps=self.eps,dbscan_min_samples=1,
            adaptation='headless partial-map port; polygonize; sparse DBSCAN(min_samples=1); no optional Voronoi',
            boundary_update='new geometry each observation; stable IDs/names matched after segmentation',
            doors=[],corridor_proposals=[])
        clean,directions=self.structure(occupied)
        retained_cuts,retained_support=self.door_masks(grid)
        clean=clean|retained_support  # Do not mutate the cached FFT result.
        for door in self.confirmed_doors.get(grid.get('floor',0),[]):
            angle=door['wall_angle']
            for candidate in (angle,(angle+np.pi/2)%np.pi):
                if all(self.angle_distance(candidate,old)>.1 for old in directions):directions=directions+[candidate]
        info['directions_deg']=[round(float(np.rad2deg(a)),2) for a in directions]
        groups=self.segments(clean,directions) if directions else []
        h,w=known.shape;extent=box(-.5,-.5,w-.5,h-.5)
        diagonal=2*math.hypot(h,w)
        extended=[];accepted=[];source_line_count=0
        extension_stops=dict(unknown=0,wall=0,bounds=0)
        for group in groups:
            center=group['normal']*group['offset'];tangent=group['tangent']
            clipped=LineString([center-diagonal*tangent,center+diagonal*tangent]).intersection(extent)
            if clipped.is_empty or clipped.geom_type!='LineString':continue
            pts=np.asarray(clipped.coords);start,end=sorted(pts@tangent)
            support=self.coverage(start,end,group['intervals'])
            if support>=self.line_support:
                source_line_count+=1
                spans,stops=self.bounded_extensions(group,known,occupied,start,end,groups)
                for a,b in spans:
                    for low,high in self.known_line_spans(center,tangent,a,b,known):
                        part=LineString([center+low*tangent,center+high*tangent]).intersection(extent)
                        if not part.is_empty:extended.append(part)
                for reason,count in stops.items():extension_stops[reason]+=count
                accepted.append(group)
            else:
                # Retain supported local portions of a weak extended line, as
                # create_short_ex_lines does upstream; never invent a long wall.
                for a,b in group['intervals']:
                    if min(b,end)-max(a,start)>1e-9:source_line_count+=1
                    for low,high in self.known_line_spans(center,tangent,max(a,start),min(b,end),known):
                        part=LineString([center+low*tangent,center+high*tangent]).intersection(extent)
                        if not part.is_empty:extended.append(part)
                accepted.append(group)
        info['line_groups']=len(groups);info['extended_lines']=len(extended)
        info['source_lines_before_unknown_clipping']=source_line_count
        info['line_extension']=dict(policy='stop_at_first_unknown_or_existing_wall',
                                    all_fitted_lines_clipped_to_known=True,stops=extension_stops)
        def retained_result():
            # A later FFT/Hough failure must not silently erase confirmed portals.
            if retained_cuts.any():
                base=known.astype(np.int32)
                restored,cut,doors=self.apply_confirmed_doors(base,grid,clean)
                info.update(current_detection_status=info['status'],status='retained_geometry',doors=doors,
                    room_clusters=int(len(np.unique(restored[restored>0]))))
                return restored,clean,cut,info
            return labels,clean,gates,info
        if not extended:return retained_result()
        # Cutting one line at unknown gaps creates pieces, not extra source lines.
        # Keep the original resource limit without penalizing correct clipping.
        if source_line_count>self.max_lines:
            info['status']='line_complexity_limit';return retained_result()

        def edge_weight(a,b):
            midpoint=(a+b)/2;angle=math.atan2(*(b-a)[::-1])%np.pi
            best=0.
            for group in accepted:
                if self.angle_distance(angle,group['angle'])>.05:continue
                if abs(midpoint@group['normal']-group['offset'])>.1:continue
                start,end=sorted((float(a@group['tangent']),float(b@group['tangent'])))
                best=max(best,self.coverage(start,end,group['intervals']))
            return best

        network=union(extended+[extent.boundary])
        edges=[];weights={}
        def key(a,b):return tuple(sorted((tuple(np.round(a,6)),tuple(np.round(b,6)))))
        for line in getattr(network,'geoms',[network]):
            pts=list(line.coords)
            for p,q in zip(pts,pts[1:]):
                a,b=np.asarray(p),np.asarray(q)
                if np.linalg.norm(a-b)<1e-7:continue
                boundary=any(abs(a[axis]-value)<1e-7 and abs(b[axis]-value)<1e-7
                    for axis,value in ((0,-.5),(0,w-.5),(1,-.5),(1,h-.5)))
                weight=1. if boundary else edge_weight(a,b)
                if weight>=self.edge_support:
                    edges.append(LineString([a,b]));weights[key(a,b)]=weight
        faces=list(polygonize(union(edges))) if edges else []
        info['faces_before_mask']=len(faces)
        if len(faces)>self.max_faces:
            info['status']='face_complexity_limit';return retained_result()
        if not faces:return retained_result()
        # Rasterize atomic faces once. Unknown pixels never become map evidence.
        atomic=np.zeros_like(labels);kept=[]
        for face in sorted(faces,key=lambda f:(-f.area,f.bounds)):
            exterior=np.rint(np.asarray(face.exterior.coords)).astype(np.int32)
            x0,y0=np.maximum(exterior.min(axis=0),[0,0]);x1,y1=np.minimum(exterior.max(axis=0)+1,[w,h])
            if x1<=x0 or y1<=y0:continue
            offset=np.array([x0,y0],np.int32)
            canvas=np.zeros((y1-y0,x1-x0),np.uint8)
            cv2.fillPoly(canvas,[exterior-offset],1)
            for hole in face.interiors:
                cv2.fillPoly(canvas,[np.rint(np.asarray(hole.coords)).astype(np.int32)-offset],0)
            local=atomic[y0:y1,x0:x1]
            inside=canvas.astype(bool)&known[y0:y1,x0:x1]&(local==0)
            if int((inside&grid['free'][y0:y1,x0:x1]).sum())*self.res**2<.2:continue
            local[inside]=len(kept)+1;kept.append(face)
        if not kept:return retained_result()
        # Shared arrangement edges define face adjacency, not Euclidean closeness.
        incidence={}
        for i,face in enumerate(kept):
            for ring in [face.exterior]+list(face.interiors):
                pts=list(ring.coords)
                for a,b in zip(pts,pts[1:]):incidence.setdefault(key(a,b),[]).append(i)
        affinity={}
        for edge,ids in incidence.items():
            if len(ids)!=2 or ids[0]==ids[1]:continue
            i,j=sorted(ids);weight=weights.get(edge)
            if weight is None:weight=edge_weight(np.array(edge[0]),np.array(edge[1]))
            # Upstream L_ij = exp(-wall_coverage/sigma), L_ii=1.
            affinity[i,j]=max(affinity.get((i,j),0.),math.exp(-weight/self.sigma))
        degree=np.ones(len(kept))
        for (i,j),value in affinity.items():degree[i]+=value;degree[j]+=value
        rows=[];cols=[]
        for (i,j),value in affinity.items():
            similarity=.5*value*(1/degree[i]+1/degree[j])
            if 1-similarity<=self.eps:rows.extend([i,j]);cols.extend([j,i])
        graph=coo((np.ones(len(rows)),(rows,cols)),shape=(len(kept),len(kept))).tocsr()
        count,clusters=connected(graph,directed=False)
        lookup=np.r_[0,clusters+1].astype(np.int32);labels=lookup[atomic]
        # Fill only residual bands inside each observed non-structural component.
        # No Euclidean jump across a wall/unknown gap to copy a room label.
        domain=known&~clean
        queue=deque(zip(*np.nonzero((labels>0)&inflate((labels==0)&domain,1))[::-1]))
        while queue:
            x,z=queue.popleft()
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                a,b=x+dx,z+dz
                if 0<=a<w and 0<=b<h and domain[b,a] and not labels[b,a]:
                    labels[b,a]=labels[z,x];queue.append((a,b))
        labels,door_cuts,door_proposals=self.split_doorways(labels,grid,clean,groups)
        gates |= door_cuts
        info['doors']=door_proposals
        info['door_diagnostics']=dict(self.last_door_diagnostics)
        labels,passages=self.split_short_passages(labels,grid,clean,directions)
        labels,boundary_passages,boundary_stats=self.split_boundary_passages(labels,grid,clean,directions)
        passages.extend(boundary_passages)
        info['boundary_passage_diagnostics']=boundary_stats
        info['corridor_proposals']=passages
        for dz,dx in ((0,1),(1,0)):
            a=labels[:h-dz,:w-dx];b=labels[dz:,dx:]
            boundary=(a>0)&(b>0)&(a!=b)
            gates[:h-dz,:w-dx]|=boundary;gates[dz:,dx:]|=boundary
        gates &= known&~occupied
        info.update(status='ok',faces=len(kept),room_clusters=int(len(np.unique(labels[labels>0]))),
            rose_room_clusters=int(count),short_passage_splits=len(passages))
        return labels,clean,gates,info

    def split_doorways(self, labels, grid, walls, groups):
        """Zero-length portals: collinear supported jambs and an observed free gap.

        A portal is a logical segmentation cut, never an occupancy obstacle.
        Room-sized floor on both sides and a connectivity check reject weak gaps.
        """
        _,ndi,*_=self.libraries()
        result,cuts,proposals=self.apply_confirmed_doors(labels,grid,walls)
        free=grid['free'];known=grid.get('observed',free|grid['occupied'])
        h,w=labels.shape;next_id=int(result.max())+1;seen=[]
        # Raw wall segments supply portal candidates only; they are NOT inserted
        # into the ROSE line arrangement or allowed to change its clustering.
        directions=[]
        for group in groups:
            if all(self.angle_distance(group['angle'],old)>.05 for old in directions):
                directions.append(group['angle'])
        raw_groups=self.segments(grid['occupied'],directions) if directions else []
        diagnostics=dict(filtered_groups=len(groups),raw_candidate_groups=len(raw_groups),
            paired_gaps=0,rejected_width=0,rejected_jamb=0,rejected_observed_opening=0,
            rejected_side_floor=0,rejected_widening=0,rejected_connectivity=0,accepted=0)
        self.last_door_diagnostics=diagnostics

        def widened_both_sides(center,normal,tangent,width):
            required=max(width*1.35,width+.4)
            cap=max(required,2*self.passage_max_width)
            for sign in (-1,1):
                consecutive=0;confirmed=False
                for distance in np.arange(.4,1.5+self.res,self.res):
                    p=center+sign*distance/self.res*normal
                    x,z=np.rint(p).astype(int)
                    if not(0<=x<w and 0<=z<h and known[z,x] and free[z,x] and not grid['occupied'][z,x]):
                        consecutive=0;continue
                    widths=[]
                    for side in (-1,1):
                        span=0.
                        for lateral in np.arange(self.res/2,cap/2+self.res,self.res/2):
                            x,z=np.rint(p+side*lateral/self.res*tangent).astype(int)
                            if not(0<=x<w and 0<=z<h and known[z,x] and free[z,x] and not grid['occupied'][z,x]):break
                            span=lateral
                        widths.append(span)
                    consecutive=consecutive+1 if sum(widths)>=required else 0
                    if consecutive>=2:confirmed=True;break
                if not confirmed:return False
            return True

        for group in list(groups)+raw_groups:
            intervals=[]
            for a,b in sorted(group['intervals']):
                if intervals and a-intervals[-1][1]<=self.max_gap_m/self.res:
                    intervals[-1][1]=max(intervals[-1][1],b)
                else:intervals.append([a,b])
            tangent=group['tangent'];normal=group['normal']
            for left,right in zip(intervals,intervals[1:]):
                diagnostics['paired_gaps']+=1
                width=(right[0]-left[1])*self.res
                # A wider room entrance is not rejected by the nominal door cap.
                # Reuse the existing connector width cap and require widening.
                if not self.door_min_width<=width<=max(self.door_max_width,self.passage_max_width):
                    diagnostics['rejected_width']+=1;continue
                entrance=width>self.door_max_width
                if min(left[1]-left[0],right[1]-right[0])*self.res<.5:
                    diagnostics['rejected_jamb']+=1;continue
                center=normal*group['offset']+tangent*(left[1]+right[0])/2
                if any(np.linalg.norm(center-old)*self.res<.4 for old in seen):continue
                a=normal*group['offset']+tangent*(left[1]-.15/self.res)
                b=normal*group['offset']+tangent*(right[0]+.15/self.res)
                opening=np.rint(np.linspace(normal*group['offset']+tangent*left[1],
                    normal*group['offset']+tangent*right[0],max(3,int(width/self.res)*2))).astype(int)
                valid=(opening[:,0]>=0)&(opening[:,0]<w)&(opening[:,1]>=0)&(opening[:,1]<h)
                if not valid.all():
                    diagnostics['rejected_observed_opening']+=1;continue
                opening_free=free[opening[:,1],opening[:,0]]&known[opening[:,1],opening[:,0]]&~grid['occupied'][opening[:,1],opening[:,0]]
                if float(opening_free.mean())<.7 or not known[opening[:,1],opening[:,0]].all():
                    diagnostics['rejected_observed_opening']+=1;continue
                # Seen floor must extend into both sides of the wall plane.
                sides=[];support=[]
                for sign in (-1,1):
                    probe=center+sign*.45/self.res*normal;x,z=np.rint(probe).astype(int)
                    sides.append(int(result[z,x]) if 0<=x<w and 0<=z<h and free[z,x] else 0)
                    support_cells=set()
                    for depth in np.arange(.2,1.5,self.res):
                        for lateral in np.arange(-.8,.8+self.res,self.res):
                            x,z=np.rint(center+sign*depth/self.res*normal+lateral/self.res*tangent).astype(int)
                            if 0<=x<w and 0<=z<h and free[z,x]:support_cells.add((int(x),int(z)))
                    support.append(len(support_cells)*self.res**2)
                if 0 in sides or min(support)<1.:
                    diagnostics['rejected_side_floor']+=1;continue
                if entrance and not widened_both_sides(center,normal,tangent,width):
                    diagnostics['rejected_widening']+=1;continue
                gate=np.zeros_like(walls)
                xy=np.rint(np.linspace(a,b,max(3,int(np.linalg.norm(b-a)*3)))).astype(int)
                valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                xy=xy[valid];gate[xy[:,1],xy[:,0]]=True
                gate=ndi.binary_dilation(gate,iterations=1)&known
                if sides[0]==sides[1]:
                    source=sides[0];domain=(result==source)&known&~walls&~grid['occupied']&~cuts&~gate
                    components_,number=ndi.label(domain)
                    areas=np.bincount(components_[free].ravel(),minlength=number+1)*self.res**2
                    endpoints=[]
                    for sign in (-1,1):
                        x,z=np.rint(center+sign*.45/self.res*normal).astype(int)
                        endpoints.append(int(components_[z,x]) if 0<=x<w and 0<=z<h else 0)
                    if 0 in endpoints or endpoints[0]==endpoints[1] or min(areas[k] for k in endpoints)<self.passage_min_side_area:
                        diagnostics['rejected_connectivity']+=1;continue
                    # A thin portal creates only its two adjoining spaces. Other
                    # fragments are attached to a side, not promoted to door regions.
                    substantial=endpoints
                    largest=max(substantial,key=lambda k:areas[k]);assigned=np.zeros_like(labels)
                    for component in substantial:
                        label=source if component==largest else next_id
                        if component!=largest:next_id+=1
                        assigned[components_==component]=label
                    seeds=assigned>0
                    _,indices=ndi.distance_transform_edt(~seeds,return_indices=True)
                    mask=result==source
                    assigned[mask&~seeds]=assigned[tuple(indices[:,mask&~seeds])]
                    result[mask]=assigned[mask]
                    mode='split'
                else:mode='existing_boundary'
                cuts |= gate;seen.append(center)
                diagnostics['accepted']+=1
                world_center=center+grid['origin']
                floor=grid.get('floor',0);memory=self.confirmed_doors.setdefault(floor,[])
                match=next((door for door in memory if np.linalg.norm(world_center-np.asarray(door['center_cell']))*self.res<.4
                    and self.angle_distance(door['wall_angle'],group['angle'])<np.deg2rad(15)),None)
                supported=set()
                for start,end in (left,right):
                    for t in np.linspace(start,end,max(2,int((end-start)*3)+1)):
                        x,z=np.rint(normal*group['offset']+tangent*t).astype(int)
                        if 0<=x<w and 0<=z<h and known[z,x]:
                            supported.add((int(x+grid['origin'][0]),int(z+grid['origin'][1])))
                if match is None:
                    zz,xx=np.nonzero(gate)
                    memory.append(dict(id='D%d'%self.next_door,
                        center_cell=np.rint(world_center).astype(int).tolist(),normal=normal.tolist(),
                        wall_angle=float(group['angle']),width_m=round(width,2),
                        boundary_cells=np.column_stack((xx+grid['origin'][0],zz+grid['origin'][1])).tolist(),
                        support_cells=[list(c) for c in sorted(supported)],
                        side_observed_floor_m2=[round(value,2) for value in support],
                        kind='entrance' if entrance else 'doorway',role='boundary_only'))
                    self.next_door+=1
                else:
                    match['support_cells']=[list(c) for c in sorted({tuple(c) for c in match['support_cells']}|supported)]
        # Reapply the complete accumulated set, not only this frame's detections.
        result,cuts,proposals=self.apply_confirmed_doors(result,grid,walls)
        return result,cuts,proposals

    def split_short_passages(self, labels, grid, walls, directions, _crop=True):
        """Separate a short, wall-bounded connector from substantial spaces at both ends.

        Multi-scale wide-space cores propose bottlenecks. Actual wall cross-sections,
        widening at BOTH ends and a connectivity test authorize each split. This is
        an extra partial-map refinement, not part of the upstream ROSE2 method.
        """
        _,ndi,*_=self.libraries()
        if _crop:
            result=labels.copy();proposals=[];next_id=int(labels.max())+1
            for source in np.unique(labels):
                if not source:continue
                zs,xs=np.nonzero(labels==source)
                if len(xs)*self.res**2<2*self.passage_min_side_area+.3:continue
                x0=max(0,int(xs.min())-2);x1=min(labels.shape[1],int(xs.max())+3)
                z0=max(0,int(zs.min())-2);z1=min(labels.shape[0],int(zs.max())+3)
                local=labels[z0:z1,x0:x1].copy();local[local!=source]=0
                local_grid={k:grid[k][z0:z1,x0:x1] for k in ('free','occupied')}
                local_grid['observed']=grid.get('observed',grid['free']|grid['occupied'])[z0:z1,x0:x1]
                local_grid['origin']=grid['origin']+np.array([x0,z0])
                split,rows=self.split_short_passages(local,local_grid,walls[z0:z1,x0:x1],directions,False)
                remap={}
                for label in np.unique(split):
                    if label and label!=source:remap[int(label)]=next_id;next_id+=1
                mapped=split.copy()
                for old,new in remap.items():mapped[split==old]=new
                region=local==source;result[z0:z1,x0:x1][region]=mapped[region]
                for row in rows:row['label']=remap[row['label']]
                proposals.extend(rows)
            return result,proposals
        result=labels.copy();proposals=[];next_id=int(labels.max())+1
        known=grid.get('observed',grid['free']|grid['occupied'])
        h,w=labels.shape
        wall_hits=ndi.binary_dilation(walls,iterations=1)
        for source in np.unique(labels):
            if not source:continue
            mask=result==source;domain=mask&~walls
            if int(domain.sum())*self.res**2<2*self.passage_min_side_area+.3:continue
            clearance=ndi.distance_transform_edt(domain)*self.res
            accepted=False
            for cap in (self.passage_max_width,self.passage_max_width*.75,self.passage_max_width*.5):
                if accepted:break
                cores,num=ndi.label(domain&(clearance>=cap/2+.15))
                sizes=np.bincount(cores.ravel());valid=np.flatnonzero(sizes*self.res**2>=.12)
                valid=valid[valid!=0]
                if len(valid)<2:continue
                core_mask=np.isin(cores,valid)
                distance,indices=ndi.distance_transform_edt(~core_mask,return_indices=True)
                nearest=cores[tuple(indices)]
                borders={}
                for dz,dx in ((0,1),(1,0)):
                    a=nearest[:h-dz,:w-dx];b=nearest[dz:,dx:]
                    eligible=domain[:h-dz,:w-dx]&domain[dz:,dx:]&(a!=b)&(a>0)&(b>0)
                    zs,xs=np.nonzero(eligible)
                    for x,z in zip(xs,zs):
                        if not grid['free'][z,x] or not .2<=clearance[z,x]<=cap/2+self.res:continue
                        pair=tuple(sorted((int(a[z,x]),int(b[z,x]))))
                        borders.setdefault(pair,[]).append((float(distance[z,x]),int(x),int(z)))

                def section(center,axis):
                    side=np.array([-axis[1],axis[0]])
                    cx,cz=np.rint(center).astype(int)
                    if not(0<=cx<w and 0<=cz<h and domain[cz,cx]):return None
                    ends=[];closed=[]
                    for sign in (-1,1):
                        endpoint=None;hit=False
                        for d in np.arange(.5,2*self.passage_max_width/self.res+.5,.5):
                            point=center+sign*d*side;x,z=np.rint(point).astype(int)
                            if not(0<=x<w and 0<=z<h):break
                            if wall_hits[z,x]:endpoint=point;hit=True;break
                            if not known[z,x] or not mask[z,x]:break
                            endpoint=point
                        if endpoint is None:return None
                        ends.append(endpoint);closed.append(hit)
                    return dict(ends=ends,width=float(np.linalg.norm(ends[1]-ends[0])*self.res),closed=all(closed))

                for pair,points in sorted(borders.items(),key=lambda item:min(p[0] for p in item[1]))[:8]:
                    if accepted:break
                    centers=[]
                    for _,x,z in sorted(points):
                        if all(math.hypot(x-p[0],z-p[1])*self.res>=.4 for p in centers):centers.append(np.array([x,z],float))
                        if len(centers)>=4:break
                    for center in centers:
                        if accepted:break
                        for theta in directions:
                            axis=np.array([np.cos(theta),np.sin(theta)])
                            middle=section(center,axis)
                            if middle is None or not middle['closed'] or not .45<=middle['width']<=cap+.2:continue
                            wider=max(middle['width']*1.45,middle['width']+.45)
                            runs=[];mouths=[]
                            for sign in (-1,1):
                                run=[(center,middle)];mouth=None
                                for travel in np.arange(self.res,self.passage_max_length+self.res,self.res):
                                    point=center+sign*travel/self.res*axis
                                    cross=section(point,axis)
                                    if cross is None:break
                                    if cross['width']>=wider:
                                        confirm=section(point+sign*max(.2,self.res)/self.res*axis,axis)
                                        if confirm is not None and confirm['width']>=wider:mouth=point;break
                                    if not cross['closed'] or cross['width']>cap+.3:break
                                    run.append((point,cross))
                                runs.append(run);mouths.append(mouth)
                            if any(m is None for m in mouths):continue
                            length=float(np.linalg.norm(runs[1][-1][0]-runs[0][-1][0])*self.res+self.res)
                            if not self.passage_min_length<=length<=self.passage_max_length:continue
                            # Build the actual narrow strip from supported cross-sections.
                            strip=np.zeros_like(mask)
                            for point,cross in runs[0]+runs[1]:
                                a,b=cross['ends'];steps=max(2,int(np.linalg.norm(b-a)*2)+1)
                                xy=np.rint(np.linspace(a,b,steps)).astype(int)
                                valid_xy=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                                xy=xy[valid_xy];strip[xy[:,1],xy[:,0]]=True
                            strip=ndi.binary_dilation(strip,iterations=1)&mask&~walls
                            remaining,number=ndi.label(domain&~strip)
                            areas=np.bincount(remaining.ravel())*self.res**2
                            ends=[]
                            for sign,mouth in zip((-1,1),mouths):
                                point=mouth+sign*max(.2,2*self.res)/self.res*axis
                                x,z=np.rint(point).astype(int)
                                ends.append(int(remaining[z,x]) if 0<=x<w and 0<=z<h else 0)
                            # Reject furniture pinches with an alternate open route.
                            if 0 in ends or ends[0]==ends[1] or min(areas[e] for e in ends)<self.passage_min_side_area:continue
                            substantial=[k for k in range(1,number+1) if areas[k]>=self.passage_min_side_area]
                            if len(substantial)<2:continue
                            largest=max(substantial,key=lambda k:areas[k])
                            for component in substantial:
                                if component==largest:continue
                                result[remaining==component]=next_id;next_id+=1
                            corridor_id=next_id;next_id+=1;result[strip]=corridor_id
                            # Only boundary/support pixels remain unassigned to a lobe.
                            seeds=(remaining>0)&np.isin(remaining,substantial)
                            _,nearest_seed=ndi.distance_transform_edt(~seeds,return_indices=True)
                            fringe=mask&~strip&~seeds
                            result[fringe]=result[tuple(nearest_seed[:,fringe])]
                            proposals.append(dict(label=corridor_id,length_m=round(length,2),
                                width_m=round(middle['width'],2),method='short_connector',
                                evidence='opposing structural walls, widening at both ends, separated substantial spaces',
                                entrance_cell=np.rint(mouths[0]+grid['origin']).astype(int).tolist(),
                                exit_cell=np.rint(mouths[1]+grid['origin']).astype(int).tolist()))
                            accepted=True;break
        return result,proposals

    def split_boundary_passages(self, labels, grid, walls, directions):
        """Split a narrow appendage whose entrance is an existing region boundary.

        The other room need not be inside the same ROSE2 cluster. Both the
        external neighbor and internal destination must have observed widening;
        a label transition by itself is never a doorway or corridor.
        """
        _,ndi,*_=self.libraries()
        result=labels.copy();proposals=[];next_id=int(labels.max())+1
        stats=dict(region_contacts=0,entrance_candidates=0,wall_supported=0,
            widened_outside=0,widened_inside=0,length_supported=0,
            destination_supported=0,bypass_rejected=0,accepted=0)
        observed=grid.get('observed',grid['free']|grid['occupied'])
        sizes=np.bincount(labels.ravel())*self.res**2
        for source in np.unique(labels):
            if not source or sizes[source]<self.passage_min_side_area+.3:continue
            zs,xs=np.nonzero(labels==source)
            pad=math.ceil(max(1.5,self.passage_max_width*1.6)/self.res)
            x0=max(0,int(xs.min())-pad);x1=min(labels.shape[1],int(xs.max())+pad+1)
            z0=max(0,int(zs.min())-pad);z1=min(labels.shape[0],int(zs.max())+pad+1)
            context=labels[z0:z1,x0:x1]
            mask=context==source;known=observed[z0:z1,x0:x1]
            free=grid['free'][z0:z1,x0:x1]
            structural=walls[z0:z1,x0:x1]
            hits=ndi.binary_dilation(structural,iterations=1)
            domain=mask&known&~structural
            h,w=mask.shape
            # Only observed, unobstructed contacts to a substantial neighbor.
            contacts={}
            for dz,dx in ((0,1),(1,0)):
                a=context[:h-dz,:w-dx];b=context[dz:,dx:]
                usable=free[:h-dz,:w-dx]&free[dz:,dx:]
                usable &= ~structural[:h-dz,:w-dx]&~structural[dz:,dx:]
                for own,other in ((a,b),(b,a)):
                    zz,xx=np.nonzero(usable&(own==source)&(other>0)&(other!=source))
                    for x,z in zip(xx,zz):
                        neighbor=int(other[z,x])
                        if sizes[neighbor]>=self.passage_min_side_area:
                            contacts.setdefault(neighbor,[]).append((x+dx/2,z+dz/2))
            if not contacts:continue
            stats['region_contacts']+=len(contacts)

            def section(center,axis,region):
                cx,cz=np.rint(center).astype(int)
                if not(0<=cx<w and 0<=cz<h and known[cz,cx] and
                       context[cz,cx]==region and not structural[cz,cx]):return None
                side=np.array([-axis[1],axis[0]])
                ends=[];closed=[]
                for sign in (-1,1):
                    endpoint=None;hit=False
                    for d in np.arange(.5,2*self.passage_max_width/self.res+.5,.5):
                        point=center+sign*d*side;x,z=np.rint(point).astype(int)
                        if not(0<=x<w and 0<=z<h):break
                        if hits[z,x]:endpoint=point;hit=True;break
                        if not known[z,x] or context[z,x]!=region:break
                        endpoint=point
                    if endpoint is None:return None
                    ends.append(endpoint);closed.append(hit)
                return dict(ends=ends,width=float(np.linalg.norm(ends[1]-ends[0])*self.res),closed=all(closed))

            accepted=False
            for neighbor,points in sorted(contacts.items(),key=lambda item:-len(item[1]))[:6]:
                if accepted:break
                contact=np.zeros_like(mask)
                for x,z in points:
                    x,z=np.rint([x,z]).astype(int)
                    if 0<=x<w and 0<=z<h:contact[z,x]=True
                groups,n=ndi.label(ndi.binary_dilation(contact,iterations=1))
                centers=[]
                for group in range(1,n+1):
                    zz,xx=np.nonzero(contact&(groups==group))
                    if len(xx)<2:continue
                    mean=np.array([xx.mean(),zz.mean()])
                    xy=np.column_stack((xx,zz));centers.append(xy[np.argmin(np.linalg.norm(xy-mean,axis=1))].astype(float))
                for entrance in centers[:8]:
                    if accepted:break
                    stats['entrance_candidates']+=1
                    for theta in directions:
                        if accepted:break
                        tangent=np.array([np.cos(theta),np.sin(theta)])
                        for sign in (-1,1):
                            axis=sign*tangent
                            start=entrance+max(.2,2*self.res)/self.res*axis
                            middle=section(start,axis,source)
                            if middle is None or not middle['closed'] or not .45<=middle['width']<=self.passage_max_width:continue
                            stats['wall_supported']+=1
                            wider=max(middle['width']*1.45,middle['width']+.45)
                            # Verify actual broad floor beyond the existing boundary.
                            outside=None
                            for distance in np.arange(.2,1.5+self.res,self.res):
                                point=entrance-distance/self.res*axis
                                cross=section(point,axis,neighbor)
                                further=section(point-max(.2,self.res)/self.res*axis,axis,neighbor)
                                if cross is not None and further is not None and min(cross['width'],further['width'])>=wider:
                                    outside=point;break
                            if outside is None:continue
                            stats['widened_outside']+=1
                            run=[];destination=None
                            # Include the source-side entrance band in the strip.
                            for distance in np.arange(0.,self.passage_max_length+self.res,self.res):
                                point=entrance+distance/self.res*axis
                                cross=section(point,axis,source)
                                if cross is None:
                                    if distance<=max(.2,2*self.res):continue
                                    break
                                if cross['width']>=wider:
                                    further=section(point+max(.2,self.res)/self.res*axis,axis,source)
                                    if further is not None and further['width']>=wider:
                                        destination=point+max(.2,self.res)/self.res*axis;break
                                if not cross['closed'] or cross['width']>self.passage_max_width+.3:break
                                run.append((point,cross))
                            if destination is None or not run:continue
                            stats['widened_inside']+=1
                            length=float(np.linalg.norm(run[-1][0]-entrance)*self.res+self.res)
                            if not self.passage_min_length<=length<=self.passage_max_length:continue
                            stats['length_supported']+=1
                            strip=np.zeros_like(mask)
                            for _,cross in run:
                                a,b=cross['ends'];count=max(2,int(np.linalg.norm(b-a)*2)+1)
                                xy=np.rint(np.linspace(a,b,count)).astype(int)
                                valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                                xy=xy[valid];strip[xy[:,1],xy[:,0]]=True
                            strip=ndi.binary_dilation(strip,iterations=1)&domain
                            if int(strip.sum())*self.res**2<.25:continue
                            remaining,_=ndi.label(domain&~strip)
                            areas=np.bincount(remaining.ravel())*self.res**2
                            x,z=np.rint(destination).astype(int)
                            end=int(remaining[z,x]) if 0<=x<w and 0<=z<h else 0
                            if not end or areas[end]<self.passage_min_side_area:continue
                            stats['destination_supported']+=1
                            # Reject a furniture pinch if one can bypass the strip
                            # from the same entrance contact to the internal space.
                            yy,xx=np.ogrid[:h,:w]
                            vicinity=(xx-entrance[0])**2+(yy-entrance[1])**2<=(.6/self.res)**2
                            entrance_floor=ndi.binary_dilation(context==neighbor,iterations=1)&domain&vicinity
                            if np.any((remaining==end)&entrance_floor):
                                stats['bypass_rejected']+=1;continue
                            result[z0:z1,x0:x1][strip]=next_id
                            proposals.append(dict(label=next_id,method='boundary_short_connector',
                                neighbor_label=int(neighbor),length_m=round(length,2),width_m=round(middle['width'],2),
                                evidence='existing region contact; opposing walls; observed widening outside and inside; no local bypass',
                                entrance_cell=np.rint(entrance+grid['origin']+[x0,z0]).astype(int).tolist(),
                                exit_cell=np.rint(destination+grid['origin']+[x0,z0]).astype(int).tolist()))
                            next_id+=1;accepted=True;break
        stats['accepted']=len(proposals)
        return result,proposals,stats


# --- mapping.py ---
"""Accumulated observed map, targets, region association, rendering and export."""
import heapq
import math
from collections import deque
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
class TopdownMap:
    def __init__(self, resolution=.01, radius=.18, height=1.5, stride=2, max_cells=9000):
        self.res, self.radius, self.height, self.stride = resolution, radius, height, stride
        self.max_cells = max_cells
        # Independent acceptance bounds; never shift floor/obstacle coordinates.
        self.floor_min_height = -.30
        self.observed_min_height = -.30
        self.voxels = set()  # quantized visible surface centers; shared by all floor layers
        # MapNav-style independent projection channels, fused by per-cell maximum.
        # Explored space is NOT permission to walk there.
        self.explored, self.floor_evidence, self.obstacle_evidence = {}, {}, {}
        self.floors, self.active, self.transition = {}, None, False
        self.pending = []
        self.position = np.zeros(3)
        self.forward = np.array([0.,0.,-1.])
        self.route, self.objects, self.regions, self.semantics = [], {}, {}, {}
        self.error_routes=[]
        self.visited, self.collisions = {}, {}
        self.version, self.next_object, self.next_region = 0, 1, 1
        self.seen = set()
        self.last_floor_pose = None
        self.observed_views={}
        self.observation_stats={}
        self._partition_cache=None
        self._render_cache=None
        self._grid_cache=None
        self._metadata_cache=None
        self._views_cache=None
        self._voxel_layers={}
        self._saved_snapshot=None
        self.naming_counts={}
        self.naming_epoch=0
        self.profile=dict(partition_builds=0,partition_hits=0,render_builds=0,render_hits=0)
        self.rose2=Rose2Partition(self.res)
        self.room_wall_height=.8
        self.room_wall_length=1.2
        self.room_wall_gap=.2
        self.room_door_min=.6
        self.room_door_max=1.4
        self.room_side_area=1.5
        self.room_corridor_width=2.2
        self.room_corridor_length=2.0

    def cell(self, p):
        return tuple(int(x) for x in np.floor(np.asarray(p)[[0,2]]/self.res))

    def world(self, cell, floor=None):
        floor = self.active if floor is None else floor
        return np.array([(cell[0]+.5)*self.res,self.floors[floor],(cell[1]+.5)*self.res])

    def locate_floor(self, position):
        y = float(position[1])
        if not self.floors:
            self.floors[0] = y
        near = min(self.floors, key=lambda k: abs(self.floors[k]-y))
        if abs(self.floors[near]-y) <= .22:
            self.active, self.transition, self.pending = near, False, []
        else:
            self.transition = True
            # Turns at a stair tread must not create a floor. Require spatially
            # separated, approximately level observations before adding a layer.
            if self.last_floor_pose is None or np.linalg.norm((position-self.last_floor_pose)[[0,2]]) > .15:
                self.pending.append(np.asarray(position).copy())
                self.pending = self.pending[-4:]
                self.last_floor_pose = np.asarray(position).copy()
            if len(self.pending) == 4:
                ys = [p[1] for p in self.pending]
                span = np.linalg.norm((self.pending[-1]-self.pending[0])[[0,2]])
                if max(ys)-min(ys) < .10 and span >= .55 and abs(self.floors[near]-y) >= .65:
                    self.active = max(self.floors)+1
                    self.floors[self.active] = float(np.median(ys))
                    self.transition, self.pending = False, []
        return None if self.transition else self.active

    def set_pose(self, position, forward, step, retain=True):
        self.position, self.forward = np.asarray(position).copy(), np.asarray(forward).copy()
        floor = self.locate_floor(self.position)
        if floor is not None:
            c = self.cell(position)
            known = self.visited.setdefault(floor,set())
            known.update((c[0]+dx,c[1]+dz) for dx,dz in disk(max(1,int(self.radius/self.res))))
        if retain:
            self.route.append(dict(step=int(step),floor=floor,position=self.position.tolist()))
        self.version += 1
        return floor

    def update(self, obs, detections=()):
        key = (obs.step,obs.view)
        if key in self.seen:
            return
        self.seen.add(key)
        points,v,u = obs.project(self.stride)
        before=len(self.voxels)
        incoming=set(map(tuple,np.floor(points/self.res).astype(int).tolist()))
        added=incoming-self.voxels
        for layer in self._voxel_layers.values():
            if layer['count']==before and layer['source']==id(self.voxels) and layer['scale']==(self.res,self.height,self.floor_min_height,self.observed_min_height):
                self.extend_voxel_layer(layer,added)
                layer['count']+=len(added)
        self.voxels.update(incoming)
        stats=dict(step=obs.step,view=obs.view,projected_points=len(points),
            new_voxels=len(self.voxels)-before,camera_position=obs.camera_position.tolist(),
            camera_forward=(-obs.camera_rotation[:,2]).tolist(),floor_points=0)
        self.observation_stats[obs.view]=stats
        floor = None if self.transition else self.active
        if floor is not None:
            ground=self.floors[floor]
            dy=points[:,1]-ground
            for channel,keep,threshold in (
                    (self.explored,(dy>=self.observed_min_height)&(dy<=2.3),1.),
                    (self.floor_evidence,(dy>=self.floor_min_height)&(dy<=.12),1.),
                    (self.obstacle_evidence,(dy>=.16)&(dy<=self.height),3.)):
                cells,counts=np.unique(np.floor(points[keep][:,[0,2]]/self.res).astype(int),axis=0,return_counts=True)
                memory=channel.setdefault(floor,{})
                for cell,count in zip(cells,counts):
                    c=tuple(map(int,cell));memory[c]=max(memory.get(c,0.),min(1.,float(count)/threshold))
            selected=(dy>=self.floor_min_height)&(dy<=.12)
            stats['floor_points']=int(selected.sum())
            cells=np.floor(points[selected][:,[0,2]]/self.res).astype(int)
            h,w=obs.depth.squeeze().shape
            samples=[[int(c[0]),int(c[1]),float(x/max(1,w-1)),float(y/max(1,h-1))]
                     for c,x,y in zip(cells,u[selected],v[selected])]
            self.observed_views[obs.view]=dict(step=obs.step,floor=floor,samples=samples)
            # Fill measured floor surface between adjacent valid depth samples;
            # this is not a convex hull or an inference from the room outline.
            stats['interpolated_floor_cells']=self.fill_floor_samples(obs,points,v,u,ground,floor)
            for item in detections:
                mask = np.asarray(item['mask'],bool)
                if mask.shape != obs.depth.squeeze().shape:
                    mask = np.asarray(Image.fromarray(mask).resize(
                        obs.depth.squeeze().shape[::-1],Image.Resampling.NEAREST))
                selected = points[mask[v,u]]
                if len(selected)<5:
                    continue
                # Reject isolated mask/depth islands before locating an instance.
                projected=np.floor(selected[:,[0,2]]/self.res).astype(int)
                lo=projected.min(axis=0);hi=projected.max(axis=0)
                footprint=np.zeros((hi[1]-lo[1]+1,hi[0]-lo[0]+1),bool)
                footprint[projected[:,1]-lo[1],projected[:,0]-lo[0]]=True
                groups=components(inflate(footprint,1))
                if not groups:continue
                main=set(max(groups,key=len))
                selected=selected[[tuple(c-lo) in main for c in projected]]
                if len(selected)<5:continue
                center = np.median(selected,axis=0)
                dist = np.linalg.norm(selected-center,axis=1)
                selected = selected[dist<=max(.15,float(np.quantile(dist,.85)))]
                center = np.median(selected,axis=0)
                cells = set(map(tuple,np.floor(selected[:,[0,2]]/self.res).astype(int).tolist()))
                match = None
                for oid, obj in self.objects.items():
                    if obj['target_id']!=item['target_id'] or obj['floor']!=floor:
                        continue
                    overlap = len(cells & obj['cells'])/max(1,min(len(cells),len(obj['cells'])))
                    if overlap>.15 and np.linalg.norm(center-np.array(obj['center']))<1.5:
                        match=oid
                        break
                if match is None:
                    match='O%d'%self.next_object
                    self.next_object+=1
                    self.objects[match]=dict(target_id=item['target_id'],label=item['label'],
                        floor=floor,cells=set(),center=center.tolist(),score=0.,last_seen=obs.step,
                        cell_scores={},observations=0)
                obj=self.objects[match]
                obj['cells'].update(cells)
                scores=obj.setdefault('cell_scores',{})
                for c in cells:
                    key='%d,%d'%c;scores[key]=max(scores.get(key,0.),float(item['score']))
                # Locate the accumulated connected footprint, not the latest mask alone.
                center[[0,2]]=np.median(np.asarray(list(obj['cells'])),axis=0)*self.res+self.res/2
                obj['observations']=obj.get('observations',0)+1
                obj.update(center=center.tolist(),score=max(obj['score'],float(item['score'])),last_seen=obs.step,
                    label=item['label'],query=item.get('query',item['label']),identity_status='candidate')
        self.version += 1

    def fill_floor_samples(self, obs, points, v, u, ground, floor):
        import cv2
        h,w=obs.depth.squeeze().shape
        shape=(math.ceil(h/self.stride),math.ceil(w/self.stride))
        lattice=np.full((*shape,3),np.nan,np.float32)
        dy=points[:,1]-ground
        keep=(dy>=self.floor_min_height)&(dy<=.12)
        lattice[v[keep]//self.stride,u[keep]//self.stride]=points[keep]
        corners=np.stack((lattice[:-1,:-1],lattice[:-1,1:],lattice[1:,1:],lattice[1:,:-1]),axis=2)
        valid=np.isfinite(corners).all(axis=(2,3))
        edges=corners-np.roll(corners,1,axis=2)
        valid &= (np.linalg.norm(edges,axis=3)<=.75).all(axis=2)
        quads=corners[valid]
        if not len(quads):return 0
        cells=np.floor(quads[:,:,[0,2]]/self.res).astype(np.int32)
        lo=cells.min(axis=(0,1));hi=cells.max(axis=(0,1))
        canvas=np.zeros((int(hi[1]-lo[1]+1),int(hi[0]-lo[0]+1)),np.uint8)
        cv2.fillPoly(canvas,list(cells-lo),1)
        zs,xs=np.nonzero(canvas)
        memory=self.floor_evidence.setdefault(floor,{})
        explored=self.explored.setdefault(floor,{})
        added=0
        for x,z in zip((xs+lo[0]).tolist(),(zs+lo[1]).tolist()):
            cell=(x,z);added+=int(cell not in memory)
            memory[cell]=1.;explored[cell]=1.
        return added

    def truncate_route(self, physical_step):
        self.route=[r for r in self.route if r['step']<=physical_step]
        # Intentionally retain surfaces, observed footprints, objects and semantic masks.
        self.version+=1

    def archive_route(self, physical_step, reason):
        """Preserve withdrawn physical history separately from the effective route."""
        rows=[copy.deepcopy(r) for r in self.route if r['step']>=physical_step]
        if len(rows)>1:
            self.error_routes.append(dict(kind='withdrawn',reason=reason,points=rows))
            self.version+=1

    def record_rollback(self, before, after):
        if before['step']==after['step']:return
        if self.error_routes and self.error_routes[-1]['kind']=='rollback' and self.error_routes[-1]['points'][-1]==before:
            self.error_routes[-1]['points'].append(copy.deepcopy(after))
        else:self.error_routes.append(dict(kind='rollback',reason='physical return to checkpoint',points=copy.deepcopy([before,after])))
        self.version+=1

    def mark_collision(self, endpoint):
        if not self.transition:
            self.collisions.setdefault(self.active,set()).add(self.cell(endpoint))
            self.version+=1

    def grid(self, full=False):
        floor=self.active
        ground=self.floors[floor]
        key=(self.version,floor,ground,full,self.res,self.radius,self.height,self.max_cells,
            self.floor_min_height,self.observed_min_height,
            id(self.voxels),len(self.voxels),len(self.visited.get(floor,())),len(self.collisions.get(floor,())))
        if self._grid_cache is not None and self._grid_cache[0]==key:
            return dict(self._grid_cache[1])
        free=set(self.visited.get(floor,set()))
        occupied=set(self.collisions.get(floor,set()))
        layer=self._voxel_layers.get(floor)
        if layer is None or layer['source']!=id(self.voxels) or layer['count']!=len(self.voxels) or layer['ground']!=ground or layer['scale']!=(self.res,self.height,self.floor_min_height,self.observed_min_height):
            layer=dict(source=id(self.voxels),count=len(self.voxels),ground=ground,
                scale=(self.res,self.height,self.floor_min_height,self.observed_min_height),free=set(),occupied=set(),observed=set())
            self.extend_voxel_layer(layer,self.voxels);self._voxel_layers[floor]=layer
        free.update(layer['free'])
        if floor not in self.explored:occupied.update(layer['occupied'])
        projected=layer['observed']
        free.update(c for c,s in self.floor_evidence.get(floor,{}).items() if s>=.5)
        occupied.update(c for c,s in self.obstacle_evidence.get(floor,{}).items() if s>=.5)
        explored=set(self.explored.get(floor,{}))|projected|free|occupied
        allcells=explored|{self.cell(self.position)}
        lo=np.min(list(allcells),axis=0)-3
        hi=np.max(list(allcells),axis=0)+3
        # Bounded active view; sparse world memory itself is never cropped away.
        center=np.array(self.cell(self.position))
        if not full:
            lo=np.maximum(lo,center-self.max_cells//2)
            hi=np.minimum(hi,center+self.max_cells//2)
        shape=(int(hi[1]-lo[1]+1),int(hi[0]-lo[0]+1))
        f,o,e=np.zeros(shape,bool),np.zeros(shape,bool),np.zeros(shape,bool)
        for source,array in ((free,f),(occupied,o),(explored,e)):
            for x,z in source:
                if lo[0]<=x<=hi[0] and lo[1]<=z<=hi[1]:
                    array[z-lo[1],x-lo[0]]=True
        # Close sub-cell sampling holes only; do not ray-clear unseen floor.
        f=np.array(Image.fromarray(f).filter(ImageFilter.MaxFilter(3)),dtype=bool,copy=True)
        f &= ~o
        # Only bridge one-cell sampling holes in the display/room channel. No convex
        # hull, visibility cone fill or invented free floor behind furniture/walls.
        padded=np.pad(e,2,constant_values=False)
        e=np.asarray(Image.fromarray(padded).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)),bool)[2:-2,2:-2].copy() | e
        e |= f | o
        walk=f & ~inflate(o,math.ceil(self.radius/self.res))
        # Unknown clearance is not assumed: require footprint to stay on known floor.
        pad=math.ceil(self.radius/self.res)
        walk &= ~inflate(~f,pad)
        result=dict(origin=lo,free=f,observed=e,occupied=o,walk=walk,floor=floor,version=self.version)
        self._grid_cache=(key,result)
        return dict(result)

    def extend_voxel_layer(self, layer, voxels):
        for x,y,z in voxels:
            dy=(y+.5)*self.res-layer['ground']
            if self.observed_min_height<=dy<=2.3:layer['observed'].add((x,z))
            if self.floor_min_height<=dy<=.12:layer['free'].add((x,z))
            elif .16<=dy<=self.height:layer['occupied'].add((x,z))

    def structural_boundaries(self, grid):
        """Height-supported Manhattan wall proposals; never modifies occupancy/walk."""
        shape=grid['free'].shape
        walls=np.zeros(shape,bool); gates=np.zeros(shape,bool)
        columns={}; ground=self.floors[self.active]
        for x,y,z in self.voxels:
            dy=(y+.5)*self.res-ground
            if .2<=dy<=max(2.,self.height): columns.setdefault((x,z),set()).add(y)
        points=np.array([c for c,ys in columns.items()
                         if len(ys)>=3 and (max(ys)-min(ys))*self.res>=self.room_wall_height],float)
        if len(points)<4: return walls,gates,dict(method='provisional_morphology',doors=[])
        # A shared dominant orientation with perpendicular axes; sample every 5 degrees.
        best=None
        for degrees in range(0,90,5):
            a=np.deg2rad(degrees);r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            q=np.rint(points@r).astype(int)
            score=sum(float(np.sum(np.unique(q[:,axis],return_counts=True)[1]**2)) for axis in (0,1))
            if best is None or score>best[0]: best=(score,degrees,r,q)
        _,degrees,r,q=best
        min_length=max(2,math.ceil(self.room_wall_length/self.res))
        max_gap=max(0,math.floor(self.room_wall_gap/self.res+1e-6))
        segments=[]; proposals=[]
        def raster(axis,fixed,start,end):
            line=[]
            for t in np.linspace(start,end,max(2,int(abs(end-start)*3)+1)):
                xy=np.zeros(2);xy[axis]=fixed;xy[1-axis]=t
                cell=np.rint(xy@r.T).astype(int)-grid['origin']
                x,z=map(int,cell)
                if 0<=z<shape[0] and 0<=x<shape[1]: line.append((x,z))
            return line
        for axis in (0,1):
            for fixed in np.unique(q[:,axis]):
                positions=sorted(set(q[np.abs(q[:,axis]-fixed)<=1,1-axis].tolist()))
                runs=[]
                for p in positions:
                    if not runs or p-runs[-1][-1]-1>max_gap: runs.append([p])
                    else: runs[-1].append(p)
                runs=[run for run in runs if run[-1]-run[0]+1>=min_length and
                      len(run)/(run[-1]-run[0]+1)>=.65]
                for run in runs:
                    line=raster(axis,fixed,run[0],run[-1]);segments.extend(line)
                for left,right in zip(runs,runs[1:]):
                    gap=(right[0]-left[-1]-1)*self.res
                    if self.room_door_min<=gap<=self.room_door_max:
                        proposals.append((axis,int(fixed),left[-1]+1,right[0]-1))
        for x,z in segments: walls[z,x]=True
        if not walls.any(): return walls,gates,dict(method='provisional_morphology',doors=[])
        walls=inflate(walls,1)
        doors=[]
        known=grid.get('observed',grid['free']|grid['occupied'])
        for axis,fixed,start,end in proposals:
            center=np.zeros(2);center[axis]=fixed;center[1-axis]=(start+end)/2
            support=[]
            for side in (-1,1):
                count=0
                for d in range(2,max(3,int(1.5/self.res))):
                    for lateral in range(-int(.8/self.res),int(.8/self.res)+1):
                        local=center.copy();local[axis]+=side*d;local[1-axis]+=lateral
                        x,z=np.rint(local@r.T).astype(int)-grid['origin']
                        if 0<=z<shape[0] and 0<=x<shape[1] and grid['free'][z,x]:count+=1
                support.append(count*self.res**2)
            line=raster(axis,fixed,start,end)
            if min(support)<self.room_side_area or not line: continue
            # A doorway must have observed free opening, not just an unobserved gap.
            if sum(bool(grid['free'][z,x]) for x,z in line)/len(line)<.7:continue
            for x,z in line:gates[z,x]=True
            doors.append(dict(center_cell=np.rint(center@r.T).astype(int).tolist(),
                              width_m=round((end-start+1)*self.res,2)))
        gates=inflate(gates,1)&known
        return walls,gates,dict(method='structural_rules',dominant_angle_deg=degrees,doors=doors)

    def split_corridors(self,grid,labels,walls):
        """Separate sustained narrow passages from broad rooms; labels only, never occupancy."""
        result=labels.copy();next_id=int(labels.max())+1;found=[]
        radius=max(1,int(math.ceil(self.room_corridor_width/(2*self.res))))
        h,w=labels.shape
        for label in np.unique(labels):
            if not label:continue
            mask=labels==label
            zs,xs=np.nonzero(mask)
            if len(xs)<(2*radius+1)**2 or min(np.ptp(xs),np.ptp(zs))<2*radius:continue
            # A narrow scan wedge with no observed wide room must remain provisional.
            core=mask & ~inflate(~np.pad(mask,1,constant_values=False),radius)[1:-1,1:-1]
            seeds=np.zeros_like(mask)
            for group in components(core):
                if len(group)*self.res**2>=.5:
                    for x,z in group:seeds[z,x]=True
            if not seeds.any():continue
            broad=inflate(seeds,radius)&mask
            for group in components(mask&~broad):
                if len(group)*self.res**2<.8:continue
                pts=np.asarray(group,float);center=pts.mean(axis=0)
                _,vectors=np.linalg.eigh(np.cov(pts.T));axis=vectors[:,-1]
                side=np.array([-axis[1],axis[0]])
                along=(pts-center)@axis;across=(pts-center)@side
                length=(np.ptp(along)+1)*self.res;width=(np.ptp(across)+1)*self.res
                if length<self.room_corridor_length or width>self.room_corridor_width or length/max(width,self.res)<2.5:continue
                # Require actual opposing wall evidence, not unknown on either side.
                supported=0;total=0
                for t in np.linspace(np.quantile(along,.1),np.quantile(along,.9),12):
                    near=pts[np.abs(along-t)<=max(1.,.2/self.res)]
                    if not len(near):continue
                    mid=np.median(near,axis=0);hits=[];total+=1
                    for sign in (-1,1):
                        hit=False
                        for d in np.arange(.5,self.room_corridor_width/self.res+3,.5):
                            x,z=np.rint(mid+sign*d*side).astype(int)
                            if not(0<=x<w and 0<=z<h):break
                            if walls[z,x]:hit=True;break
                            if not grid.get('observed',grid['free']|grid['occupied'])[z,x]:break
                        hits.append(hit)
                    supported+=int(all(hits))
                if total<6 or supported/total<.6:continue
                found.append(dict(label=next_id,length_m=round(length,2),width_m=round(width,2),
                    paired_wall_support=round(supported/total,2)))
                for x,z in group:result[z,x]=next_id
                next_id+=1
            # Other narrow appendages keep their original room label.
        boundary=np.zeros_like(labels,bool)
        for dz,dx in ((0,1),(1,0)):
            a=result[:h-dz,:w-dx];b=result[dz:,dx:]
            olda=labels[:h-dz,:w-dx];oldb=labels[dz:,dx:]
            cut=(a>0)&(b>0)&(a!=b)&(olda==oldb)
            boundary[:h-dz,:w-dx]|=cut
            boundary[dz:,dx:]|=cut
        return result,boundary,found

    def refine_region_boundaries(self, proposals, grid):
        # Temporal consistency is ID/name matching below, not permanent old cuts.
        # Old fragment cores must never re-split a newly connected room.
        return proposals,{int(k):int(k) for k in np.unique(proposals) if k}

    def partition_rooms(self, domain, walls, gates):
        """Morphological markers + wavefront + evidence-gated adjacency merging.

        Furniture is inside domain. Only structural walls/doors separate it.
        A narrow frontier caused by partial visibility is not a doorway.
        """
        h,w=domain.shape
        radius=max(1,int(round(.65/self.res)))
        core=domain & ~inflate(~np.pad(domain,1,constant_values=False),radius)[1:-1,1:-1]
        labels=np.zeros(domain.shape,np.int32);queue=deque();next_label=1
        for group in components(core):
            if len(group)*self.res**2<1.5:continue
            for x,z in group:labels[z,x]=next_label;queue.append((x,z))
            next_label+=1
        while queue:
            x,z=queue.popleft()
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                a,b=x+dx,z+dz
                if 0<=a<w and 0<=b<h and domain[b,a] and not labels[b,a]:
                    labels[b,a]=labels[z,x];queue.append((a,b))
        for group in components(domain&(labels==0)):
            for x,z in group:labels[z,x]=next_label
            next_label+=1
        # Collect interfaces. Merge every artificial basin unless there is a
        # narrow opening with wall support at BOTH ends and useful space on both sides.
        borders={}
        for dz,dx in ((0,1),(1,0)):
            a=labels[:h-dz,:w-dx];b=labels[dz:,dx:]
            zs,xs=np.nonzero((a>0)&(b>0)&(a!=b))
            for x,z in zip(xs,zs):
                pair=tuple(sorted((int(a[z,x]),int(b[z,x]))))
                borders.setdefault(pair,[]).append((x+dx/2,z+dz/2))
        area=np.bincount(labels.ravel())*self.res**2
        parent=list(range(next_label))
        def root(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        protected=[]
        for (a,b),points in borders.items():
            pts=np.asarray(points);is_door=False
            if len(pts)>=3 and min(area[a],area[b])>=self.room_side_area:
                center=pts.mean(axis=0)
                _,axes=np.linalg.eigh(np.cov(pts.T));axis=axes[:,-1]
                along=(pts-center)@axis
                width=(np.ptp(along)+1)*self.res
                hits=[]
                for endpoint,sign in ((center+along.min()*axis,-1),(center+along.max()*axis,1)):
                    hit=False
                    for distance in np.arange(0.,.35/self.res+1,.5):
                        x,z=np.rint(endpoint+sign*distance*axis).astype(int)
                        if 0<=x<w and 0<=z<h and walls[z,x]:hit=True;break
                    hits.append(hit)
                is_door=self.room_door_min<=width<=self.room_door_max and all(hits)
            if is_door:protected.append((a,b))
            else:parent[root(b)]=root(a)
        lut=np.array([root(k) for k in range(next_label)],np.int32)
        result=lut[labels]
        cuts=np.zeros_like(domain)
        for a,b in protected:
            if root(a)==root(b):continue  # A wide alternate connection wins.
            for x,z in borders[(a,b)]:cuts[int(z),int(x)]=True
        return result,cuts

    def candidates(self, grid):
        free=grid['free'];known=grid.get('observed',free|grid['occupied'])
        # Geometry-only cache: moving/renaming alone must not repeat FFT/arrangement.
        key=('rose2',self.active,self.floors[self.active],tuple(grid['origin']),free.shape,
            free.tobytes(),grid['occupied'].tobytes(),known.tobytes(),
            self.rose2.signature())
        if self._partition_cache is not None and self._partition_cache[0]==key:
            grid.update(self._partition_cache[1]);self.profile['partition_hits']+=1
            return grid['regions']
        labels,walls,gates,info=self.rose2.partition(grid)
        corridors=info.get('corridor_proposals',[])
        sources={int(k):int(k) for k in np.unique(labels) if k}
        new=[]
        for k in np.unique(labels):
            if not k:continue
            zs,xs=np.nonzero(labels==k)
            cells=set(zip((xs+grid['origin'][0]).tolist(),(zs+grid['origin'][1]).tolist()))
            # A cabinet/isolated wall-top footprint is not itself a room. Keep it
            # as observed/obstacle evidence unless some floor supports this region.
            if len(cells)*self.res**2>=.2 and int(free[zs,xs].sum())*self.res**2>=.2:new.append(cells)
        result={};used=set()
        old={rid:row for rid,row in self.regions.items() if row['floor']==self.active and not row.get('retired',False)}
        if old:
            separated=[]
            ox,oz=grid['origin']
            for cells in new:
                overlaps=[cells & row['cells'] for row in old.values()]
                pieces=[piece for piece in overlaps if piece]
                if len(pieces)<2:
                    separated.append(cells)
                    continue
                unassigned=np.zeros_like(free,dtype=bool)
                for x,z in cells-set().union(*pieces):unassigned[z-oz,x-ox]=True
                for local in components(unassigned):
                    fragment={(x+ox,z+oz) for x,z in local}
                    neighbors={(x+dx,z+dz) for x,z in fragment for dx,dz in ((1,0),(-1,0),(0,1),(0,-1))}
                    parents=[piece for piece in pieces if not piece.isdisjoint(neighbors)]
                    if len(parents)==1:parents[0].update(fragment)
                separated.extend(pieces)
            covered=set().union(*separated)
            for row in old.values():
                if row['cells'] & covered:continue
                remaining={(x,z) for x,z in row['cells'] if 0<=x-ox<free.shape[1]
                    and 0<=z-oz<free.shape[0] and free[z-oz,x-ox]}
                if remaining:separated.append(remaining)
            new=separated
        for cells in sorted(new,key=len,reverse=True):
            # Keep the previous ID on the child with the largest overlap.
            scores=[(len(cells&r['cells']),rid)
                    for rid,r in old.items() if rid not in used]
            score,rid=max(scores,default=(0,None))
            if rid is None or score/max(1,min(len(cells),len(old[rid]['cells'])))<.6:
                rid='R%d'%self.next_region;self.next_region+=1
            used.add(rid)
            touches_unknown=False
            for x,z in cells:
                a,b=x-int(grid['origin'][0]),z-int(grid['origin'][1])
                if any(not(0<=a+dx<free.shape[1] and 0<=b+dz<free.shape[0]) or
                       not known[b+dz,a+dx] for dx,dz in ((1,0),(-1,0),(0,1),(0,-1))):
                    touches_unknown=True;break
            cx,cz=min(cells);label=sources.get(int(labels[cz-grid['origin'][1],cx-grid['origin'][0]]))
            for proposal in corridors:
                if proposal['label']==label:proposal['region_id']=rid
            result[rid]=dict(floor=self.active,cells=cells,
                shape_hint='narrow_passage' if any(c['label']==label for c in corridors)
                    or old.get(rid,{}).get('shape_hint')=='narrow_passage' else 'unspecified',
                provisional=info.get('status')!='ok' or touches_unknown,retired=False)
        info['observed_cells']=int(known.sum())
        info['assigned_observed_cells']=int(((labels>0)&known).sum())
        for rid,row in old.items():
            if rid not in result:
                lo=grid['origin'];h,w=free.shape
                if any(lo[0]<=x<lo[0]+w and lo[1]<=z<lo[1]+h for x,z in row['cells']):row['retired']=True
        self.regions.update(result)
        # Reassociate each label by its original spatial evidence, never clone onto children.
        for row in self.semantics.values():
            if row['floor']!=self.active:continue
            anchors=row.get('evidence_cells',set())
            scores=sorted([(len(anchors&r['cells'])/max(1,len(anchors)),rid) for rid,r in result.items()],reverse=True)
            if scores and scores[0][0]>=.6 and (len(scores)==1 or scores[1][0]<.25):
                old_count=self.naming_counts.get(row['region_id'],0)
                row.update(region_id=scores[0][1],cells=set(result[scores[0][1]]['cells']),association='matched')
                if old_count:self.naming_counts[scores[0][1]]=1
            elif anchors and any(grid['origin'][0]<=x<grid['origin'][0]+free.shape[1] and
                                 grid['origin'][1]<=z<grid['origin'][1]+free.shape[0] for x,z in anchors):
                row.update(cells=set(),association='uncertain')
        self.enforce_region_label_uniqueness()
        # Uncertain/lost semantic associations may be visually grounded again;
        # geometry does not decide which functional name the new child deserves.
        matched={row['region_id'] for row in self.semantics.values() if row['floor']==self.active
            and row.get('cells') and row.get('association')=='matched'}
        for rid in result:
            if rid not in matched:self.naming_counts[rid]=0
        grid.update(regions=result,room_partition=info,structural_walls=walls,door_cuts=gates)
        grid['_partition_key']=key
        self._partition_cache=(key,{k:grid[k] for k in ('regions','room_partition','structural_walls','door_cuts','_partition_key')})
        self.profile['partition_builds']+=1
        return result

    def region_views(self,grid,step,views=None):
        key=(self.version,self.active,step,None if views is None else tuple(sorted(views)),
            id(grid.get('regions')),tuple(grid['origin']),grid['free'].shape)
        if self._views_cache is not None and self._views_cache[0]==key:return self._views_cache[1]
        rows={}
        for name,obs in self.observed_views.items():
            if obs['step']!=step or obs['floor']!=self.active or views is not None and name not in views:continue
            for rid,region in grid['regions'].items():
                samples=[p for p in obs['samples'] if (p[0],p[1]) in region['cells']]
                cells={(p[0],p[1]) for p in samples}
                if len(cells)<3:continue
                uv=np.array([p[2:] for p in samples])
                rows.setdefault(rid,{})[name]=dict(cells=cells,uv=uv.tolist(),
                    image_box=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())])
        self._views_cache=(key,rows)
        return rows

    def naming_overlay(self,image,grid,step,view):
        image=image.copy();draw=ImageDraw.Draw(image);w,h=image.size
        for rid,views in self.region_views(grid,step,[view]).items():
            uv=np.array(views[view]['uv']);x,y=np.median(uv,axis=0)*[w-1,h-1]
            for u,v in uv[::max(1,len(uv)//60)]:
                a,b=u*(w-1),v*(h-1);draw.ellipse((a-1,b-1,a+1,b+1),fill='yellow')
            draw.text((x,y),rid,fill='yellow',stroke_width=2,stroke_fill='black')
        return image

    @staticmethod
    def region_label_key(label):
        return ' '.join(label.split()).casefold()

    def enforce_region_label_uniqueness(self):
        # Each region has one label slot. A name may belong to several regions.
        groups={}
        for key,row in self.semantics.items():
            if not row.get('cells') or row.get('association')!='matched':continue
            groups.setdefault((row['floor'],row['region_id']),[]).append(key)
        conflicts={key for keys in groups.values() if len(keys)>1 for key in keys}
        for key in conflicts:
            self.semantics[key].update(cells=set(),association='uncertain',
                association_reason='Multiple names refer to the same geometric region')
            self.naming_counts[self.semantics[key]['region_id']]=0

    def validate_room_labels(self, assignments, grid, allowed, step, views=None):
        visible=self.region_views(grid,step,views)
        if not isinstance(assignments,list):raise ValueError('room_labels must be a list')
        region_ids=set();by_name={}
        for item in assignments:
            if not isinstance(item,dict) or set(item)-{'target_id','region_id','view','confidence','evidence'}:
                raise ValueError('naming only: boundaries, polygons and region_ids are forbidden')
            if item.get('target_id') not in allowed or item.get('confidence') not in ('confirmed','tentative'):
                raise ValueError('unknown target or confidence')
            rid=item.get('region_id');view=item.get('view')
            tid=item['target_id'];name=self.region_label_key(allowed[tid])
            if rid in region_ids:raise ValueError('Each region may receive exactly one label; duplicate region: '+str(rid))
            if self.naming_counts.get(rid,0):raise ValueError('Region already named in this correction cycle: '+str(rid))
            if rid not in visible or view not in visible[rid]:raise ValueError('region lacks current depth-supported visibility in this view')
            if not isinstance(item.get('evidence'),str) or not item['evidence'].strip():raise ValueError('missing visual evidence')
            if item['evidence'].strip().lower().rstrip('.') in (
                    'observed semantic cues in the labeled patch','...', 'visual evidence', 'semantic cues'):
                raise ValueError('Naming evidence must describe actual observed cues, not example placeholder text; abstain if unsupported')
            region_ids.add(rid)
            by_name.setdefault(name,set()).add((self.active,rid))
        for row in self.semantics.values():
            name=self.region_label_key(row['label'])
            if name in by_name and row.get('naming_epoch',0)==self.naming_epoch and row.get('association')=='matched':
                by_name[name].add((row['floor'],row['region_id']))
        for name, owners in by_name.items():
            if len(owners)<2:continue
            if any(floor!=self.active or rid not in grid['regions'] or
                   grid['regions'][rid].get('shape_hint')=='narrow_passage' for floor,rid in owners):
                raise ValueError('Shared labels require ordinary regions on the same floor: '+name)
            remaining={rid for _,rid in owners}
            frontier=[remaining.pop()]
            while frontier:
                cells=grid['regions'][frontier.pop()]['cells']
                neighbors={(x+dx,z+dz) for x,z in cells for dx,dz in ((1,0),(-1,0),(0,1),(0,-1))}
                adjacent={rid for rid in remaining if not neighbors.isdisjoint(grid['regions'][rid]['cells'])}
                remaining-=adjacent
                frontier.extend(adjacent)
            if remaining:raise ValueError('Shared label regions must form an adjacent group: '+name)
        return visible

    def label_regions(self, assignments, grid, allowed, step=None, views=None):
        visible=self.validate_room_labels(assignments,grid,allowed,step,views)
        changed=False
        renamed={item['region_id'] for item in assignments}
        targets={item['target_id'] for item in assignments}
        names={self.region_label_key(allowed[tid]) for tid in targets}
        # A reset permits replacing earlier names. Keep old evidence until a new
        # supported assignment succeeds; then replace the whole region atomically.
        self.semantics={key:row for key,row in self.semantics.items()
            if not((row['floor']==self.active and row['region_id'] in renamed) or
                (row.get('naming_epoch',0)!=self.naming_epoch and
                 (row['target_id'] in targets or self.region_label_key(row['label']) in names)))}
        for item in assignments:
            rid=item['region_id'];key='%s:%s'%(item['target_id'],rid)
            old=self.semantics.get(key)
            new=dict(target_id=item['target_id'],label=allowed[item['target_id']],region_id=rid,
                floor=self.active,cells=set(grid['regions'][rid]['cells']),confidence=item['confidence'],
                evidence=item['evidence'],evidence_cells=set(visible[rid][item['view']]['cells']),association='matched',
                naming_epoch=self.naming_epoch)
            if old!=new:self.semantics[key]=new;changed=True
        if renamed:
            for rid in renamed:self.naming_counts[rid]=1
            self.version+=1
        return changed

    def reset_region_naming(self):
        self.naming_counts.clear()
        self.naming_epoch+=1
        self.version+=1

    def reachable_points(self, grid, max_distance=4., limit=24):
        start=tuple((np.array(self.cell(self.position))-grid['origin']).tolist())
        walk=grid['walk']
        h,w=walk.shape
        if not (0<=start[0]<w and 0<=start[1]<h and walk[start[1],start[0]]): return []
        todo=deque([start]); seen={start}; result=[]
        stride=max(1,int(.6/self.res))
        while todo:
            p=todo.popleft()
            distance=math.dist(start,p)*self.res
            if .5<=distance<=max_distance and p[0]%stride==0 and p[1]%stride==0:
                world=self.world(np.array(p)+grid['origin'])
                result.append(dict(cell=list(p),world=world.tolist(),distance=distance))
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                q=p[0]+dx,p[1]+dz
                if q not in seen and 0<=q[0]<w and 0<=q[1]<h and walk[q[1],q[0]] and math.dist(start,q)*self.res<=max_distance:
                    seen.add(q); todo.append(q)
        # Angular diversity, not only nearest points in one direction.
        result.sort(key=lambda r:math.atan2(r['cell'][1]-start[1],r['cell'][0]-start[0]))
        if len(result)>limit:
            result=[result[i] for i in np.linspace(0,len(result)-1,limit,dtype=int)]
        for i,r in enumerate(result): r['id']='P%d'%(i+1)
        return result

    def plan(self, target):
        if self.transition or abs(target[1]-self.floors[self.active])>.25: return []
        g=self.grid()
        a=tuple((np.array(self.cell(self.position))-g['origin']).tolist())
        b=tuple((np.array(self.cell(target))-g['origin']).tolist())
        return [self.world(np.array(p)+g['origin']) for p in astar(g['walk'],a,b)]

    def metadata(self, grid):
        key=(self.version,self.active,self.transition,tuple(self.position),tuple(self.forward),
            id(grid.get('regions')),tuple(grid['origin']),grid['free'].shape)
        if self._metadata_cache is not None and self._metadata_cache[0]==key:
            return self._metadata_cache[1]
        agent_cell=self.cell(self.position)
        relations=[]
        for row in self.semantics.values():
            if row['floor']!=self.active:continue
            cells=row['cells'];boundary=[c for c in cells if any(
                (c[0]+dx,c[1]+dz) not in cells for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)))]
            nearest=min(boundary,key=lambda c:math.dist(c,agent_cell)) if boundary else None
            relations.append(dict(target_id=row['target_id'],label=row['label'],region_id=row['region_id'],
                confidence=row['confidence'],association=row.get('association','uncertain'),
                agent_inside=agent_cell in cells,nearest_boundary_cell=nearest,
                distance_to_observed_region_m=round(min((math.dist(c,agent_cell) for c in cells),default=float('inf'))*self.res,3) if cells else None,
                distance_to_observed_boundary_m=round(math.dist(nearest,agent_cell)*self.res,3) if nearest else None,
                boundary_cells_sample=sorted(boundary)[::max(1,math.ceil(len(boundary)/64))],
                caution='Observed geometric boundary may be a visibility frontier, not a doorway; broad region membership alone does not establish an entrance/endpoint.'))
        result=dict(version=self.version,floor=self.active,floor_heights=self.floors,
            on_stairs_or_unconfirmed_level=self.transition,position=self.position.tolist(),
            agent_cell=list(agent_cell),forward=self.forward.tolist(),
            agent_region_ids=[rid for rid,row in grid.get('regions',{}).items() if agent_cell in row['cells']],
            retained_route=self.route,
            target_region_relations=relations,
            coordinates='world meters; X right, Z down; image origin is minimum X,Z',
            origin_cell=grid['origin'].tolist(),meters_per_cell=self.res,
            objects=[dict(id=k,**{a:v for a,v in r.items() if a not in ('cells','cell_scores')},
                distance_xz_m=round(float(np.linalg.norm((np.array(r['center'])-self.position)[[0,2]])),3)) for k,r in self.objects.items()],
            room_partition=grid.get('room_partition',{}),
            naming_epoch=self.naming_epoch,
            regions=[dict(id=k,naming_count=self.naming_counts.get(k,0),provisional=r.get('provisional',True),shape_hint=r.get('shape_hint','unspecified'),area_m2=round(len(r['cells'])*self.res**2,2)) for k,r in grid.get('regions',{}).items()],
            semantic_regions=[{a:v for a,v in r.items() if a not in ('cells','evidence_cells')} for r in self.semantics.values()],
            trajectory_legend=dict(blue='retained effective route; not proof that every step is correct',
                red='withdrawn/deviating route or actual rollback; historical evidence, NOT completed required progress or obstacles'),
            red_routes=self.error_routes)
        self._metadata_cache=(key,result)
        return result

    def render(self, grid=None, points=(), target=None, show_agent=True):
        g=self.grid() if grid is None else grid
        key=(g.get('_partition_key'),self.version,self.active,tuple(self.position),tuple(self.forward),show_agent)
        cacheable=g.get('_partition_key') is not None and not points and target is None
        if cacheable and self._render_cache is not None and self._render_cache[0]==key:
            self.profile['render_hits']+=1
            return self._render_cache[1].copy()
        pixels=np.full((*g['free'].shape,3),55,np.uint8)
        visible=g.get('observed',g['free']) & ~g['occupied']
        pixels[visible]=(255,255,255); pixels[g['occupied']]=(22,22,26)
        def local(c): return int(c[0]-g['origin'][0]),int(c[1]-g['origin'][1])
        palette=((80,170,245),(245,180,70),(160,100,230),(70,200,130),(230,100,150))
        def region_color(region_id):
            import colorsys
            number=int(region_id[1:])-1
            if number<len(palette):return np.array(palette[number])
            # Never wrap R6 back onto R1. Colors depend only on the region ID,
            # so camera motion and dictionary order cannot change the palette.
            hue=(.35+(number-len(palette))*.618033988749895)%1.
            return np.rint(np.array(colorsys.hsv_to_rgb(hue,.72,.94))*255).astype(np.uint8)
        for rid,r in g.get('regions',{}).items():
            color=region_color(rid)
            for c in r['cells']:
                x,z=local(c)
                if 0<=z<len(pixels) and 0<=x<pixels.shape[1] and visible[z,x]:
                    pixels[z,x]=(.8*pixels[z,x]+.2*color).astype(np.uint8)
        # Orange virtual door cuts are a rendering layer, never navigation obstacles.
        if 'door_cuts' in g:pixels[g['door_cuts'] & visible]=(245,160,50)
        for i,r in enumerate(self.semantics.values()):
            if r['floor']!=self.active: continue
            for c in r['cells']:
                x,z=local(c)
                if 0<=z<len(pixels) and 0<=x<pixels.shape[1] and visible[z,x]:
                    pixels[z,x]=(.65*pixels[z,x]+.35*region_color(r['region_id'])).astype(np.uint8)
        scale=max(1,min(5,1500//max(pixels.shape[:2])))
        im=Image.fromarray(pixels).resize((pixels.shape[1]*scale,pixels.shape[0]*scale),Image.Resampling.NEAREST)
        draw=ImageDraw.Draw(im)
        def xy(c):
            x,z=local(c); return (x+.5)*scale,(z+.5)*scale
        for a,b in zip(self.route,self.route[1:]):
            if a['floor']==b['floor']==self.active:
                draw.line([xy(self.cell(a['position'])),xy(self.cell(b['position']))],fill=(30,110,235),width=2)
        # Draw red after blue so withdrawn history stays visible even before rollback succeeds.
        for row in self.error_routes:
            for a,b in zip(row['points'],row['points'][1:]):
                if a['floor']==b['floor']==self.active:
                    draw.line([xy(self.cell(a['position'])),xy(self.cell(b['position']))],fill=(225,45,45),width=2)
        named={r['region_id'] for r in self.semantics.values() if r['floor']==self.active and
               r['cells'] and r.get('association')=='matched'}
        for rid,r in g.get('regions',{}).items():
            if rid in named:continue
            center=np.mean(list(r['cells']),axis=0)
            c=min(r['cells'],key=lambda c:math.dist(c,center))
            draw.text(xy(c),rid+(' ?' if r.get('provisional') else ''),fill='black',stroke_width=1,stroke_fill='white')
        name_offsets={}
        for r in self.semantics.values():
            if r['floor']!=self.active or not r['cells']: continue
            center=np.mean(list(r['cells']),axis=0)
            c=min(r['cells'],key=lambda c:math.dist(c,center))
            label=r['label'] if r['label'].isascii() else r['target_id']
            label_position=xy(c)
            offset=name_offsets.get(r['region_id'],0);name_offsets[r['region_id']]=offset+13
            draw.text((label_position[0],label_position[1]+offset),label+(' ?' if r['confidence']=='tentative' else ''),
                      fill=(90,35,110),stroke_width=1,stroke_fill='white')
        object_legend=[]
        for oid,r in self.objects.items():
            if r['floor']==self.active:
                p=xy(self.cell(r['center']))
                if not(0<=p[0]<im.width and 0<=p[1]<im.height):continue
                color=(0,125,65)
                draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=color,outline='white',width=1)
                short=oid+'/'+r['target_id']+': '+r.get('query',r['label']).encode('ascii','replace').decode()
                bbox=draw.textbbox((0,0),short)
                tx=max(2,min(p[0]+7,im.width-(bbox[2]-bbox[0])-2))
                ty=max(2,min(p[1]-14,im.height-14))
                draw.line([p,(tx,ty+7)],fill=color,width=1)
                draw.text((tx,ty),short,fill=color,stroke_width=1,stroke_fill='white')
                name=r['label'].encode('ascii','replace').decode()
                object_legend.extend(textwrap.wrap('%s/%s: %s | X=%.2f Z=%.2f m (candidate)'%
                    (oid,r['target_id'],name,r['center'][0],r['center'][2]),width=88))
        for p in points:
            x,z=xy(self.cell(p['world']))
            draw.ellipse((x-3,z-3,x+3,z+3),fill=(30,150,50))
            draw.text((x+4,z),p['id'],fill='green',stroke_width=1,stroke_fill='white')
        if target is not None:
            x,z=xy(self.cell(target)); draw.ellipse((x-5,z-5,x+5,z+5),outline='red',width=2)
        if show_agent:
            p=xy(self.cell(self.position))
            q=(p[0]+self.forward[0]*15,p[1]+self.forward[2]*15)
            draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill='blue'); draw.line([p,q],fill='blue',width=3)
        canvas=Image.new('RGB',(max(620,im.width),im.height+70+14*len(object_legend)),'white'); canvas.paste(im,(0,70))
        legend_draw=ImageDraw.Draw(canvas)
        legend_draw.text((6,4),'F%d | +X right, +Z down | blue: retained route\nred: withdrawn / rollback; green dot: observed object target candidate\nwhite: observed (not necessarily walkable); color: region; black: obstacle\ndark gray: unknown; orange: logical partition; R?: unnamed provisional'%self.active,fill='black')
        for i,line in enumerate(object_legend):legend_draw.text((6,im.height+70+i*14),line,fill=(0,100,50))
        self.profile['render_builds']+=1
        if cacheable:self._render_cache=(key,canvas.copy())
        return canvas

    def save(self, folder, ordered=False, force=False):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        snapshot=(str(folder.resolve()),self.version,self.active,self.transition)
        if not force and self._saved_snapshot==snapshot and (folder/'map_latest.json').exists() and (folder/'map_latest.png').exists():return
        grid=self.grid(); self.candidates(grid)
        sequence=sorted if ordered else list
        def serial(rows):
            return {k:{a:(sequence(v) if isinstance(v,set) else v) for a,v in r.items()} for k,r in rows.items()}
        value=dict(version=self.version,resolution=self.res,floors=self.floors,active=self.active,
            confirmed_doors=self.rose2.confirmed_doors,
            region_naming_counts=self.naming_counts,naming_epoch=self.naming_epoch,
            room_rules={k:getattr(self,k) for k in ('room_wall_height','room_wall_length','room_wall_gap',
                'room_door_min','room_door_max','room_side_area','room_corridor_width','room_corridor_length')},room_partition=grid.get('room_partition',{}),
            observation_stats=self.observation_stats,
            height_channels=dict(floor=[self.floor_min_height,.12],
                observed=[self.observed_min_height,2.3],obstacle=[.16,self.height]),
            projection_channels={name:{str(floor):[[int(c[0]),int(c[1]),float(score)] for c,score in values.items()]
                for floor,values in channel.items()} for name,channel in
                (('explored',self.explored),('floor',self.floor_evidence),('obstacle',self.obstacle_evidence))},
            transition=self.transition,position=self.position.tolist(),voxels=sequence(self.voxels),route=self.route,error_routes=self.error_routes,
            objects=serial(self.objects),regions=serial(self.regions),semantic_regions=serial(self.semantics),
            visited={k:sequence(v) for k,v in self.visited.items()},collisions={k:sequence(v) for k,v in self.collisions.items()})
        tmp=folder/'map_latest.json.tmp'; tmp.write_text(json.dumps(value,separators=(',',':')),encoding='utf-8')
        os.replace(tmp,folder/'map_latest.json')
        tmp=folder/'map_latest.tmp.png'; self.render(grid).save(tmp,compress_level=1); os.replace(tmp,folder/'map_latest.png')
        self._saved_snapshot=snapshot

    def save_final(self, folder, termination=''):
        """One final atlas + full state per episode, with separate full-extent floor PNGs."""
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        actual_floor=self.active
        panels=[]; floor_files=[]
        try:
            for floor in sorted(self.floors):
                self.active=floor
                grid=self.grid(full=True); self.candidates(grid)
                panel=self.render(grid,show_agent=floor==actual_floor and not self.transition)
                if len(self.floors)>1:
                    name='final_map_F%d.png'%floor
                    tmp=folder/(name+'.tmp')
                    panel.save(tmp,format='PNG'); os.replace(tmp,folder/name)
                    floor_files.append(name)
                preview=panel.copy()
                if len(self.floors)>1:preview.thumbnail((900,900),Image.Resampling.LANCZOS)
                panels.append(preview)
        finally:
            self.active=actual_floor
        if not panels: return
        cols=min(2,len(panels));rows=math.ceil(len(panels)/cols)
        width=max(p.width for p in panels);height=max(p.height for p in panels)
        atlas=Image.new('RGB',(cols*width,rows*height+28),'white')
        ImageDraw.Draw(atlas).text((6,6),'FINAL OBSERVED MAP | '+str(termination),fill='black')
        for i,panel in enumerate(panels): atlas.paste(panel,((i%cols)*width,(i//cols)*height+28))
        tmp=folder/'final_map.png.tmp';atlas.save(tmp,format='PNG');os.replace(tmp,folder/'final_map.png')
        self.save(folder,ordered=True,force=True)
        value=json.loads((folder/'map_latest.json').read_text(encoding='utf-8'))
        value.update(termination_reason=termination,final_map_image='final_map.png',floor_images=floor_files)
        tmp=folder/'final_map.json.tmp';tmp.write_text(json.dumps(value),encoding='utf-8')
        os.replace(tmp,folder/'final_map.json')
        # Remove only this writer's transient artifacts, after both final files exist.
        for name in ('map_latest.png','map_latest.json'):
            (folder/name).unlink(missing_ok=True)


# --- pose.py ---
"""Simulator, supplied-pose and RGB-D localization adapters."""
import math
def transform(rotation=None, position=None):
    t=np.eye(4)
    if rotation is not None: t[:3,:3]=rotation
    if position is not None: t[:3,3]=position
    return t


class RGBDOdometry:
    def __init__(self, hfov, min_depth, max_depth, normalized, extrinsic):
        import cv2
        self.cv=cv2; self.hfov=hfov
        self.min_depth,self.max_depth,self.normalized=min_depth,max_depth,normalized
        self.extrinsic=extrinsic
        self.orb=cv2.ORB_create(nfeatures=2500)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous=None
        self.camera_pose=extrinsic.copy()
        self.inliers=0

    def update(self,rgb,depth):
        cv=self.cv
        depth=np.asarray(depth).squeeze().astype(float)
        if self.normalized: depth=depth*(self.max_depth-self.min_depth)+self.min_depth
        h,w=depth.shape
        rh,rw=np.asarray(rgb).shape[:2]
        if abs(rw/rh-w/h)>1e-4: raise ValueError('RGB/depth aspect ratios differ')
        image=cv.resize(rgb,(w,h))
        gray=cv.cvtColor(image,cv.COLOR_RGB2GRAY)
        keypoints,descriptors=self.orb.detectAndCompute(gray,None)
        current=(keypoints,descriptors,depth)
        if self.previous is None:
            if descriptors is None or len(keypoints)<20: return None,'insufficient_initial_features'
            self.previous=current
            return self.camera_pose @ np.linalg.inv(self.extrinsic),'initialized'
        old,desc,old_depth=self.previous
        if descriptors is None: return None,'no_features'
        matches=self.matcher.knnMatch(desc,descriptors,k=2)
        f=w/(2*np.tan(np.deg2rad(self.hfov)/2))
        k=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
        xyz,uv=[],[]
        for pair in matches:
            if len(pair)!=2: continue
            a,b=pair
            if a.distance>=.72*b.distance: continue
            x,y=old[a.queryIdx].pt
            d=old_depth[min(h-1,max(0,round(y))),min(w-1,max(0,round(x)))]
            if not np.isfinite(d) or not max(.02,self.min_depth)<d<self.max_depth-1e-4: continue
            xyz.append([(x-k[0,2])*d/f,(y-k[1,2])*d/f,d])
            uv.append(keypoints[a.trainIdx].pt)
        if len(xyz)<12: return None,'insufficient_depth_matches'
        ok,rvec,tvec,inliers=cv.solvePnPRansac(np.asarray(xyz,np.float32),np.asarray(uv,np.float32),k,None,
            iterationsCount=150,reprojectionError=3.,confidence=.999,flags=cv.SOLVEPNP_EPNP)
        self.inliers=0 if inliers is None else len(inliers)
        if not ok or self.inliers<10 or self.inliers/len(xyz)<.35: return None,'pnp_rejected'
        rotation,_=cv.Rodrigues(rvec)
        delta=transform(rotation,tvec[:,0])  # previous camera -> current camera, OpenCV axes
        angle=np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))
        if np.linalg.norm(tvec)>1.0 or angle>np.deg2rad(55): return None,'implausible_relative_motion'
        axes=np.diag([1.,-1.,-1.,1.])
        self.camera_pose=self.camera_pose @ axes @ np.linalg.inv(delta) @ axes
        self.previous=current
        return self.camera_pose @ np.linalg.inv(self.extrinsic),'tracked'


class PoseProvider:
    def __init__(self,source,depth_config,rgb_config):
        if abs(depth_config.hfov-rgb_config.hfov)>1e-4:
            raise ValueError('RGB and depth must have the same field of view')
        def extrinsic(c):
            angles=np.asarray(getattr(c,'orientation',[0,0,0]),float)
            # Habitat sensor orientation is XYZ Euler angles in radians.
            cx,cy,cz=np.cos(angles);sx,sy,sz=np.sin(angles)
            rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            return transform(rz@ry@rx,np.asarray(c.position,float))
        self.ext=extrinsic(depth_config)
        if not np.allclose(self.ext,extrinsic(rgb_config),atol=1e-4):
            raise ValueError('RGB/depth extrinsics must agree; registration is required otherwise')
        self.source,self.selected,self.cfg=source,None,depth_config
        self.pose=None; self.valid=False; self.reason='not_initialized'; self.vo=None

    def update(self,observations,sim=None):
        supplied=observations.get('agent_pose')
        if self.selected is None:
            self.selected=('provided' if supplied is not None else 'rgbd') if self.source=='auto' else self.source
        if self.selected=='sim':
            import quaternion
            state=sim.get_agent_state()
            pose=transform(quaternion.as_rotation_matrix(state.rotation),np.asarray(state.position))
            reason='sim_pose'
        elif self.selected=='provided':
            pose=None if supplied is None else np.asarray(supplied,float)
            reason='provided_pose' if pose is not None else 'missing_provided_pose'
            if pose is not None and (pose.shape!=(4,4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=.01)
                or np.linalg.det(pose[:3,:3])<.99):
                raise ValueError('agent_pose must be a rigid 4x4 agent-to-world matrix, meters, Y-up / forward -Z')
        else:
            if self.vo is None:
                c=self.cfg
                self.vo=RGBDOdometry(c.hfov,c.min_depth,c.max_depth,c.normalize_depth,self.ext)
            pose,reason=self.vo.update(observations['rgb'],observations['depth'])
        self.valid=pose is not None
        self.reason=reason
        if self.valid: self.pose=pose
        return self.valid

    def camera(self,yaw=0.):
        c,s=np.cos(yaw),np.sin(yaw)
        rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        return self.pose @ transform(rot) @ self.ext


# --- segmentation.py ---
"""Offline Grounding DINO + SAM2 process and worker."""
# Offline segmentation worker

import argparse
import base64
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import zlib

import numpy as np
from PIL import Image


class Segmenter:
    def __init__(self, args):
        env=os.environ.copy()
        for key in ('PYTHONHOME','PYTHONPATH','RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
            env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES=str(args.seg_gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        cmd=[args.seg_python,'-u',str(Path(__file__).resolve()),'--seg-worker',
             '--dino',args.dino_model_path,'--sam',args.sam_checkpoint,'--config',args.sam_config,
             '--threshold',str(args.seg_threshold),'--max-instances',str(args.seg_max_instances)]
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env=env)
        self.responses=queue.Queue()
        self.timeout=args.seg_timeout
        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=reader,daemon=True).start()
        self.read()

    def read(self):
        try: line=self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.process.kill()
            raise TimeoutError('Segmentation worker timeout')
        if line is None: raise RuntimeError('Segmentation worker exited')
        result=json.loads(line)
        if not result['ok']: raise RuntimeError('Segmentation: '+result['error'])
        return result

    def predict(self, rgb, targets):
        objects=[dict(t,query=t['name']) for t in targets if t['kind']=='object']
        if not objects: return []
        buffer=io.BytesIO(); Image.fromarray(rgb).convert('RGB').save(buffer,format='PNG')
        self.process.stdin.write(json.dumps(dict(image=base64.b64encode(buffer.getvalue()).decode(),targets=objects))+'\n')
        self.process.stdin.flush()
        result=self.read()['instances']
        for item in result:
            packed=np.frombuffer(zlib.decompress(base64.b64decode(item.pop('packed'))),dtype=np.uint8)
            h,w=item.pop('shape')
            item['mask']=np.unpackbits(packed,count=h*w).reshape(h,w).astype(bool)
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()


def worker(args):
    protocol=sys.stdout; sys.stdout=sys.stderr
    def emit(value): protocol.write(json.dumps(value)+'\n'); protocol.flush()
    try:
        import inspect
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        processor=AutoProcessor.from_pretrained(args.dino,local_files_only=True)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(args.dino,local_files_only=True).to('cuda:0').eval()
        sam=SAM2ImagePredictor(build_sam2(args.config,args.sam,device='cuda:0'))
        post=processor.post_process_grounded_object_detection
        threshold_key='threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
        emit(dict(ok=True))
    except Exception as exc:
        emit(dict(ok=False,error=str(exc))); return 1
    for line in sys.stdin:
        try:
            request=json.loads(line)
            image=Image.open(io.BytesIO(base64.b64decode(request['image']))).convert('RGB')
            results=[]
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                sam.set_image(np.asarray(image))
            for target in request['targets']:
                # One phrase per query gives an unambiguous instruction target ID.
                text=target['query'].strip().lower().rstrip('.')+'.'
                inputs=processor(images=image,text=text,return_tensors='pt').to('cuda:0')
                with torch.inference_mode(): outputs=model(**inputs)
                detected=post(outputs,inputs.input_ids,target_sizes=[image.size[::-1]],
                    text_threshold=.25,**{threshold_key:args.threshold})[0]
                boxes=detected['boxes'].detach().cpu().numpy()
                scores=detected['scores'].detach().cpu().numpy()
                keep=[]
                for idx in np.argsort(-scores):
                    box=boxes[idx]
                    duplicate=False
                    for old in keep:
                        a=boxes[old]
                        inter=np.prod(np.maximum(0,np.minimum(a[2:],box[2:])-np.maximum(a[:2],box[:2])))
                        union=np.prod(a[2:]-a[:2])+np.prod(box[2:]-box[:2])-inter
                        if inter/max(float(union),1e-6)>.5: duplicate=True; break
                    if not duplicate: keep.append(int(idx))
                    if len(keep)>=args.max_instances: break
                if not keep: continue
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    masks,quality,_=sam.predict(box=boxes[keep],multimask_output=False)
                masks=np.asarray(masks).reshape(len(keep),image.height,image.width)>0
                for j,idx in enumerate(keep):
                    packed=zlib.compress(np.packbits(masks[j]).tobytes())
                    results.append(dict(target_id=target['id'],label=target['name'],query=target['query'],score=float(scores[idx]),
                        box=boxes[idx].tolist(),shape=[image.height,image.width],packed=base64.b64encode(packed).decode()))
            emit(dict(ok=True,instances=results))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            emit(dict(ok=False,error=type(exc).__name__+': '+str(exc)))
    return 0


# --- prompts.py ---
"""Active extraction, map reasoning, naming and compression prompts."""
TARGET_PROMPT='''Read the ENTIRE ORIGINAL instruction and inventory its evidence, not new tasks.
Return JSON:
{"regions":[{"name":"complete spatial noun phrase","head":"semantic head",
"modifiers":["original disambiguating relation"],"query":"short area category"}],
"stuffs":[{"name":"complete physical entity noun phrase","head":"semantic head",
"modifiers":[],"query":"short visually detectable object phrase"}],
"clauses":[{"text":"verbatim consecutive instruction span including punctuation",
"regions":[0],"stuffs":[0]}]}
Inventory ALL spatial and physical noun phrases, including references in later clauses and
within modifiers. For each, resolve its head, modifiers, relative clauses and attachment.
A modifier's noun must not replace its head. Keep full relations in name/modifiers; query
names the visible category rather than an entire spatial clause. Never invent typical furniture.
DISTINGUISH PHYSICAL OBJECTS FROM SPATIAL REGIONS before assigning categories.
stuffs contains localizable physical entities, including fixed architectural structures;
an entity need not be movable or furniture to be an object. An archway denotes the
physical arch/opening structure and is an OBJECT target, not a room/region name.
regions contains spatial areas with floor extent, such as rooms, corridors and explicitly
described entrance areas. Distinguish a physical structure from the space before,
beneath, around or beyond it. Classify the referent of the noun phrase's head, not
the fact that navigation goes to/through it or waits there. A stop destination is
not automatically a region, and a traversable structure is not automatically an area.
Keep object descriptions in stuffs for segmentation and depth-projected map localization;
never name their entire containing room after the object. When an instruction explicitly
names an area in relation to a structure, extract that area as a region and its referenced
physical structure as an object; these are distinct referents, not duplicate categories
for the same entity. Do not invent an area target merely to fit the region pipeline.
Regions must be extracted even when no object is named. Visibility is irrelevant:
this is textual extraction before observation. Referenced unnamed places count.
Resolve aliases by meaning, not merely a shared head; distinguish different entrances/rooms.
Contextual references do not create entry goals. Do not rewrite the original instruction,
insert actions, drop completed-looking clauses, or alter the final stopping relation.
clauses must cover the COMPLETE original text in order, exactly once (whitespace may vary).
Each clause selects relevant entries DIRECTLY from the final inventories using
zero-based INTEGER list indices: regions indexes the regions list, stuffs indexes
the stuffs list. Index 0 selects the first entry of that category, 1 the second.
Use [] when no entry of that category applies. NEVER rewrite names in references,
and do not output quoted indices, aliases, query strings or target IDs there.
Build the final inventories first, then select their indices. Every inventory item
needs a reference. If an audit reorders or recategorizes entries, update indices
to select the intended entries in the FINAL lists. Renaming an entry alone requires
no reference change: the program retrieves the complete name from the selected row.
For pure motion without any named/referenced place/entity, both inventories may be empty.
Do not use a fixed noun vocabulary or scene-specific rules.'''

TARGET_AUDIT_PROMPT=TARGET_PROMPT+'''
Independently audit the draft against the ORIGINAL instruction, clause by clause.
Recover missing spatial/physical nouns, including those inside modifiers; correct categories,
query phrases, aliases, relation attachments and references. Return the WHOLE corrected inventory.
Explicitly recheck physical-entity versus spatial-area classification against the noun
phrase's referent. Architectural structures remain objects even when traversed or used
as stopping landmarks. Correct an archway mistakenly placed in regions into stuffs;
do not simply accept the draft category or assign the same referent to both lists.
After any category correction, update clause indices to the final category lists.
Only for a purely motion-based instruction with no named/referenced landmark or area,
return empty regions/stuffs with complete clauses, plus
"no_named_targets":true and "reason":"explain using the actual instruction".
Do not infer absence of targets from absence of images or from the earlier empty result.'''

REVIEW_PROMPT='''Judge navigation for the ORIGINAL instruction using the CURRENT map and its trace.
The STOP question is PROSPECTIVE: "Given the actual mapped route so far, would
issuing STOP HERE NOW satisfy the instruction's route and final-position/stay requirements?"
It is NOT "Has every requested behavior, including waiting, already happened?"
Your only image is the map. Text includes current measured positions, mapped targets,
the original instruction and historical summaries. No discrete action history or
precomputed entry/turn-event table is provided: interpret the map and trace yourself.
All navigation decisions must be grounded in the map, mapped labels and measured trace.
Timestamped episode/correction initialization descriptions are fallible semantic
context only; they describe those scan positions, not the current position or progress.
No RGB views or naming-observation narratives are provided to this decision task.
Never let an initialization description override current map/trace evidence.

READ CURRENT EVIDENCE FIRST
1. Locate the current robot and read the latest retained position. Compare it with
   the trace start and earlier samples BEFORE reading historical descriptions.
   Remaining in the same room does NOT mean stationary or still at the start.
   If positions change, do not say "no movement", even if the task remains unfinished.
2. Current measured position/trace and current geometry override old model judgments
   and summaries. Summaries may be wrong about arrival: check the trace endpoint
   against mapped boundaries.
   Produce a fresh CURRENT stage, location and reason; do not copy a prior review.
   position_review.location and .reason must be grounded in the CURRENT map's
   position/agent_region_ids and retained trace, never a raw observation or the old
   initialization description. Anchor the route reason in the current region and
   the actual start/end position comparison, then interpret its task significance.
3. Map axes: +X right, +Z down, coordinates in meters. Blue arrow is heading, not
   proof of a travel turn. Blue trace is retained motion, not proof of correctness.
   Red is withdrawn/rollback history, not obstacles or completed required progress.
   White is observed, black obstacle, dark gray unknown; colors are geometric regions.
   Green object markers are depth-localized candidates, not verified modifiers/identity.

USE THE TRACE TO INTERPRET THE INSTRUCTION
4. Follow the retained trace chronologically through named regions. Seeing/naming a
   later destination early is harmless. Actually entering it before an explicitly
   required earlier traversal can be off_route, when the identities and crossings
   are supported. Naming time is not entry time. Entry and returning out of the same
   doorway is not automatically 'through'. Missing evidence does not prove a violation.
5. Infer substantial travel-direction changes from sustained incoming/outgoing
   portions of the trace. Ignore small kinks, stationary heading changes and minor
   steering. For an instruction-constrained turn, locate the bend relative to its
   referenced target and use scale/coordinates to judge the required spatial relation,
   including whether the turn is early or beyond that target. Increasing distance
   alone does not prove passing. No universal
   arrival radius is given. An outgoing segment too short to establish a turn is uncertain.
6. Segmentation may change as doors are revealed. Reinterpret old trace against
   CURRENT regions, not merely old R IDs or names. A polygon split is not physical
   entry. One broad room label is not evidence of being at its specific entrance.
   Conversely, if the mapped region itself is the identified entrance, being inside
   it can establish 'in the entrance'; no unmentioned exact pixel/coordinate is needed.
7. Read relations and references in the ORIGINAL text. The target inventory is not
   an automatic visit checklist: only required motion relations constrain progress.
   Two descriptions can refer to the same place; do not invent another move merely
   because two phrases were extracted. If mapped labels conflict with that reading,
   report the semantic uncertainty instead of pretending the route is completed.

DECIDE ROUTE AND STOP SEPARATELY
8. route_state normal means current progression is consistent with the required route;
   it does NOT require reaching the final destination or proving every future clause.
   Supported straight progression through a required room may be normal while STOP
   is continue. Use off_route only for a positively supported spatial/order violation;
   uncertain means current route correspondence itself is unresolved.
9. Evaluate the consequence of STOP at the CURRENT position, keeping the actual
   earlier trace fixed. Choose stop when required prior route relations are supported
   and stopping here meets the final spatial/stay condition; choose continue when
   stopping here would fail because further required travel/spatial progress remains;
   choose uncertain when the map cannot decide. Recognition alone is not arrival.
   A future target being unmapped does not invalidate earlier supported progress.
10. When required prior route relations and the final location relation are supported,
   an instruction to remain/wait there can be fulfilled by STOP NOW. Requiring that
   waiting already occurred before allowing STOP reverses the task logic. Evaluate
   the consequence of stopping at the current location, not a past wait record.
   No prior wait action, pause, timer or already-completed waiting must be observed.
   The completed list describes supported PAST clauses; it need not already contain
   wait/stay for stop to be correct. Apply the stated endpoint relation, without
   inventing another exact coordinate or extra boundary requirement.

OUTPUT ONE CONSISTENT CURRENT VERDICT
11. map_evidence.route and .stop are independent: sufficient=true/false, existing
   region_ids/object_ids, physical route_steps and one short concrete relation.
   Cite measured/map references rather than copying a summary. If insufficient,
   abstain: no correction, forced STOP or veto of native STOP. Visibility is attached
   from map projection by the controller; do not generate or infer absent targets.
12. Before returning, check that stage/location/reason/completed and STOP describe
   the SAME current map and trajectory facts. Infer that common factual reading
   FIRST, then derive route and prospective STOP from it within this ONE response.
   The two decisions differ in purpose, not in where the agent is or whether it moved.
   Do not claim both 'no movement' and
   'route completed'. Do not assert unobserved progress or repeat placeholder text.
   'Completed' contains only supported original clauses; partial progress is allowed.

Return one JSON object, no markdown. Keep reasons brief:
{"route_state":"normal",
 "position_review":{"stage":"current stage","completed":[],"completion_status":"current progress",
 "location":"current mapped location","history_match":"trace-based comparison","reason":"current spatial evidence"},
 "stop_proposal":{"decision":"uncertain","reason":"whether STOP here now meets the instruction"},
 "map_evidence":{"route":{"sufficient":false,"region_ids":[],"object_ids":[],"route_steps":[],"relation":"evidence"},
                 "stop":{"sufficient":false,"region_ids":[],"object_ids":[],"route_steps":[],"relation":"evidence"}}}
route_state is normal/off_route/uncertain; STOP decision is stop/continue/uncertain.
stop cannot coexist with off_route. Optional region_label_conflicts lists affected
instruction target IDs only. Do not output actions, paths or region masks.'''


REGION_PROMPT='''Ground the eligible instruction region descriptions in CURRENT observations and map geometry.
Observation supplies semantic identity; depth projection supplies spatial correspondence.
This task names SPATIAL AREAS only. A physical object or architectural structure is
not a name for its containing region. If an eligible description mistakenly denotes
an object rather than an area, omit it; do not label the surrounding floor/room with
that object's name. Seeing the object does not establish that the whole region is it.
The map alone does not determine the meaning of an unnamed region. Name existing
regions only; never merge, split or redraw their boundaries.
This is a SPARSE target-to-region matching task, NOT a task to name every region
or assign every target. Many geometric regions may legitimately remain unnamed.
For each target, compare the supported candidates against its COMPLETE description;
name only regions with a justified semantic correspondence. A target being unused, a
region being unnamed or all candidates being imperfect never justifies a guess.
If the best correspondence is unavailable/not eligible, do not use a less suitable
leftover region as a substitute. Leave the target unassigned in this response.

1. Interpret the COMPLETE target description from the original instruction. Distinguish
   its semantic head (the entity/place to name), referenced landmarks and all identity-
   distinguishing modifiers or relations. Do not substitute a referenced landmark for
   the target itself, or match only a generic category while ignoring its modifiers.
2. Identify the described target in the CURRENT RGB views. Verify the distinguishing
   properties and relations from observed semantics and perspective. Generic visual
   features shared by several places do not establish a unique identity. The original
   instruction defines what to search for; it is not proof that a candidate is that target.
3. Locate the observed target's corresponding floor/extent using depth-projected samples,
   image region labels and current region_view_support. Semantic cues and their relevant
   floor need not occupy the same pixels: establish the spatial correspondence, not
   merely overlap of a cue's image position and a region label. Select the supported
   region or adjacent ordinary regions whose extent corresponds to the described place.
4. Distinguish an opening/boundary, the space reached through it, the place containing
   the robot and a referenced neighboring place. Visibility in one view does not make
   them the same region. Do not transfer a semantic cue to a different place just
   because it appears along the same viewing direction or shares a geometric outline.
5. Use ONLY eligible region IDs and currently supported region/view pairs. Evidence
   must state an actual distinguishing observation/relation AND why the selected
   projected region is its correspondence. If identity, a necessary modifier, or
   correspondence is unresolved, omit the assignment. Map names and older descriptions
   are fallible context, not independent current observation evidence. An unnamed
   region is an acceptable map state, not a missing answer that must be filled.
6. Assign at most one target label per region ID. Adjacent regions on the same floor
   may share the same target ID only when the observations support that they are parts
   of the same described physical place. Cite that continuity and the specific visual
   evidence for each region. Preserve their individual IDs and boundaries. Corridors,
   passageways and doorways must have separate labels if named. A shared category or
   proximity alone is insufficient. Abstain when the visual relationship is unclear.
7. If geometry refines or splits an older region, re-establish the semantic-to-spatial
   correspondence from current observations. A supported child may keep the old name.
   Apply the same adjacency and observation checks before sharing it with any other
   child. A new region ID alone provides no evidence of a new semantic identity.
8. A successful assignment locks that region for the current correction cycle. Empty
   output consumes no allowance. confirmed requires supported identity, distinguishing
   relations and correspondence; tentative does not excuse ignoring the target description.

Return {"room_labels":[]} when no assignment is supported. Otherwise:
{"room_labels":[{"target_id":"an eligible target ID","region_id":"an eligible R ID",
"view":"a currently supported view","confidence":"confirmed",
"evidence":"actual distinguishing observation/relation and projected correspondence"}]}.
confidence is confirmed or tentative. No generic placeholder evidence, actions, STOP,
navigation verdicts, masks, polygons, boundary edits or scene-specific assumptions.'''


POINT_PROMPT='''Choose ONE local correction waypoint for the ORIGINAL unfinished instruction.
1. Your sole image is the latest MAP. Also use the ORIGINAL instruction, supplied
   text summaries, position history and measured spatial/trajectory metadata.
   No discrete action history is supplied. Respect the required order of physical
   visits and turns inferred from the trace; early visibility is not entry or completed progress.
2. Map axes are fixed world X right / Z down; the blue arrow gives robot heading.
   Blue route is retained progress. Red routes are withdrawn motion or actual rollback;
   neither color proves a goal completed. Red lines are not walls. Consult their logged
   reason before revisiting; do not repeat the same deviation without new evidence.
3. Listed P IDs are reachable candidates on the current observed floor. Their relative coordinates
   are [right, forward] meters in the CURRENT robot frame, defined by the blue arrow.
4. Use the supplied progress and original landmark identities. Do not repeat completed clauses.
Choose a point that restores the required route or approaches a relevant visible boundary.
Do not invent a hidden destination. Never substitute an unrelated salient object for a named target.
Return {"point_id":"P1","reason":"instruction-grounded reason"}.
If no listed point has adequate semantic support, return {"point_id":null,"reason":"..."}.
The controller handles A*. Do not output paths, global STOP, or arbitrary coordinates.'''

SUMMARY_PROMPT='''Summarize this actual 16-step navigation interval for the ORIGINAL instruction.
Your sole image is the latest map; supplied ordered positions describe the interval.
All environmental evidence comes from this map, mapped labels and measured trace.
No RGB, initial observation description or naming-observation narrative is supplied.
Do not invent view contents or summarize observations; compress mapped progress only.
No motor commands, action history, crossing-event tables or turn-event tables are supplied.
Infer spatial entry order and substantial changes of travel direction from the trace.
Recognition may occur after a visit: use the trace, not naming time. Early visibility
is not entry; small steering is not a route turn; red paths are not completed progress.
Preserve uncertainty and distinguish entering from traversing to the required side.
Verify the interval's final position against the mapped region before claiming arrival;
seeing goal labels on the map is not enough. If current map context is later than the
interval end, it supplies geometry only; use the interval's final position for progress.
Do not import motion after the interval end; the map is current geometric context.
Return {"summary":"concise progress and uncertainty","grounding":"visual evidence",
"anchor_steps":[two distinct supplied state IDs]}.
Choose anchors that best preserve a route-relevant transition; otherwise use interval endpoints.'''


# --- planner.py ---
"""Governor VLM requests, target extraction, map review and waypoint selection."""
class Planner(HistoryPlanner):
    def __init__(self,*args):
        super().__init__(*args)
        self.targets=[]
        self.map=None
        self.pose=None
        self.last_grid=None
        self.initialization_description=None

    def ask(self,prompt,data,images=(),**kwargs):
        # RGB is limited to annotation and descriptive initialization scans.
        if prompt not in (REGION_PROMPT,LOCATION_PROMPT) and any(not label.startswith('LATEST MAP') for label,_ in images):
            raise ValueError('Governor inference accepts map images only; RGB is restricted to naming and initialization descriptions')
        return super().ask(prompt,data,images,**kwargs)

    def context(self,include_initial=True):
        text='Map-derived historical progress summaries (fallible; verify against current map/trace):\n'+json.dumps(self.summaries,ensure_ascii=False)
        if include_initial and self.initial_location is not None:
            text+='\nEPISODE INITIALIZATION description at state 0 ONLY (historical semantic context, not current progress): '+json.dumps(self.initial_location,ensure_ascii=False)
        if include_initial and self.initialization_description is not None:
            text+='\nLATEST CORRECTION INITIALIZATION snapshot ONLY (not proof of current pose or route completion): '+json.dumps(self.initialization_description,ensure_ascii=False)
        return text

    def inference_metadata(self,grid):
        metadata=dict(self.map.metadata(grid))
        # Keep semantic annotation results, not RGB descriptions used to name them.
        metadata['semantic_regions']=[{k:v for k,v in row.items() if k!='evidence'}
            for row in metadata['semantic_regions']]
        return metadata

    def actions(self,history,end=None):
        # Compatibility guard for inherited callers: NEVER serialize frame.action.
        end=len(history)-1 if end is None else end
        step=history[end].physical_step
        rows=[r for r in self.map.route if r['step']<=step]
        return 'Measured map positions only (not commands): '+json.dumps(rows,ensure_ascii=False)

    def summarize(self,history):
        begin,end=self.boundary,self.boundary+WINDOW
        grid=self.map.grid();self.map.candidates(grid)
        final_step=history[end].physical_step
        rows=[r for r in self.map.route if r['step']<=final_step]
        data='Earlier progress (fallible): '+json.dumps(self.summaries,ensure_ascii=False)
        data+='\nInterval state IDs: '+json.dumps([dict(state=i,physical_step=history[i].physical_step)
            for i in range(begin,end+1)])
        data+='\nMeasured retained positions up to interval end: '+json.dumps(rows)
        metadata=self.inference_metadata(grid)
        data+='\nCurrent map context (not additional interval motion): '+json.dumps(
            {k:v for k,v in metadata.items() if k not in ('retained_route','red_routes')},ensure_ascii=False)
        result=self.ask(SUMMARY_PROMPT,data,[('LATEST MAP; spatial context, no future motion evidence',self.map.render(grid))])
        anchors=[]
        for value in result.get('anchor_steps',[]):
            if type(value) is int and begin<=value<=end and value not in anchors:anchors.append(value)
        fallback=len(anchors)<2
        for value in (begin,end):
            if len(anchors)<2 and value not in anchors:anchors.append(value)
        return dict(text=result['summary'],grounding=result.get('grounding',''),anchor_steps=sorted(anchors[:2]),
            anchor_source='boundary_fallback' if fallback else 'vlm',evidence_source='map_positions_only')

    def extract(self):
        def validate(value):
            names={}
            for category in ('regions','stuffs'):
                rows=value.get(category)
                if not isinstance(rows,list):raise ValueError('Return separate regions and stuffs lists')
                names[category]=set()
                for row in rows:
                    if not isinstance(row,dict):raise ValueError('Inventory entries must be objects')
                    for key in ('name','head','query'):
                        if not isinstance(row.get(key),str) or not row[key].strip():raise ValueError('Missing inventory '+key)
                    if not isinstance(row.get('modifiers'),list) or any(not isinstance(v,str) for v in row['modifiers']):
                        raise ValueError('modifiers must be a string list')
                    if row['name'] in names[category]:raise ValueError('Duplicate inventory name')
                    names[category].add(row['name'])
            clauses=value.get('clauses')
            if not isinstance(clauses,list) or not clauses:raise ValueError('clauses must cover the full original instruction')
            used={key:set() for key in names};texts=[]
            for index,clause in enumerate(clauses):
                if not isinstance(clause,dict) or not isinstance(clause.get('text'),str):raise ValueError('Missing clause text')
                texts.append(clause['text'])
                for key in names:
                    refs=clause.get(key)
                    if not isinstance(refs,list) or any(type(v) is not int or not 0<=v<len(value[key]) for v in refs):
                        choices={i:row['name'] for i,row in enumerate(value[key])}
                        raise ValueError('clauses[%d].%s must select zero-based INTEGER indices from %s; use [] for no references, never repeat name strings'%
                            (index,key,json.dumps(choices,ensure_ascii=False)))
                    clause[key]=list(dict.fromkeys(refs))
                    used[key].update(clause[key])
            norm=lambda text:re.sub(r'\s+','',text)
            if norm(''.join(texts))!=norm(self.instruction):raise ValueError('Clause spans must preserve ALL original instruction text in order, including punctuation')
            if any(used[key]!=set(range(len(value[key]))) for key in names):
                raise ValueError('Every inventory entry must be selected by its integer index in at least one clause')
            return value
        trace=lambda **r:self.log(dict(phase='target_extraction_reply',**r))
        result=self.ask(TARGET_PROMPT,'Extract all object AND region targets from the original instruction.',
            validator=validate,trace=trace)
        if not result['regions'] and not result['stuffs']:
            self.log(dict(phase='target_extraction_empty',action='independent_text_audit'))
        def audit(value):
            validate(value)
            if not value['regions'] and not value['stuffs'] and (value.get('no_named_targets') is not True or
                    not isinstance(value.get('reason'),str) or not value['reason'].strip()):
                raise ValueError('Empty inventories require an explicit no_named_targets verdict and instruction-grounded reason')
            return value
        draft=copy.deepcopy(result)
        try:
            result=self.ask(TARGET_AUDIT_PROMPT,'Audit draft (may contain omissions): '+json.dumps(draft,ensure_ascii=False),
                validator=audit,trace=lambda **r:self.log(dict(phase='target_extraction_audit_reply',**r)))
        except (ValueError,TypeError) as exc:
            # A failed audit must not destroy a structurally valid, nonempty extraction.
            # An empty draft still needs the explicit no-target audit; do not silently accept it.
            if not draft['regions'] and not draft['stuffs']:raise
            result=draft
            self.log(dict(phase='target_extraction_audit_fallback',reason=str(exc),
                source='validated_extraction_draft',audit_confirmed=False))
        # A reference can only take its complete name from the selected inventory row.
        # Resolve after auditing, against that response's FINAL lists, never by aliases.
        for clause in result['clauses']:
            for category in ('regions','stuffs'):
                clause[category]=[result[category][index]['name'] for index in clause[category]]
        self.inventory=result
        self.targets=[]
        for category,kind in (('regions','region'),('stuffs','object')):
            for row in result[category]:
                self.targets.append(dict(row,id='T%d'%(len(self.targets)+1),kind=kind))
        self.log(dict(phase='instruction_targets',targets=self.targets,
            regions=result['regions'],stuffs=result['stuffs'],clauses=result['clauses'],
            empty_confirmed=not bool(self.targets),reason=result.get('reason','')))
        return self.targets

    def pending_region_targets(self,result,grid,step,views,mode):
        allowed={t['id']:t['name'] for t in self.targets if t['kind']=='region'}
        supported=self.map.region_views(grid,step,views)
        if not any(not self.map.naming_counts.get(rid,0) for rid in supported):return {}
        return allowed

    def scene(self,history,views=None,mapping=False):
        grid=self.map.grid(); self.map.candidates(grid); self.last_grid=grid
        data='CURRENT state: %d, physical step: %d. All decisions refer to this state.'%(len(history)-1,history[-1].physical_step)
        data+='\nInstruction targets: '+json.dumps(self.targets,ensure_ascii=False)
        metadata=self.map.metadata(grid) if mapping else self.inference_metadata(grid)
        if mapping:
            metadata={k:v for k,v in metadata.items() if k not in ('retained_route','red_routes')}
        data+='\nLatest map: '+json.dumps(metadata,ensure_ascii=False)
        if mapping:
            support=self.map.region_views(grid,history[-1].physical_step,list(views) if views else ['front'])
            data+='\nregion_view_support: '+json.dumps({rid:{name:dict(image_box=row['image_box'],
                floor_cells=len(row['cells'])) for name,row in names.items()} for rid,names in support.items()})
        data+='\nmapped_visibility (controller-derived): '+json.dumps(
            self.mapped_visibility(grid,history[-1].physical_step,views),ensure_ascii=False)
        if not mapping:
            data+='\nHISTORICAL CONTEXT ONLY; defer to current map/trace on conflict:\n'+self.context()
            data+='\nCURRENT verdict requested at physical step %d; current world position %s. Re-read the current map and trace above; do not copy a historical stage or stationary claim.'%(history[-1].physical_step,json.dumps(self.map.position.tolist()))
            data+='\nFor route reasons use the CURRENT MAP and retained trace. In this single response, decide whether issuing STOP HERE NOW would satisfy the instruction, not whether waiting has already happened.'
        images=[]
        if mapping:
            current=views or {'front':history[-1].image()}
            images = self.naming_images(current, grid, history[-1].physical_step)
        images += [('LATEST MAP; fixed world coordinates; labels refer to this map only',self.map.render(grid))]
        return data,images,grid

    def naming_images(self, views, grid, step):
        images = []
        sectors = []
        for name, image in views.items():
            overlay = self.map.naming_overlay(image, grid, step, name)
            if re.fullmatch(r'sector_\d{2}', name):
                sectors.append((name, overlay))
            else:
                images.append(('CURRENT ' + name + ' RGB for map NAMING ONLY; R labels mark depth-supported floor', overlay))

        # Pack perspective views without changing the IDs used by depth projection.
        sectors.sort(key=lambda pair: pair[0])
        for start in range(0, len(sectors), 8):
            batch = sectors[start:start + 8]
            sheet = Image.new('RGB', (4 * 224, ((len(batch) + 3) // 4) * 184), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, (name, image) in enumerate(batch):
                x, y = (index % 4) * 224, (index // 4) * 184
                draw.text((x + 4, y + 2), name, fill='black')
                tile = image.copy()
                tile.thumbnail((224, 168), Image.Resampling.LANCZOS)
                sheet.paste(tile, (x + (224 - tile.width) // 2, y + 16))
            images.append(('CURRENT PANORAMA RGB for map NAMING ONLY. Each tile label is its view ID. '
                           'R labels mark depth-supported floor in that view.', sheet))
        return images

    def mapped_visibility(self,grid,step,views=None):
        support=self.map.region_views(grid,step,list(views) if views else ['front'])
        rows=[]
        for target in self.targets:
            row=dict(target_id=target['id'],state='uncertain',view=None,source='map_projection',
                evidence='No current mapped observation support; this does not prove absence.')
            if target['kind']=='object':
                objects=[(oid,obj) for oid,obj in self.map.objects.items() if
                    obj['target_id']==target['id'] and obj['floor']==self.map.active and obj['last_seen']==step]
                if objects:
                    row.update(state='visible',evidence='Current depth-projected object candidates: '+
                        ', '.join(oid for oid,_ in objects)+'. Candidate identity/modifiers remain unverified.')
            else:
                matches=[(r['region_id'],support[r['region_id']]) for r in self.map.semantics.values()
                    if r['target_id']==target['id'] and r['floor']==self.map.active and
                    r.get('association')=='matched' and r.get('cells') and r['region_id'] in support]
                if matches:
                    names=sorted({view for _,by_view in matches for view in by_view})
                    row.update(state='visible',view=','.join(names),evidence='Current depth floor support in '+
                        ', '.join(sorted({rid for rid,_ in matches}))+'. Region identity is inherited from map naming.')
            rows.append(row)
        return rows

    def review(self,history,mode='normal',views=None,**unused):
        if not self.pose.valid:
            result=dict(route_state='uncertain',visibility=[dict(target_id=t['id'],state='uncertain',view='front',
                evidence='Localization unavailable') for t in self.targets],position_review=dict(stage='Uncertain',completed=[],
                completion_status='uncertain',location='uncertain',history_match='unavailable',reason=self.pose.reason),
                stop_proposal=dict(decision='uncertain',reason='Localization unavailable'))
        else:
            data,images,grid=self.scene(history,views)
            expected={t['id'] for t in self.targets}
            def validate(v):
                if v.get('route_state') not in ('normal','off_route','uncertain'): raise ValueError('invalid route_state')
                # Projection support is a deterministic map field, not a mandatory
                # VLM transcription task. Legacy/missing/partial replies cannot
                # invalidate otherwise usable route and endpoint judgments.
                v['visibility']=self.mapped_visibility(grid,history[-1].physical_step,views)
                p=v.get('position_review'); s=v.get('stop_proposal')
                if not isinstance(p,dict) or not isinstance(p.get('completed'),list) or any(not isinstance(x,str) for x in p['completed']):
                    raise ValueError('position_review.completed must be string list')
                for k in ('stage','completion_status','location','history_match','reason'):
                    if not isinstance(p.get(k),str): raise ValueError('missing position_review.'+k)
                if not isinstance(s,dict) or s.get('decision') not in ('stop','continue','uncertain') or not isinstance(s.get('reason'),str):
                    raise ValueError('invalid STOP decision')
                if s['decision']=='stop' and v['route_state']=='off_route': raise ValueError('STOP contradicts deviation')
                conflicts=v.get('region_label_conflicts',[])
                if not isinstance(conflicts,list) or any(not isinstance(t,str) or t not in expected for t in conflicts):
                    raise ValueError('region_label_conflicts must list supplied target IDs')
                evidence=v.get('map_evidence')
                if not isinstance(evidence,dict):raise ValueError('Missing map_evidence for route and stop')
                steps={r['step'] for r in self.map.route}
                for kind in ('route','stop'):
                    row=evidence.get(kind)
                    if not isinstance(row,dict) or type(row.get('sufficient')) is not bool or not isinstance(row.get('relation'),str) or not row['relation'].strip():
                        raise ValueError('Missing '+kind+' map evidence sufficient/relation')
                    for key,ids in (('region_ids',set(grid['regions'])),('object_ids',set(self.map.objects)),('route_steps',steps)):
                        refs=row.get(key)
                        if not isinstance(refs,list) or any(type(ref) not in (str,int) or ref not in ids for ref in refs):
                            raise ValueError('Evidence must reference existing '+key)
                    if row['sufficient'] and not any(row[k] for k in ('region_ids','object_ids','route_steps')):
                        row['sufficient']=False
                if not evidence['route']['sufficient']:
                    v['route_state']='uncertain'
                    prefix='ABSTAIN: insufficient map route evidence. '
                    p['reason']=prefix+re.sub(r'^(?:ABSTAIN: insufficient map route evidence\.\s*)+','',p['reason'])
                if not evidence['stop']['sufficient'] or s['decision']=='stop' and v['route_state']=='uncertain':
                    s['decision']='uncertain'
                    prefix='ABSTAIN: insufficient map endpoint/route evidence. '
                    s['reason']=prefix+re.sub(r'^(?:ABSTAIN: insufficient map endpoint/route evidence\.\s*)+','',s['reason'])
                return v
            self.log(dict(phase='navigation_evidence',source='map_only',mode=mode,
                physical_step=history[-1].physical_step,image_labels=[label for label,_ in images],
                map=self.map.metadata(grid)))
            result=self.ask(REVIEW_PROMPT,data+'\nMode: '+mode,images,tokens=self.args.review_max_new_tokens,
                validator=validate,trace=lambda **r:self.log(dict(phase='map_review_reply',**r)))
        result['route_check']=dict(off_route=result['route_state']=='off_route',reason=result['position_review']['reason'])
        result['stop_proposal']['reject_vln_stop']=result['stop_proposal']['decision']=='continue'
        self.position_reviews.append(dict(state=len(history)-1,observed_at=history[-1].physical_step,
            source=mode,route_state=result['route_state'],**result['position_review']))
        return result

    def waypoint(self,history,views,review):
        data,images,grid=self.scene(history,views)
        points=self.map.reachable_points(grid,self.args.point_max_distance)
        if not points: return None
        rotation=self.pose.pose[:3,:3]
        for p in points:
            local=rotation.T@(np.array(p['world'])-self.map.position)
            p['relative_right_forward']=[round(float(local[0]),2),round(float(-local[2]),2)]
        images[-1]=('LATEST MAP with reachable P candidate IDs',self.map.render(grid,points))
        data+='\nProgress: '+json.dumps(review,ensure_ascii=False)+'\nCandidates: '+json.dumps(points)
        ids={p['id']:p for p in points}
        def validate(v):
            if 'point_id' not in v or v['point_id'] is not None and v['point_id'] not in ids: raise ValueError('select a listed P ID or null')
            if not isinstance(v.get('reason'),str): raise ValueError('missing reason')
            return v
        answer=self.ask(POINT_PROMPT,data,images,validator=validate)
        self.log(dict(phase='map_point_selection',**answer,candidates=points))
        return None if answer['point_id'] is None else np.array(ids[answer['point_id']]['world'])


# --- episode.py ---
"""Observed-map episode controller, initialization, rollback and correction."""
class Episode(LegacyEpisode):
    segmenter=None

    def __init__(self,*args):
        super().__init__(*args)
        cfg=self.policy.e.sim_sensors_config
        self.pose=PoseProvider(self.args.map_pose_source,cfg.depth_sensor,cfg.rgb_sensor)
        self.map=TopdownMap(self.args.map_resolution,self.args.map_robot_radius,
            self.args.map_robot_height,self.args.map_depth_stride,self.args.map_max_cells)
        self.map.floor_min_height=self.args.map_floor_min_height
        self.map.observed_min_height=self.args.map_observed_min_height
        for name in ('room_wall_height','room_wall_length','room_wall_gap','room_door_min','room_door_max','room_side_area','room_corridor_width','room_corridor_length'):
            setattr(self.map,name,getattr(self.args,name,getattr(self.map,name)))
        for name in ('filter_level','line_support','edge_support','sigma','eps',
                     'passage_min_length','passage_max_length','passage_max_width','passage_min_side_area',
                     'door_min_width','door_max_width'):
            setattr(self.map.rose2,name,getattr(self.args,'rose_'+name,getattr(self.map.rose2,name)))
        self.vlm.map,self.vlm.pose=self.map,self.pose
        self.status_records=[]; self.target=None; self.correction_steps=0
        self.pose_failures=0; self.invalid_reviews=0; self.native_stop_retries=0
        self.map_poses={}

    def observation(self,raw,view='front',yaw=0.):
        cfg=self.policy.e.sim_sensors_config.depth_sensor
        camera=self.pose.camera(yaw)
        if self.pose.selected=='sim':
            import quaternion
            sensor=self.env.sim.get_agent_state().sensor_states['depth']
            camera[:3,3]=sensor.position
            camera[:3,:3]=quaternion.as_rotation_matrix(sensor.rotation)
        rh,rw=np.asarray(raw['rgb']).shape[:2]
        dh,dw=np.asarray(raw['depth']).squeeze().shape
        if abs(rw/rh-dw/dh)>1e-4:
            raise ValueError('RGB/depth aspect ratios differ; explicit registration is required')
        return Observation(np.asarray(raw['rgb'])[...,:3].copy(),np.asarray(raw['depth']).copy(),
            camera[:3,3],camera[:3,:3],self.pose.pose[:3,3],cfg.hfov,cfg.min_depth,cfg.max_depth,
            cfg.normalize_depth,self.steps,view)

    def fuse(self,raw,view='front',yaw=0.,refresh=False):
        if not self.pose.valid: return
        obs=self.observation(raw,view,yaw)
        if refresh:self.map.seen.discard((obs.step,obs.view))
        if (obs.step,obs.view) in self.map.seen:
            self.event('map_observation_reused',view=view,reason='same step and view already fused')
            return
        detections=self.segmenter.predict(obs.rgb,self.vlm.targets)
        self.map.update(obs,detections)
        self.event('map_observation',view=view,instances=len(detections),map_version=self.map.version,
            localization_source=self.pose.selected,floor=self.map.active,transition=self.map.transition,
            projection=self.map.observation_stats[view],
            queried_objects=[dict(id=t['id'],name=t['name'],query=t['query']) for t in self.vlm.targets if t['kind']=='object'],
            mapped_objects=[dict(id=oid,target_id=row['target_id'],name=row['label'],center=row['center'],floor=row['floor'])
                for oid,row in self.map.objects.items() if row['last_seen']==obs.step])

    def update_pose(self,raw,retain=True):
        if self.pose.update(raw,self.env.sim):
            self.pose_failures=0
            floor=self.map.set_pose(self.pose.pose[:3,3],-self.pose.pose[:3,2],self.steps,retain)
            self.map_poses[self.steps]=self.pose.pose.copy()
            self.current.map_floor=floor
            self.fuse(raw)
        else:
            self.pose_failures+=1
            self.event('localization_uncertain',reason=self.pose.reason)
            if self.pose_failures>=3: self.termination='localization_lost'

    def scan(self, persist=True):
        import quaternion
        from PIL import Image
        sim=self.env.sim
        state=copy.deepcopy(sim.get_agent_state())
        previous=getattr(sim,'_prev_sim_obs',None)
        views={}
        try:
            for sector in range(24):
                name = f'sector_{sector:02}'
                yaw = -math.radians(15 * sector)
                rotation=state.rotation*quaternion.from_rotation_vector([0.,yaw,0.])
                raw=sim.get_observations_at(state.position,rotation,keep_agent_at_new_pose=True)
                views[name]=Image.fromarray(raw['rgb'][...,:3].copy())
                self.fuse(raw,name,yaw,refresh=True)
        finally:
            sim.get_agent(0).set_state(state,reset_sensors=False,infer_sensor_states=False)
            if hasattr(sim,'_prev_sim_obs'): sim._prev_sim_obs=previous
        if persist:
            self.map.save(self.folder)
        grid=self.map.grid();self.map.candidates(grid)
        self.event('map_scan_complete',coverage_degrees=360,degrees_per_sector=15,views=list(views),
            projection={name:self.map.observation_stats.get(name) for name in views},
            observed_free_cells=int(grid['free'].sum()),occupied_cells=int(grid['occupied'].sum()),
            region_cells=sum(len(row['cells']) for row in grid['regions'].values()))
        return views

    def panorama_image(self, views):
        hfov = self.policy.e.sim_sensors_config.rgb_sensor.hfov
        panorama = Image.new('RGB', (24 * 64, 350), 'white')
        draw = ImageDraw.Draw(panorama)
        for sector in range(24):
            image = views[f'sector_{sector:02}']
            width, height = image.size
            focal = width / (2 * math.tan(math.radians(hfov) / 2))
            half = focal * math.tan(math.radians(7.5))
            center = (width - 1) / 2
            left = max(0, round(center - half))
            right = min(width, round(center + half) + 1)
            strip = image.crop((left, 0, right, height)).resize((64, 320))
            x = sector * 64
            panorama.paste(strip, (x, 30))
            draw.text((x + 3, 2), f'{sector:02} {sector * 15}', fill='black')
            draw.line((x, 0, x, 349), fill='yellow', width=1)
        return panorama

    def locate_panorama(self, views, initial=False):
        panorama = self.panorama_image(views)
        filename = 'initial_panorama.png' if initial else 'correction_panorama.png'
        panorama.save(self.folder / filename)
        label = ('CURRENT 360-degree PANORAMA, clockwise from the robot heading. '
                 '00=front, 06=right, 12=back, 18=left. Labels give sector and angle.')
        return self.vlm.locate({label: panorama}, initial=initial)

    def move(self,action,phase,retain=True,controller=False):
        # Use the original action interface, logging, video and StreamVLN frames.
        # Avoid the legacy point-completion callback: tasks remain a single global task.
        before=self.map.position.copy(); forward=self.map.forward.copy()
        old_blocked=self.blocked
        ok=super().move(action,phase,retain=retain,controller=controller)
        if not ok: return False
        if action!=0:
            raw=getattr(self.current,'raw_observations',self.current.observations)
            self.update_pose(raw,retain)
            if self.pose.valid and action==1 and np.linalg.norm(self.map.position-before)<.08:
                self.map.mark_collision(before+forward*.25)
                self.blocked=old_blocked+1
            elif action==1: self.blocked=0
            # Persist once in review_now, after the per-action map judgment.
        return True

    def review_now(self,views=None,phase=None):
        phase=phase or ('correction' if self.target is not None else 'normal')
        review=self.call('map_review',lambda:self.vlm.review(self.history,mode=phase,views=views))
        if review is None:
            self.invalid_reviews+=1
            review=dict(route_state='uncertain',stop_proposal=dict(decision='uncertain',reason='Invalid VLM reply'),
                position_review=dict(reason='Invalid VLM reply'))
            self.event('governor_abstained',reason='Invalid review; native policy keeps control',
                consecutive_invalid=self.invalid_reviews)
        else: self.invalid_reviews=0
        row=dict(state=len(self.history)-1,observed_at=self.steps,phase=phase,
            route_state=review['route_state'],eligible=phase!='rollback',
            reason=review['position_review']['reason'])
        self.status_records.append(row)
        self.event('governor_review',**review)
        self.map.save(self.folder)
        return review

    def immediate_stop(self,review,native_proposal=False):
        if self.done or not(native_proposal or review['stop_proposal']['decision']=='stop'):return False
        decision=review['stop_proposal']['decision']
        accepted=decision=='stop' and review.get('route_state')=='normal'
        # Abstention is not a STOP veto. No additional scan or second VLM verdict.
        unopposed=native_proposal and decision=='uncertain' and review.get('route_state')!='off_route'
        self.event('stop_decision',source='streamvln' if native_proposal else 'governor',
            accepted=accepted or unopposed,decision=decision,map_version=self.map.version,
            reason=review['stop_proposal'].get('reason',''),
            map_evidence=review.get('map_evidence',{}),verification=False)
        if not(accepted or unopposed):return False
        phase='native_stop_unopposed' if unopposed else 'governor_stop'
        self.move(0,phase,retain=False)
        self.termination=phase
        return True

    def map_action(self):
        if np.linalg.norm((self.map.position-self.target)[[0,2]])<=self.args.map_goal_tolerance:
            self.target=None; self.point_completed+=1
            self.event('correction_end',reason='waypoint_reached')
            self.policy.restore(self.history,self.vlm.instruction)
            return None
        if self.map.transition:
            self.target=None
            self.policy.restore(self.history,self.vlm.instruction)
            return None
        path=self.map.plan(self.target)
        if not path:
            self.recover_map('path_invalidated',rollback=False)
            return None
        # Choose a short visible segment, then validate the actual 0.25 m discrete action.
        waypoint=next((p for p in path if np.linalg.norm((p-self.map.position)[[0,2]])>=.30),path[-1])
        local=self.pose.pose[:3,:3].T@(waypoint-self.map.position)
        angle=math.atan2(-local[0],-local[2])
        if abs(angle)>math.radians(7.5): return 2 if angle>0 else 3
        grid=self.map.grid()
        for d in np.linspace(.02,.25,12):
            p=self.map.position+self.map.forward*d
            x,z=np.array(self.map.cell(p))-grid['origin']
            if not(0<=z<grid['walk'].shape[0] and 0<=x<grid['walk'].shape[1] and grid['walk'][z,x]):
                self.recover_map('discrete_action_blocked',rollback=False)
                return None
        return 1

    def run(self):
        try:
            return self._run()
        finally:
            if self.map.floors:
                self.map.save_final(self.folder,self.termination or 'interrupted_or_error')



# --- api_client.py ---
"""Stateless OpenAI-compatible Chat Completions transport (standard library only)."""
import base64
import json
import os
from pathlib import Path
import stat
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        old,new=urlsplit(request.full_url),urlsplit(newurl)
        if (old.scheme,old.netloc)!=(new.scheme,new.netloc):
            raise RuntimeError('Refusing API redirect to a different origin')
        return super().redirect_request(request,fp,code,msg,headers,newurl)


class APIVLM:
    def __init__(self, args):
        config={}
        if args.vlm_api_config:
            path=Path(args.vlm_api_config).expanduser()
            if os.name=='posix' and stat.S_IMODE(path.stat().st_mode)&0o077:
                raise ValueError('API config must be private; run chmod 600 on the config file')
            try:
                with path.open(encoding='utf-8-sig') as handle:
                    config=json.load(handle)
            except json.JSONDecodeError:
                raise ValueError('Invalid API config JSON; file contents are not logged') from None
            if not isinstance(config,dict):raise ValueError('API config must be a JSON object')
        key=config.get('api_key') or os.environ.get(args.vlm_api_key_env,'')
        if not isinstance(key,str):raise ValueError('API key must be a string')
        self.key = key.strip()
        if not self.key:
            raise ValueError('Provide api_key in the private JSON config, or set '+args.vlm_api_key_env)
        base_url=args.vlm_api_base_url or config.get('base_url') or 'https://openrouter.icu/v1'
        self.model = args.vlm_api_model or config.get('model') or 'gpt-5.6-terra'
        if not isinstance(base_url,str) or not isinstance(self.model,str) or not self.model.strip():
            raise ValueError('API base_url and model must be nonempty strings')
        parsed=urlsplit(base_url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('VLM API base URL must be HTTPS without credentials, query or fragment')
        self.url = base_url.rstrip('/')+'/chat/completions'
        self.opener=urllib.request.build_opener(_SameOriginRedirect())
        self.timeout = args.local_vlm_request_timeout
        self.retries = args.vlm_api_retries

    def generate(self, messages, tokens):
        converted = []
        for message in messages:
            content = message['content']
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part['type'] == 'image':
                        encoded = part['image']
                        mime = 'image/png' if base64.b64decode(encoded[:32]).startswith(b'\x89PNG') else 'image/jpeg'
                        parts.append(dict(type='image_url', image_url=dict(url='data:'+mime+';base64,'+encoded)))
                    elif part['type'] == 'text':
                        parts.append(dict(type='text', text=part['text']))
                    else:
                        raise ValueError('Unsupported API content type: '+str(part['type']))
                content = parts
            converted.append(dict(role=message['role'], content=content))
        payload = json.dumps(dict(model=self.model, messages=converted,
            max_tokens=tokens, stream=False), ensure_ascii=False).encode('utf-8')
        for attempt in range(self.retries+1):
            request = urllib.request.Request(self.url, data=payload, headers={
                'Authorization':'Bearer '+self.key, 'Content-Type':'application/json'})
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    reply = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429,500,502,503,504) and attempt < self.retries:
                    time.sleep(min(2**attempt,8)); continue
                # Do not print response bodies: gateways can echo request content/credentials.
                raise RuntimeError('VLM API HTTP %d; check endpoint, model/image support, quota and key permissions'%exc.code) from None
            except (urllib.error.URLError, TimeoutError) as exc:
                # Do not retry an ambiguous transport timeout automatically: it may already be billed.
                reason=getattr(exc,'reason',exc)
                detail=(type(reason).__name__+': '+str(reason)).replace(self.key,'[REDACTED]')
                raise RuntimeError('VLM API connection failed ('+detail[:600]+'); no automatic local fallback') from None
        choice = reply['choices'][0]
        content = choice['message'].get('content')
        if isinstance(content,list):
            content=''.join(part.get('text','') for part in content if part.get('type')=='text')
        if not isinstance(content,str) or not content.strip():
            raise ValueError('VLM API returned empty text; choose an image-capable chat model')
        if choice.get('finish_reason') == 'length':
            error=ValueError('VLM API output reached the token limit')
            error.raw_reply=content
            raise error
        return content

    def close(self):
        pass


# --- app.py ---
"""Arguments, worker entry points and original evaluation integration."""
ORIGINAL_ARGUMENTS=base_arguments
ORIGINAL_EVALUATE=base_evaluate
ORIGINAL_CAPTURE=Frame.capture.__func__


def capture(cls,observations,sim,step,phase):
    frame=ORIGINAL_CAPTURE(cls,observations,sim,step,phase)
    # Optional pose supplied by the supervised model/environment adapter is retained.
    frame.raw_observations={k:np.array(v,copy=True) for k,v in observations.items() if k in ('rgb','depth','agent_pose')}
    return frame


def arguments():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--vlm_backend',choices=('local','api'),default='local')
    p.add_argument('--vlm_api_config',default=None,help='Private JSON path; its contents are not copied into results')
    p.add_argument('--vlm_api_base_url',default=None)
    p.add_argument('--vlm_api_model',default=None)
    p.add_argument('--vlm_api_key_env',default='GOVERNOR_API_KEY')
    p.add_argument('--vlm_api_retries',type=int,default=2)
    p.add_argument('--seg_python',default='/home/ubuntun/miniconda3/envs/sam2/bin/python')
    p.add_argument('--seg_gpu',default='5')
    p.add_argument('--dino_model_path',default=str(ROOT/'weights/grounding-dino-base'))
    p.add_argument('--sam_checkpoint',default=str(ROOT/'weights/sam2.1_hiera_base_plus.pt'))
    p.add_argument('--sam_config',default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    p.add_argument('--seg_threshold',type=float,default=.35)
    p.add_argument('--seg_max_instances',type=int,default=12)
    p.add_argument('--seg_timeout',type=float,default=300)
    for name,default in (('room_wall_height',.8),('room_wall_length',1.2),('room_wall_gap',.2),
                         ('room_door_min',.6),('room_door_max',1.4),('room_side_area',1.5),
                         ('room_corridor_width',2.2),('room_corridor_length',2.0)):
        p.add_argument('--'+name,type=float,default=default)
    for name,default in (('rose_filter_level',.18),('rose_line_support',.1),
                         ('rose_edge_support',.1),('rose_sigma',.125),('rose_eps',.85),
                         ('rose_passage_min_length',.5),('rose_passage_max_length',3.5),
                         ('rose_passage_max_width',2.2),('rose_passage_min_side_area',1.5),
                         ('rose_door_min_width',.5),('rose_door_max_width',1.6)):
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--map_pose_source',choices=('auto','provided','sim','rgbd'),default='auto')
    p.add_argument('--map_resolution',type=float,default=.01)
    p.add_argument('--map_floor_min_height',type=float,default=-.30)
    p.add_argument('--map_observed_min_height',type=float,default=-.30)
    p.add_argument('--map_robot_radius',type=float,default=.18)
    p.add_argument('--map_robot_height',type=float,default=1.5)
    p.add_argument('--map_depth_stride',type=int,default=2)
    p.add_argument('--map_max_cells',type=int,default=9000)
    p.add_argument('--map_goal_tolerance',type=float,default=.18)
    p.add_argument('--rollback_tolerance',type=float,default=.15)
    p.add_argument('--rollback_angle_tolerance',type=float,default=8.)
    p.add_argument('--correction_max_steps',type=int,default=60)
    extra,remaining=p.parse_known_args()
    original=sys.argv
    if '--help' in remaining or '-h' in remaining:
        print('Topdown / segmentation options:'); p.print_help(); print('\nOriginal governor options:')
    try:
        sys.argv=[original[0]]+remaining
        args=ORIGINAL_ARGUMENTS(vlm_backend=extra.vlm_backend)
    finally: sys.argv=original
    vars(args).update(vars(extra))
    if any(not np.isfinite(value) or value>0 for value in
            (args.map_floor_min_height,args.map_observed_min_height)):
        p.error('floor/observed lower bounds must be finite offsets <= 0 meters')
    if args.vlm_api_retries<0:p.error('--vlm_api_retries must be nonnegative')
    if any(x<=0 for x in (args.map_resolution,args.map_robot_radius,args.map_robot_height,args.map_depth_stride,
                          args.map_max_cells,args.map_goal_tolerance,args.seg_timeout,args.correction_max_steps,args.seg_max_instances)):
        p.error('map and segmentation sizes/timeouts must be positive')
    if min(args.room_wall_height,args.room_wall_length,args.room_door_min,args.room_side_area,args.room_corridor_width,args.room_corridor_length)<=0 or args.room_wall_gap<0 or args.room_door_max<=args.room_door_min:
        p.error('invalid room geometry thresholds')
    if not 0<args.seg_threshold<1: p.error('seg_threshold must be between 0 and 1')
    if not (0<args.rose_filter_level<1 and 0<args.rose_line_support<1 and
            0<=args.rose_edge_support<1 and args.rose_sigma>0 and 0<args.rose_eps<1):
        p.error('invalid ROSE2 thresholds')
    if not args.seg_gpu.isdecimal(): p.error('seg_gpu must be one physical GPU index')
    if not 0<args.rose_door_min_width<args.rose_door_max_width:p.error('invalid door widths')
    if not (0<args.rose_passage_min_length<=args.rose_passage_max_length and
            args.rose_passage_max_width>=.6 and args.rose_passage_min_side_area>0):
        p.error('invalid short-passage thresholds')
    return args


def evaluate(evaluator,client,args,records):
    Episode.segmenter=Segmenter(args)
    try: return ORIGINAL_EVALUATE(evaluator,client,args,records)
    finally: Episode.segmenter.close()

def segmentation_worker_main(argv):
    p=argparse.ArgumentParser(description='Offline segmentation worker')
    p.add_argument('--dino',required=True)
    p.add_argument('--sam',required=True)
    p.add_argument('--config',default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    p.add_argument('--threshold',type=float,default=.35)
    p.add_argument('--max-instances',type=int,default=12)
    return worker(p.parse_args(argv))



class GridFastDetector:
    """GRID-FAST occupancy-grid adaptation from Fredriksson et al. (2024),
    DOI 10.1007/s10846-024-02180-6, Sections 3.1--3.7.
    Multi-angle gap connectivity, wall contours, openings and >=3-opening regions.
    Optimization uses bounded discrete wall-contour search, not the complete ROS
    optimizer; Section 3.8 skeleton/path export is intentionally not used here.
    All filtering is performed on a detector-local copy, never navigation evidence.
    """
    def __init__(self):
        self.cache_key=None;self.cache=[];self.info={}

    @staticmethod
    def runs(row):
        edges=np.diff(np.r_[False,row,False].astype(np.int8))
        return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1))

    @staticmethod
    def points(a,b):
        count=max(2,int(np.ceil(np.max(np.abs(np.asarray(b)-a))))+1)
        return np.rint(np.linspace(a,b,count)).astype(int)

    def detect(self,grid,robot_cell,res,args,diameter):
        import cv2
        from scipy import ndimage as ndi
        from scipy.spatial import cKDTree
        radius=int(math.ceil(args.node_roi_radius/res));x,z=map(int,robot_cell)
        h,w=grid['free'].shape
        x0,x1=max(0,x-radius),min(w,x+radius+1);z0,z1=max(0,z-radius),min(h,z+radius+1)
        free=grid['free'][z0:z1,x0:x1].copy()
        occupied=grid['occupied'][z0:z1,x0:x1].copy()
        known=grid['observed'][z0:z1,x0:x1].copy()
        offset=grid['origin']+np.array([x0,z0]);h,w=free.shape
        key=(tuple(offset),free.shape,free.tobytes(),occupied.tobytes(),known.tobytes(),
            res,args.gridfast_directions,args.gridfast_unknown_filter,args.gridfast_object_perimeter)
        if key==self.cache_key:return self.cache
        gmin=max(2,int(math.ceil(diameter/res)))
        fuk=max(0,int(round(args.gridfast_unknown_filter/res)))
        # Sec 3.3: short closed obstacle contours may be omitted only in this copy.
        groups,num=ndi.label(occupied)
        for slc in ndi.find_objects(groups):
            if slc is None:continue
            local=occupied[slc];labels=groups[slc]
            for k in np.unique(labels[local]):
                if not k:continue
                component=labels==k
                contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                perimeter=sum(cv2.arcLength(c,True) for c in contours)*res
                if perimeter>args.gridfast_object_perimeter:continue
                # Do not erase an object whose boundary touches an unknown frontier.
                pad=max(1,gmin)
                ya,yb=max(0,slc[0].start-pad),min(h,slc[0].stop+pad)
                xa,xb=max(0,slc[1].start-pad),min(w,slc[1].stop+pad)
                if not known[ya:yb,xa:xb].all():continue
                view=free[slc];view[component]=True
                occupied[slc][component]=False
        # Sec 3.1: bridge bounded interior UNKNOWN runs, not occupied points.
        topology=free.copy()
        for axis in (0,1):
            table=topology if axis==0 else topology.T
            kn=known if axis==0 else known.T
            ob=occupied if axis==0 else occupied.T
            for i in range(len(table)):
                for a,b in self.runs(~kn[i]&~ob[i]):
                    if fuk and b-a+1<=fuk and a>0 and b+1<len(table[i]) and table[i,a-1] and table[i,b+1]:
                        table[i,a:b+1]=True
        frontier=topology&ndi.binary_dilation(~known)&~ndi.binary_dilation(occupied)
        contours,_=cv2.findContours(topology.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        contours=[c[:,0,:].astype(float) for c in contours if len(c)>=3]
        if not contours:
            self.cache_key,self.cache=key,[];return []
        wall_points=np.concatenate(contours);wall_ids=np.concatenate([np.full(len(c),i,int) for i,c in enumerate(contours)])
        tree=cKDTree(wall_points)
        def wall(point):
            distance,index=tree.query(point)
            return int(wall_ids[index]) if distance<=2.5 else None
        def clear(a,b):
            pts=self.points(np.asarray(a),np.asarray(b));xs,zs=pts[:,0],pts[:,1]
            return bool(((xs>=0)&(xs<w)&(zs>=0)&(zs<h)).all() and topology[zs,xs].all())
        def at_frontier(point):
            px,pz=np.rint(point).astype(int)
            return not(0<=px<w and 0<=pz<h) or frontier[max(0,pz-1):min(h,pz+2),max(0,px-1):min(w,px+2)].any()
        detections=[]
        for scan in range(args.gridfast_directions):
            angle=180.*scan/args.gridfast_directions
            size=int(math.ceil(math.hypot(h,w)))+4
            matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,1.)
            matrix[:,2]+=np.array([(size-w)/2,(size-h)/2])
            inverse=cv2.invertAffineTransform(matrix)
            rotated=cv2.warpAffine(topology.astype(np.uint8),matrix,(size,size),flags=cv2.INTER_NEAREST,borderValue=0).astype(bool)
            # Sec 3.2 narrow passages are removed from detection, not from the real map.
            lines=[]
            for row in rotated:
                lines.append([(int(a),int(b)) for a,b in self.runs(row) if b-a>=gmin])
            def world(v):return inverse[:,:2]@np.asarray(v,float)+inverse[:,2]
            for i,gaps in enumerate(lines):
                for a,b in gaps:
                    for direction in (-1,1):
                        j=i+direction
                        if not 0<=j<len(lines):continue
                        connected=[(c,d) for c,d in lines[j] if min(b,d)-max(a,c)>=gmin]
                        if len(connected)<2:continue
                        for order,(c,d) in enumerate(connected):
                            inside='E' if order==0 else 'S' if order==len(connected)-1 else 'B'
                            category=inside if direction==1 or inside=='B' else ('E' if inside=='S' else 'S')
                            aa,bb=world((c,j)),world((d,j))
                            wa,wb=wall(aa),wall(bb)
                            if wa is None or wb is None:continue
                            aa=wall_points[tree.query(aa)[1]].copy();bb=wall_points[tree.query(bb)[1]].copy()
                            point=aa if inside=='S' else bb if inside=='E' else (aa+bb)/2
                            if at_frontier(point):continue
                            detections.append(dict(a=aa,b=bb,point=point,category=category,
                                inward=inverse[:,:2]@np.array([0.,-direction]),walls=(wa,wb)))
        # Cap duplicate detections from consecutive rows without preferring large areas.
        unique=[]
        for record in detections:
            if any(record['category']==old['category'] and np.linalg.norm(record['a']-old['a'])*res<.12 and
                np.linalg.norm(record['b']-old['b'])*res<.12 for old in unique):continue
            unique.append(record)
        detections=unique[:args.gridfast_max_gaps]
        from shapely.geometry import LineString
        candidates=[]
        for i,g1 in enumerate(detections):
            if g1['category']!='S':continue
            for j,g2 in enumerate(detections):
                if g2['category']!='E':continue
                cross=LineString([g1['a'],g1['b']]).intersection(LineString([g2['a'],g2['b']]))
                if cross.is_empty or cross.geom_type!='Point':continue
                a,b=g1['point'],g2['point'];length=np.linalg.norm(b-a)
                if length<gmin or not clear(a,b):continue
                normal=np.array([-(b-a)[1],(b-a)[0]])
                if normal@g1['inward']>0:normal=-normal
                if normal@(np.array(cross.coords[0])-a)>=-1e-6:continue
                candidates.append((float(length),i,j,a,b))
        best={}
        for length,i,j,a,b in candidates:
            for index in (i,j):best[index]=min(best.get(index,float('inf')),length)
        openings=[];used=set()
        for length,i,j,a,b in sorted(candidates,key=lambda c:c[0]):
            if i in used or j in used or length>best[i]+1e-6 or length>best[j]+1e-6:continue
            if at_frontier(a) or at_frontier(b):continue
            openings.append(dict(a=a,b=b,walls=(wall(a),wall(b))));used.update((i,j))
        for i,g in enumerate(detections):
            if i not in used and not at_frontier(g['a']) and not at_frontier(g['b']) and clear(g['a'],g['b']):
                openings.append(dict(a=g['a'],b=g['b'],walls=g['walls']))
        # Sec 3.5: resolve crossing openings with same-wall endpoint swaps; otherwise
        # retain the shorter opening. All trial replacements must remain observed free.
        accepted=[]
        for opening in sorted(openings,key=lambda o:np.linalg.norm(o['b']-o['a'])):
            duplicate=False
            for old in accepted:
                line=LineString([opening['a'],opening['b']]);other=LineString([old['a'],old['b']])
                if line.hausdorff_distance(other)*res<.15:duplicate=True;break
                if not line.crosses(other):continue
                shared=set(opening['walls'])&set(old['walls']);resolved=False
                if shared:
                    for first in (0,1):
                        for second in (0,1):
                            if opening['walls'][first]!=old['walls'][second]:continue
                            o=[opening['a'].copy(),opening['b'].copy()];q=[old['a'].copy(),old['b'].copy()]
                            o[first],q[second]=q[second].copy(),o[first].copy()
                            if min(np.linalg.norm(o[1]-o[0]),np.linalg.norm(q[1]-q[0]))<gmin:continue
                            if clear(*o) and clear(*q) and not LineString(o).crosses(LineString(q)):
                                opening.update(a=o[0],b=o[1]);old.update(a=q[0],b=q[1]);resolved=True;break
                        if resolved:break
                if not resolved:duplicate=True;break
            if not duplicate:accepted.append(opening)
        def components_for(openings):
            cuts=np.zeros_like(topology,np.uint8)
            for op in openings:cv2.line(cuts,tuple(np.rint(op['a']).astype(int)),tuple(np.rint(op['b']).astype(int)),1,1)
            labels,n=ndi.label(topology&~cuts.astype(bool))
            adjacency={}
            for index,op in enumerate(openings):
                pts=self.points(op['a'],op['b']);touch=set()
                for dx,dz in ((0,1),(0,-1),(1,0),(-1,0)):
                    xx,zz=pts[:,0]+dx,pts[:,1]+dz
                    valid=(xx>=0)&(xx<w)&(zz>=0)&(zz<h)
                    touch.update(int(k) for k in labels[zz[valid],xx[valid]] if k)
                if len(touch)!=2:continue
                for k in touch:adjacency.setdefault(k,[]).append(index)
            return labels,adjacency
        labels,adjacency=components_for(accepted)
        # Sec 3.7 bounded wall-contour optimization of opening width plus center
        # distance penalty (Eq.7). Other openings remain fixed, preventing overlap.
        for region,indices in list(adjacency.items()):
            if len(indices)<3:continue
            zz,xx=np.nonzero(labels==region)
            if not len(xx):continue
            center=np.array([xx.mean(),zz.mean()]);minimum=diameter/res
            for index in indices:
                op=accepted[index];best_pair=[op['a'],op['b']]
                def score(a,b):
                    dc=np.linalg.norm((a+b)/2-center)
                    penalty=1e6*(1-dc/minimum) if dc<minimum else dc-minimum
                    return np.linalg.norm(b-a)+penalty
                best_score=score(*best_pair)
                for side in (0,1):
                    wid=op['walls'][side]
                    if wid is None:continue
                    contour=contours[wid];at=int(np.argmin(np.linalg.norm(contour-best_pair[side],axis=1)))
                    for delta in range(-args.gridfast_opt_steps,args.gridfast_opt_steps+1):
                        trial=list(best_pair);trial[side]=contour[(at+delta)%len(contour)]
                        if np.linalg.norm(trial[1]-trial[0])<gmin or not clear(*trial):continue
                        if at_frontier(trial[side]):continue
                        pts=self.points(*trial);values=labels[pts[:,1],pts[:,0]]
                        if region not in values:continue
                        line=LineString(trial)
                        if any(k!=index and line.crosses(LineString([other['a'],other['b']])) for k,other in enumerate(accepted)):continue
                        value=score(*trial)
                        if value<best_score:best_pair=trial;best_score=value
                op.update(a=best_pair[0],b=best_pair[1])
        labels,adjacency=components_for(accepted);results=[]
        for region,indices in adjacency.items():
            if len(indices)<3:continue
            mask=labels==region
            # Definition 3 excludes a complete enclosed internal wall contour.
            boundary,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None and (hierarchy[0,:,3]>=0).any():continue
            zz,xx=np.nonzero(mask&grid['free'][z0:z1,x0:x1])
            if len(xx)*res**2<.2:continue
            cells=set(zip((xx+offset[0]).tolist(),(zz+offset[1]).tolist()))
            polygon=[]
            if boundary:
                contour=max(boundary,key=cv2.contourArea)[:,0,:]
                polygon=(contour[::max(1,len(contour)//48)]+offset).tolist()
            results.append(dict(cells=cells,polygon_cells=polygon,openings=[
                [(accepted[k]['a']+offset).tolist(),(accepted[k]['b']+offset).tolist()] for k in indices],
                branches=len(indices),method='GRID-FAST_gap_opening_region'))
        self.info=dict(method='GRID-FAST',scan_directions=args.gridfast_directions,gap_detections=len(detections),
            openings=len(accepted),intersections=len(results),robot_gap_m=diameter,
            optimization='bounded contour coordinate search of Eq.7; no full ROS Eq.8/global optimizer',
            filtering_scope='topology detector copy only')
        self.cache_key,self.cache=key,results
        return results

"""V6: panoramic initialization and map decision nodes."""

PANORAMA_PROMPT = '''Choose the INITIAL navigation heading for the ORIGINAL instruction.
The panoramic image has 24 numbered 15-degree sectors, clockwise from the original
front: sector 0=front, 6=right, 12=back, 18=left. Each strip shows the actual view
around that sector's center. Read the labels rather than guessing image direction.
Use the original instruction, full target heads/modifiers, observed candidates and
their mapped bearings. A detector's category match does not verify a modifier or
relation. Verify object_matches from actual observations; reject unsupported identity
claims, and keep uncertainty when a relation cannot be seen. Do not substitute a
salient unrelated entity. Choose a heading for the FIRST required traversal, not
automatically toward a later visible destination. Do not claim any motion occurred.
Return {"sector":0,"reason":"...","object_matches":[{"object_id":"O1",
"target_id":"T1","confidence":"confirmed|rejected|uncertain","evidence":"..."}],
"initial_description":{"area":"...","front":"...","right":"...","back":"...","left":"..."}}.
The description is an observation-only initialization snapshot, not completed progress.
sector must be an integer 0..23 or null if no direction is justified. Do not output STOP.'''

NODE_REVIEW_RULES = '''
DECISION-NODE CONTROL
N markers are actual visited, retained decision checkpoints, not desired destinations.
Return is performed by reversing the actual executed action history to that checkpoint;
the history itself is private to the controller and is NOT supplied to this VLM.
Candidate nodes include current geometry/semantic context, position and visit step.
Previous normal/uncertain verdicts are fallible; reevaluate against the LATEST map.
When deviation or late STOP needs recovery, choose the most recent reachable retained
node whose semantic route position is best supported as not deviating. Recency breaks
ties only after route suitability. Do not choose the largest/nearest region blindly.
Early STOP with a suitable current node/progress needs continuation, not mandatory
rollback. Late STOP means an explicitly required stopping relation was ALREADY passed
on the retained trace, not simply that the object was detected earlier. Reinterpret
split regions against their verified semantic names; unverified split labels cannot
prove route/STOP conditions. Red return paths are not completed progress.
This STOP task is called for native_stop, or for correction_stop/node_correction_stop
ONLY during correction after a rejected native STOP. Distinguish early/late/off-route.
Add "stop_proposal.timing": "early|late|on_time|uncertain" and
"recovery":{"needed":false,"node_id":null,"current_node_suitable":true,"reason":"..."}.
needed=true requires a supplied reachable node_id. If no suitable node is supported,
return needed=false with uncertainty; do not invent a checkpoint or teleport.
All route and STOP reasoning still uses only the map image and its measured trace,
the instruction and textual context. No action sequence is supplied.
'''

NODE_ROUTE_PROMPT='''Judge ROUTE CONSISTENCY at this visited decision node only.
The only visual input is the current map. Use the ORIGINAL instruction, actual
ordered retained position trace, current verified labels and semantic spatial
relations. Ignore red withdrawn/rollback paths as completed progress. Naming or
first seeing a target is not entering it. Infer substantial travel-direction changes
from sustained trajectory segments, not tiny steering or orientation changes.
Map X is right and Z down, in meters. N markers are visited return checkpoints;
their intersection outlines are GRID-FAST areas, not target names. Prior node verdicts
and text summaries are fallible. Read current position and trace before old context.
For off_route, choose the most recent supplied checkpoint whose current map semantics
support correct progress; recency only breaks ties after suitability. Controller
returns by reversing executed actions; no action history is sent to you. Do not
choose the current checkpoint if its retained prefix already includes the deviation.
If no candidate has sufficient support, abstain instead of inventing a safe node.
Do not judge STOP, propose STOP or evaluate stop timing in this node-only request.
Return exactly this JSON structure:
{"route_state":"normal|off_route|uncertain",
"position_review":{"stage":"...","completed":[],"completion_status":"...",
"location":"current mapped location","history_match":"...","reason":"map/trace evidence"},
"map_evidence":{"route":{"sufficient":false,"region_ids":[],"object_ids":[],
"route_steps":[],"relation":"concrete spatial evidence referencing supplied map IDs"}},
"recovery":{"needed":false,"node_id":null,"current_node_suitable":true,"reason":"..."}}
Evidence IDs/steps must exist in the supplied map. needed=true requires a supplied
node_id. Insufficient map evidence means no intervention. Never infer route progress
from an initialization description instead of current measured geometry and trace.
'''

STOP_REVIEW_MODES=('native_stop','correction_stop','node_correction_stop')


class DecisionNodes:
    def __init__(self):
        self.nodes={};self.pending=[];self.active=None;self.last=None;self.edges=set()
        self.gridfast=GridFastDetector();self.area_memory={}

    def checkpoint(self,episode,kind,branches=0,area=None,openings=None,polygon=None):
        p=episode.map.position;floor=episode.map.active
        match=next((n for n in self.nodes.values() if n['floor']==floor and n.get('retained',True) and
            ((area and len(area&self.area_memory.get(n['id'],set()))/max(1,min(len(area),len(self.area_memory.get(n['id'],set()))))>=.3)
             or np.linalg.norm((np.array(n['position'])-p)[[0,2]])<.8)),None)
        if match is None:
            nid='N%d'%(len(self.nodes)+1)
            match=dict(id=nid,position=p.tolist(),floor=floor,kind=kind,branches=branches,
                state=len(episode.history)-1,step=episode.steps,route_state='uncertain',retained=True)
            self.nodes[nid]=match
        else:
            nid=match['id'];match.update(position=p.tolist(),state=len(episode.history)-1,
                step=episode.steps,branches=max(branches,match['branches']))
        if self.last and self.last!=nid:self.edges.add((self.last,nid))
        if area:
            self.area_memory[nid]=set(area)
            match.update(intersection_area_m2=round(len(area)*episode.map.res**2,3),
                intersection_polygon_cells=polygon or [],opening_segments_cells=openings or [])
        self.last=self.active=nid
        episode.node_poses[nid]=episode.pose.pose.copy()
        episode.event('decision_node',node=copy.deepcopy(match))
        return nid

    def observe(self,episode):
        m=episode.map
        if not episode.pose.valid or m.transition:return None
        p=m.position;cell=m.cell(p);grid=m.grid();m.candidates(grid)
        if self.active:
            node=self.nodes[self.active]
            inside=cell in self.area_memory.get(self.active,set())
            if node['floor']==m.active and (inside or np.linalg.norm((p-np.array(node['position']))[[0,2]])<1.):return None
            self.active=None
        regions=self.gridfast.detect(grid,np.array(cell)-grid['origin'],m.res,episode.args,2*m.radius)
        for region in regions:
            area=region['cells']
            pending=next((r for r in self.pending if r['floor']==m.active and
                len(r['cells']&area)/max(1,min(len(area),len(r['cells'])))>=.3),None)
            if pending is None:
                pending=dict(cells=set(area),floor=m.active,votes=1,last=p.copy(),geometry=self.gridfast.cache_key);self.pending.append(pending)
            elif np.linalg.norm((pending['last']-p)[[0,2]])>=.15 or pending['geometry']!=self.gridfast.cache_key:
                pending['votes']+=1;pending.update(cells=set(area),last=p.copy(),geometry=self.gridfast.cache_key)
            if pending['votes']<2 or cell not in area:continue
            return self.checkpoint(episode,'GRID-FAST_intersection',region['branches'],
                area=area,openings=region['openings'],polygon=region['polygon_cells'])
        return None

_MetricMap=TopdownMap
class DecisionMap(_MetricMap):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.graph=None;self.semantic_snapshots={};self.semantic_mode=False
        self.split_footprints={}

    def semantic_reassociate(self,result,old,grid):
        for row in self.semantics.values():
            if row['floor']!=self.active:continue
            footprint=self.split_footprints.get(row['target_id'],row['cells'])
            children=[rid for rid,r in result.items() if len(footprint&r['cells'])*self.res**2>=.2]
            if len(children)>1 or row.get('association')=='pending_semantic_split':
                if footprint:self.split_footprints[row['target_id']]=set(footprint)
                row.update(cells=set(),association='pending_semantic_split',split_candidates=children)
                for rid in children:self.naming_counts[rid]=0
                continue
            anchors=row.get('evidence_cells',set())
            scores=sorted([(len(anchors&r['cells'])/max(1,len(anchors)),rid) for rid,r in result.items()],reverse=True)
            if scores and scores[0][0]>=.6 and (len(scores)==1 or scores[1][0]<.25):
                count=self.naming_counts.get(row['region_id'],0)
                row.update(region_id=scores[0][1],cells=set(result[scores[0][1]]['cells']),association='matched')
                if count:self.naming_counts[scores[0][1]]=1
            elif anchors:row.update(cells=set(),association='uncertain')

    def region_views(self,grid,step,views=None):
        result={rid:dict(rows) for rid,rows in super().region_views(grid,step,views).items()}
        if self.semantic_mode:
            for name,snapshot in self.semantic_snapshots.items():
                if snapshot['floor']!=self.active or views is not None and name not in views:continue
                for rid,r in grid['regions'].items():
                    pts=[v for v in snapshot['samples'] if (v[0],v[1]) in r['cells']]
                    cells={(v[0],v[1]) for v in pts}
                    if len(cells)<3:continue
                    uv=np.array([v[2:] for v in pts])
                    result.setdefault(rid,{})[name]=dict(cells=cells,uv=uv.tolist(),
                        image_box=[*uv.min(axis=0).tolist(),*uv.max(axis=0).tolist()],source_step=snapshot['step'])
        return result

    def metadata(self,grid):
        result=dict(super().metadata(grid))
        result['objects']=[{k:v for k,v in row.items() if k!='identity_evidence'} for row in result['objects']]
        if self.graph:
            result['decision_nodes']=[copy.deepcopy(r) for r in self.graph.nodes.values()]
            result['decision_edges']=[list(e) for e in sorted(self.graph.edges)]
            result['gridfast']=dict(self.graph.gridfast.info)
        return result

    def render(self,grid=None,*args,**kwargs):
        grid=self.grid() if grid is None else grid
        image=super().render(grid,*args,**kwargs).copy()
        if self.graph:
            draw=ImageDraw.Draw(image);scale=max(1,min(5,1500//max(grid['free'].shape)))
            for nid,n in self.graph.nodes.items():
                if n['floor']!=self.active:continue
                x,z=(np.array(self.cell(n['position']))-grid['origin']+.5)*scale
                if not (0<=x<grid['free'].shape[1]*scale and 0<=z<grid['free'].shape[0]*scale):continue
                color='navy' if n.get('retained',True) else 'red'
                draw.ellipse((x-4,z+66,x+4,z+74),outline=color,width=2)
                draw.text((x+5,z+70),nid,fill=color,stroke_fill='white',stroke_width=1)
        return image

    def save_final(self,folder,termination=''):
        super().save_final(folder,termination)
        path=Path(folder)/'final_map.json'
        if self.graph and path.exists():
            value=json.loads(path.read_text());value['decision_nodes']=list(self.graph.nodes.values())
            value['decision_edges']=[list(e) for e in sorted(self.graph.edges)]
            path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


_MapPlanner=Planner
class NodePlanner(_MapPlanner):
    def __init__(self,*args):
        super().__init__(*args);self.node_mode='node';self.node_candidates=[]
        self.last_naming_signature=None;self.last_naming_position=None;self.last_naming_image=None

    def ask(self,prompt,data,images=(),validator=None,**kwargs):
        if prompt==PANORAMA_PROMPT:
            return HistoryPlanner.ask(self,prompt,data,images,validator=validator,**kwargs)
        if prompt==REVIEW_PROMPT:
            data+='\nReachable rollback node candidates: '+json.dumps(self.node_candidates)
            data+='\nRequested check type: '+self.node_mode
            original=validator
            def node_validate(value):
                if self.node_mode not in STOP_REVIEW_MODES:
                    value['stop_proposal']=dict(decision='uncertain',timing='uncertain',reason='Not evaluated: node-only route check')
                    value.setdefault('map_evidence',{})['stop']=dict(sufficient=False,region_ids=[],object_ids=[],route_steps=[],relation='STOP not requested')
                value=original(value) if original else value
                timing=value['stop_proposal'].get('timing')
                if timing not in ('early','late','on_time','uncertain'):raise ValueError('Missing STOP timing')
                recovery=value.get('recovery')
                if not isinstance(recovery,dict) or type(recovery.get('needed')) is not bool or type(recovery.get('current_node_suitable')) is not bool:
                    raise ValueError('Missing recovery needed/current_node_suitable')
                ids={r['id'] for r in self.node_candidates}
                if recovery.get('node_id') is not None and recovery['node_id'] not in ids:raise ValueError('Select a reachable retained node ID')
                if recovery['needed'] and recovery.get('node_id') is None:raise ValueError('Recovery requires a supplied node ID')
                return value
            task=prompt+NODE_REVIEW_RULES if self.node_mode in STOP_REVIEW_MODES else NODE_ROUTE_PROMPT
            return super().ask(task,data,images,validator=node_validate,**kwargs)
        return super().ask(prompt,data,images,validator=validator,**kwargs)

    def orient(self,panorama):
        pose=self.pose.pose;objects=[]
        for oid,row in self.map.objects.items():
            if row['floor']!=self.map.active:continue
            local=pose[:3,:3].T@(np.array(row['center'])-pose[:3,3])
            objects.append(dict(object_id=oid,target_id=row['target_id'],name=row['label'],
                bearing_clockwise_deg=round(math.degrees(math.atan2(local[0],-local[2]))%360,1),
                center=row['center'],status=row.get('identity_status','candidate')))
        data='Targets (full heads/modifiers): '+json.dumps(self.targets,ensure_ascii=False)
        data+='\nProjected candidate locations/bearings: '+json.dumps(objects,ensure_ascii=False)
        def validate(value):
            sector=value.get('sector')
            if sector is not None and (type(sector) is not int or not 0<=sector<24):raise ValueError('sector must be 0..23 or null')
            description=value.get('initial_description')
            if not isinstance(description,dict) or any(not isinstance(description.get(k),str) for k in ('area','front','right','back','left')):
                raise ValueError('Include the descriptive initialization snapshot')
            for row in value.get('object_matches',[]):
                obj=self.map.objects.get(row.get('object_id'))
                if obj is None or obj['target_id']!=row.get('target_id') or row.get('confidence') not in ('confirmed','rejected','uncertain'):
                    raise ValueError('Object match must refer to a supplied candidate/target')
            return value
        answer=self.ask(PANORAMA_PROMPT,data,[('INITIAL PANORAMA; 24 clockwise 15-degree sectors',panorama),
            ('LATEST MAP with projected object candidates',self.map.render())],validator=validate)
        for row in answer.get('object_matches',[]):
            if row['confidence']=='rejected':self.map.objects.pop(row['object_id'],None)
            else:self.map.objects[row['object_id']].update(identity_status=row['confidence'],identity_evidence=row.get('evidence',''))
        self.map.version+=1
        self.initial_location=answer['initial_description']
        self.log(dict(phase='panorama_orientation',**answer))
        return answer.get('sector')

    def update_semantics(self,history,views=None,force=False):
        if not self.pose.valid or self.map.transition:return
        grid=self.map.grid();self.map.candidates(grid);step=history[-1].physical_step
        current=views or {'front':history[-1].image()}
        pending=[r for r in self.map.semantics.values() if r.get('association')=='pending_semantic_split']
        if pending:
            for name,s in self.map.semantic_snapshots.items():
                if s['floor']==self.map.active:current=dict(current,**{name:s['image']})
        self.map.semantic_mode=True
        try:
            allowed=self.pending_region_targets(None,grid,step,list(current),'naming')
            if not allowed:return
            support=self.map.region_views(grid,step,list(current))
            eligible=[rid for rid in support if not self.map.naming_counts.get(rid,0)]
            signature=(tuple(eligible),tuple(sorted(allowed)),tuple((r['target_id'],tuple(r.get('split_candidates',[]))) for r in pending))
            thumb=np.asarray(history[-1].image().resize((32,24)),float)
            changed=self.last_naming_image is None or np.abs(thumb-self.last_naming_image).mean()>15
            moved=self.last_naming_position is None or np.linalg.norm((self.map.position-self.last_naming_position)[[0,2]])>=.5
            if not force and signature==self.last_naming_signature and not (changed and moved):return
            self.last_naming_signature=signature;self.last_naming_image=thumb;self.last_naming_position=self.map.position.copy()
            data,images,_=self.scene(history,current,mapping=True)
            images=[(('HISTORICAL NAMING SNAPSHOT at physical step %d; projected against current region boundaries: '%self.map.semantic_snapshots[label.split('CURRENT ',1)[-1].split(' RGB',1)[0]]['step']+label)
                if label.startswith('CURRENT semantic_') else label,image) for label,image in images]
            data+='\nPending semantic splits: '+json.dumps([{k:v for k,v in row.items() if k not in ('cells','evidence_cells')}
                for row in pending],ensure_ascii=False)
            data+='\nAfter a geometric split, compare CURRENT and explicitly HISTORICAL observations, the full description and projected child regions. Each named child needs its own evidence. Adjacent ordinary children may share the same target when visual evidence shows one physical place. Use current map geometry for current progress.'
            data+='\nEligible targets: '+json.dumps(allowed,ensure_ascii=False)+'\nEligible region IDs: '+json.dumps(eligible)
            def validate(value):
                self.map.validate_room_labels(value.get('room_labels'),grid,allowed,step,list(current));return value
            try:
                answer=self.ask(REGION_PROMPT,data,images,validator=validate,
                    trace=lambda **r:self.log(dict(phase='region_naming_reply',**r)))
            except (ValueError,TypeError) as exc:
                self.log(dict(phase='region_naming_abstained',reason=str(exc)));return
            self.map.label_regions(answer['room_labels'],grid,allowed,step,list(current))
            for row in answer['room_labels']:
                view=row['view']
                if view in self.map.observed_views:
                    obs=self.map.observed_views[view]
                    name='semantic_'+row['target_id']+'_'+row['region_id']
                    self.map.semantic_snapshots[name]=dict(image=current[view].copy(),samples=list(obs['samples']),
                        step=step,floor=self.map.active)
                self.map.split_footprints.pop(row['target_id'],None)
            self.log(dict(phase='region_naming',naming_epoch=self.map.naming_epoch,**answer))
        finally:self.map.semantic_mode=False


_MapEpisode=Episode
class NodeEpisode(_MapEpisode):
    def __init__(self,*args):
        super().__init__(*args)
        self.graph=DecisionNodes();self.map.graph=self.graph;self.node_poses={}
        self.forward_check=0;self.last_action=None;self._last_review=None;self.last_recovery_step=-1
        self.stop_recovery_active=False

    def immediate_stop(self,review,native_proposal=False):
        if not native_proposal and not (self.stop_recovery_active and review.get('stop_checked',False)):
            return False
        return super().immediate_stop(review,native_proposal=native_proposal)

    def end_stop_recovery(self,reason):
        if self.stop_recovery_active:
            self.stop_recovery_active=False;self.forward_check=0;self.last_action=None
            self.event('stop_recovery_checks_disabled',reason=reason)

    def map_action(self):
        completed=self.point_completed
        action=super().map_action()
        if self.point_completed>completed:self.end_stop_recovery('correction_waypoint_reached')
        return action

    def panorama(self):
        views = self.scan(persist=False)
        panorama = self.panorama_image(views)
        panorama.save(self.folder/'initial_panorama.png')
        self.event('initial_panorama',sectors=24,degrees_per_sector=15,map_updated=True)
        return panorama,views

    def candidates_for_review(self):
        rows=[]
        for nid,node in self.graph.nodes.items():
            if not node.get('retained',True) or node['floor']!=self.map.active:continue
            if node['state']>=len(self.history):continue
            row=copy.deepcopy(node)
            row['return_method']='inverse_executed_actions'
            cell=self.map.cell(node['position']);grid=self.map.grid()
            row['current_region_ids']=[rid for rid,r in grid.get('regions',{}).items() if cell in r['cells']]
            rows.append(row)
        return sorted(rows,key=lambda r:r['step'],reverse=True)

    def check(self,mode,views=None):
        self.vlm.node_mode=mode;self.vlm.node_candidates=self.candidates_for_review()
        review=super().review_now(views,phase=mode)
        review['stop_checked']=mode in STOP_REVIEW_MODES
        if mode in ('node','initial','recovery','node_correction_stop') and self.graph.active:
            self.graph.nodes[self.graph.active]['route_state']=review['route_state']
        self._last_review=review
        if mode in STOP_REVIEW_MODES:self.forward_check=0
        return review

    def review_now(self,views=None,phase=None):
        return self.check(phase or 'node',views)

    def rollback_node(self,nid,reason):
        node=self.graph.nodes.get(nid)
        if node is None or not node.get('retained',True) or node['floor']!=self.map.active:return False
        if node['state']>=len(self.history):return False
        original=list(self.history);saved_index=node['state'];saved_step=node['step']
        pose=self.node_poses[nid];target=np.array(node['position'])
        self.map.archive_route(saved_step,reason);self.target=None
        self.event('node_rollback_start',node_id=nid,target_world=target.tolist(),method='inverse_executed_actions')
        limit=self.args.node_return_max_steps;returned=0;complete=True
        for a,b in reversed(list(zip(original[saved_index:],original[saved_index+1:]))):
            if self.done or not self.pose.valid or returned>=limit:complete=False;break
            action={1:4,4:1,2:3,3:2}.get(a.action)
            if action is None:complete=False;break
            pa,pb=self.map_poses.get(a.physical_step),self.map_poses.get(b.physical_step)
            if a.action in (1,4) and pa is not None and pb is not None and np.linalg.norm(pa[:3,3]-pb[:3,3])<1e-5:continue
            before=dict(step=self.steps,floor=self.map.active,position=self.map.position.tolist())
            if not self.move(action,'rollback',retain=False,controller=True):complete=False;break
            returned+=1
            self.map.record_rollback(before,dict(step=self.steps,floor=self.map.active,position=self.map.position.tolist()))
        error=float(np.linalg.norm(self.pose.pose[:3,3]-pose[:3,3]))
        relative=self.pose.pose[:3,:3].T@pose[:3,:3]
        angle=float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
        success=complete and self.pose.valid and error<=self.args.rollback_tolerance and angle<=self.args.rollback_angle_tolerance
        if not success:
            self.termination='node_return_unavailable'
            self.event('node_rollback_failed',node_id=nid,position_error_m=error,angle_error_deg=angle,
                reason='Inverse history incomplete, measured pose mismatch or action budget exhausted')
            return False
        self.history=original[:saved_index]+[self.current]
        self.current.action=None;self.current.query=True;self.current.phase='rollback'
        self.map.truncate_route(saved_step)
        if self.map.route:self.map.route.pop()
        self.map.set_pose(self.pose.pose[:3,3],-self.pose.pose[:3,2],self.steps,True)
        self.vlm.summaries=[r for r in self.vlm.summaries if r['end']<saved_index]
        self.vlm.position_reviews=[r for r in self.vlm.position_reviews if r['observed_at']<saved_step]
        for other in self.graph.nodes.values():
            if other['state']>saved_index:other['retained']=False
        node.update(step=self.steps,state=len(self.history)-1,position=self.map.position.tolist())
        self.graph.active=self.graph.last=nid;self.blocked=0;self.forward_check=0;self.last_action=None
        self.policy.restore(self.history,self.vlm.instruction)
        self.event('node_rollback_end',node_id=nid,geometry_preserved=True)
        return True

    def recover_map(self,reason,rollback=True):
        if self.done:return
        review=self._last_review or {};recovery=review.get('recovery',{})
        nid=recovery.get('node_id') if recovery.get('needed') else None
        if rollback and nid is None:
            self.event('recovery_abstained',reason='VLM did not support a reachable semantic rollback node')
            self.target=None;self.policy.restore(self.history,self.vlm.instruction);return
        self.corrections+=1;self.deviations+=int(reason=='off_route')
        self.blockage_recoveries+=int(reason=='blocked_motion')
        if rollback and not self.rollback_node(nid,reason):return
        self.target=None;self.map.reset_region_naming()
        views=self.scan()
        description=self.call('correction_initial_location',lambda:self.locate_panorama(views))
        self.vlm.initialization_description=dict(physical_step=self.steps,state=len(self.history)-1,
            world_position=self.map.position.tolist(),description=description)
        self.vlm.update_semantics(self.history,views,force=True)
        review=self.check('node',views) if rollback else (self._last_review or {})
        if self.done:return
        if review.get('route_state','uncertain')=='uncertain' or self.map.transition:
            self.policy.restore(self.history,self.vlm.instruction);return
        target=self.call('map_waypoint',lambda:self.vlm.waypoint(self.history,views,review))
        if target is not None and self.map.plan(target):
            self.target=target;self.correction_steps=0;self.point_runs+=1
            self.event('correction_start',target_world=target.tolist(),floor=self.map.active)
        else:self.policy.restore(self.history,self.vlm.instruction)

    def initialize_checkpoint(self):
        grid=self.map.grid();self.map.candidates(grid)
        start_area=next((r for r in self.graph.gridfast.detect(grid,
            np.array(self.map.cell(self.map.position))-grid['origin'],self.map.res,self.args,2*self.map.radius)
            if self.map.cell(self.map.position) in r['cells']),None)
        self.graph.checkpoint(self,'start',branches=start_area['branches'] if start_area else 0,
            area=start_area['cells'] if start_area else None,openings=start_area['openings'] if start_area else None,
            polygon=start_area['polygon_cells'] if start_area else None)

    def _run(self):
        self.event('episode_start',instruction=self.vlm.instruction,control='panorama_decision_nodes')
        self.video.append_data(self.current.observations['rgb']);self.vlm.extract()
        raw=getattr(self.current,'raw_observations',self.current.observations);self.update_pose(raw)
        if self.pose.pose is None:raise RuntimeError('Cannot initialize map localization: '+self.pose.reason)
        panorama,views=self.panorama()
        sector=self.call('initial_heading',lambda:self.vlm.orient(panorama))
        self.vlm.update_semantics(self.history,views,force=True)
        turns=0 if sector is None else sector if sector<=12 else sector-24
        for _ in range(abs(turns)):
            if not self.move(3 if turns>0 else 2,'initial_alignment',controller=True):break
        # Replay actual 15-degree alignment actions into StreamVLN's own execution context.
        self.policy.restore(self.history,self.vlm.instruction)
        self.vlm.update_semantics(self.history,force=True)
        self.initialize_checkpoint()
        self.check('initial')
        self.map.save(self.folder)
        idle=0
        while not self.done:
            correcting=self.target is not None
            action=self.map_action() if correcting else self.policy.predict(self.current)
            if action is None:
                if self.done:break
                idle+=1
                if idle>=3:self.termination='correction_no_progress'
                continue
            idle=0
            if action==0:
                review=self.check('native_stop')
                if self.immediate_stop(review,native_proposal=True) or self.done:break
                self.stop_rejections+=1;self.native_stop_retries+=1
                if not self.stop_recovery_active:
                    self.event('stop_recovery_checks_enabled',reason='Native STOP rejected; enable four-forward and turn-end checks during correction')
                self.stop_recovery_active=True;self.forward_check=0;self.last_action=None
                if self.native_stop_retries>2:self.termination='native_stop_unresolved';break
                late=review.get('stop_proposal',{}).get('timing')=='late'
                recover=review.get('recovery',{}).get('needed',False) or review['route_state']=='off_route'
                self.recover_map('late_stop' if late else 'early_stop',rollback=late or recover)
                continue
            if self.stop_recovery_active and self.last_action in (2,3) and action not in (2,3):
                review=self.check('correction_stop')
                self.last_action=None
                if self.immediate_stop(review) or self.done:break
                if review.get('stop_proposal',{}).get('timing')=='late' and review.get('recovery',{}).get('needed'):
                    self.recover_map('late_stop',rollback=True);continue
            if not self.move(action,'correction' if correcting else 'normal',controller=correcting):break
            self.native_stop_retries=0
            self.forward_check=self.forward_check+1 if action==1 else 0
            self.last_action=action
            if not correcting:self.policy.advance()
            else:
                self.correction_steps+=1
                if self.correction_steps>=self.args.correction_max_steps:
                    self.target=None;self.policy.restore(self.history,self.vlm.instruction)
            self.vlm.update_semantics(self.history)
            node=self.graph.observe(self)
            if node:
                combined=self.stop_recovery_active and self.forward_check>=4
                review=self.check('node_correction_stop' if combined else 'node')
                if combined and (self.immediate_stop(review) or self.done):break
                if review['route_state']=='off_route':self.recover_map('off_route',rollback=True)
                elif combined and review.get('stop_proposal',{}).get('timing')=='late' and review.get('recovery',{}).get('needed'):
                    self.recover_map('late_stop',rollback=True)
                elif review['route_state']=='normal' and self.target is None:
                    self.end_stop_recovery('consistent_decision_node_handoff')
            if self.stop_recovery_active and self.forward_check>=4 and not self.done:
                review=self.check('correction_stop')
                if self.immediate_stop(review) or self.done:break
                if review.get('stop_proposal',{}).get('timing')=='late' and review.get('recovery',{}).get('needed'):
                    self.recover_map('late_stop',rollback=True)
            if self.blocked>=4 and not self.done:
                self.recover_map('blocked_motion',rollback=False)
            self.compress();self.map.save(self.folder)
        self.termination=self.termination or 'max_steps';self.map.save(self.folder)
        self.event('episode_end',reason=self.termination)
        return dict(steps=self.steps,effective_steps=len(self.history)-1,num_rewrites=0,corrections=self.corrections,
            deviations=self.deviations,blockage_recoveries=self.blockage_recoveries,stop_returns=0,
            stop_rejections=self.stop_rejections,max_task_depth=1,point_runs=self.point_runs,
            point_completed=self.point_completed,correction_images=[],final_instruction=self.vlm.instruction,
            termination_reason=self.termination,summaries=self.vlm.summaries,pose_source=self.pose.selected,
            floors=self.map.floors,decision_nodes=list(self.graph.nodes.values()),
            intervened=bool(self.corrections or self.stop_rejections or self.termination=='governor_stop'))


_MapArguments=arguments
def arguments():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--node_roi_radius',type=float,default=8.)
    p.add_argument('--gridfast_directions',type=int,default=4)
    p.add_argument('--gridfast_unknown_filter',type=float,default=.10)
    p.add_argument('--gridfast_object_perimeter',type=float,default=.50)
    p.add_argument('--gridfast_max_gaps',type=int,default=512)
    p.add_argument('--gridfast_opt_steps',type=int,default=24)
    p.add_argument('--node_return_max_steps',type=int,default=80)
    extra,remaining=p.parse_known_args();original=sys.argv
    try:
        sys.argv=[original[0]]+remaining;args=_MapArguments()
    finally:sys.argv=original
    if min(extra.node_roi_radius,extra.node_return_max_steps,extra.gridfast_directions,
           extra.gridfast_max_gaps,extra.gridfast_opt_steps)<=0 or min(extra.gridfast_unknown_filter,extra.gridfast_object_perimeter)<0:
        p.error('node scales/budget must be positive; GRID-FAST filter tolerances must be nonnegative')
    vars(args).update(vars(extra))
    if args.vlm_backend=='api' and args.vlm_api_config is None:
        private=Path.home()/'.config/streamvln/governor_api.json'
        if private.is_file():args.vlm_api_config=str(private)
    return args


_OriginalLocalVLM=LocalVLM
def main():
    global Planner,Episode,TopdownMap,LocalVLM
    if sys.argv[1:2]==['--vlm-worker']:return vlm_worker(sys.argv[2:])
    if sys.argv[1:2]==['--seg-worker']:return segmentation_worker_main(sys.argv[2:])
    if '--help' not in sys.argv and '-h' not in sys.argv:Rose2Partition(.05).libraries()
    Planner,Episode,TopdownMap=NodePlanner,NodeEpisode,DecisionMap
    LocalVLM=lambda args:APIVLM(args) if args.vlm_backend=='api' else _OriginalLocalVLM(args)
    Frame.capture=classmethod(capture)
    return base_main()


if __name__=='__main__':
    raise SystemExit(main())
