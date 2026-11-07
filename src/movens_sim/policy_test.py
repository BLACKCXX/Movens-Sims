"""Minimal, single-episode compatibility adapter for the frozen policy test.

No estimation, optimization, training, or policy fallback is performed here.
The original runtime and checkpoints are read-only inputs. Local frozen copies
are patched only at explicitly checked compatibility seams, recorded in JSON.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def snapshot(model, data):
    result = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, result, mujoco.mjtState.mjSTATE_INTEGRATION)
    return result


def video_schedule(end, duration, fps=30, control_fps=50):
    """Keep real-video indices separate from the policy's control grid."""
    return [dict(sim_time_s=i/fps,
                 source_frame_index=i if i/fps <= end else None,
                 control_reference_index_50hz=round(i/fps*control_fps),
                 phase="trajectory" if i/fps <= end else "tail_verification")
            for i in range(1, int(duration*fps)+1)]


def migrate_environment(source, source_data, target, target_data):
    """Map each named environment joint, including free objects and flex nodes."""
    widths = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
    names = []
    for j in range(source.njnt):
        name = source.joint(j).name
        if not name:
            raise ValueError("Unnamed environment joint cannot be safely migrated")
        k = mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_JOINT, name)
        if k < 0 or source.jnt_type[j] != target.jnt_type[k]:
            raise ValueError("Joint missing or changed type: " + name)
        nq, nv = widths[int(source.jnt_type[j])]
        a, b = source.jnt_qposadr[j], target.jnt_qposadr[k]
        target_data.qpos[b:b+nq] = source_data.qpos[a:a+nq]
        a, b = source.jnt_dofadr[j], target.jnt_dofadr[k]
        target_data.qvel[b:b+nv] = source_data.qvel[a:a+nv]
        names.append(name)
    target_data.time = source_data.time
    return names


def geom_bounds(model, data, gid):
    """True compiled mesh bounds, with original box behavior for other shapes."""
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        points = model.mesh_vert[start:start+count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        return points.min(axis=0), points.max(axis=0)
    half = abs(data.geom_xmat[gid].reshape(3, 3)) @ model.geom_size[gid]
    return data.geom_xpos[gid] - half, data.geom_xpos[gid] + half


def merge_robot(environment_xml, robot_xml, destination, prior):
    """Copy ONLY the robot; prefix assets so shared object meshes cannot change."""
    env = ET.parse(environment_xml)
    source = ET.parse(robot_xml)
    root, robot_root = env.getroot(), source.getroot()
    pelvis = copy.deepcopy(robot_root.find("./worldbody/body[@name='pelvis']"))
    if pelvis is None or root.find(".//body[@name='pelvis']") is not None:
        raise ValueError("Expected robot-free environment and one source G1")
    pelvis.set("pos", " ".join(map(str, prior["root"]["xyz_m"])))
    pelvis.set("quat", " ".join(map(str, prior["root"]["quaternion_wxyz"])))
    for node in list(pelvis.iter()):
        for child in list(node):
            if child.tag == "camera":
                node.remove(child)
    # Motor defaults are unchanged between imported and reconstructed XML.
    for default in robot_root.findall("./default/default"):
        name = default.get("class", "")
        if not name.endswith("_motor"):
            continue
        existing = root.find(f"./default/default[@class='{name}']")
        if existing is None or ET.tostring(existing) != ET.tostring(default):
            raise ValueError("Robot motor default differs: " + name)
    assets = {x.get("name"): x for x in robot_root.find("asset")}
    needed = set()
    refs = ("mesh", "material", "texture", "hfield")
    def collect(node):
        for element in node.iter():
            for key in refs:
                value = element.get(key)
                if value and value in assets and value not in needed:
                    needed.add(value)
                    collect(assets[value])
    collect(pelvis)
    prefix = "policy_g1_"
    def rewrite(node):
        for element in node.iter():
            for key in refs:
                if element.get(key) in needed:
                    element.set(key, prefix + element.get(key))
    rewrite(pelvis)
    asset_manifest = []
    for name in sorted(needed):
        item = copy.deepcopy(assets[name])
        item.set("name", prefix + name)
        rewrite(item)
        if item.get("file"):
            source_path = (robot_xml.parent / item.get("file")).resolve()
            digest = sha(source_path)
            rel = Path("assets/robot") / (digest[:16] + "_" + source_path.name)
            target = destination.parent / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            item.set("file", rel.as_posix())
            asset_manifest.append(dict(source=str(source_path), portable=str(rel), sha256=digest))
        root.find("asset").append(item)
    root.find("worldbody").append(pelvis)
    for tag in ("actuator", "sensor", "contact", "tendon", "equality"):
        section = robot_root.find(tag)
        if section is None:
            continue
        # This imported source has only robot excludes/sensors/actuators.
        if tag in ("tendon", "equality") and len(section):
            raise ValueError("Additional reference migration required for " + tag)
        target = root.find(tag)
        if target is None:
            target = ET.SubElement(root, tag)
        for node in section:
            target.append(copy.deepcopy(node))
    env.write(destination, encoding="unicode")
    return asset_manifest


def render_check(model, data, hide_robot=False):
    # A disposable model is used; no geometry or state in the physics copy changes.
    option = mujoco.MjvOption()
    option.flexgroup[5] = 0
    debug = ((model.geom_group == 0) & ((model.geom_contype | model.geom_conaffinity) != 0)
             & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE))
    old_group = model.geom_group.copy()
    model.geom_group[debug] = 3
    option.geomgroup[3] = 0
    if hide_robot:
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        bodies = {pelvis}
        for b in range(pelvis + 1, model.nbody):
            if int(model.body_parentid[b]) in bodies:
                bodies.add(b)
        model.geom_group[np.isin(model.geom_bodyid, list(bodies))] = 5
        option.geomgroup[5] = 0
    model.vis.quality.offsamples = 0
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer.update_scene(data, camera="approx_ego_camera", scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            return renderer.render().copy()
    finally:
        model.geom_group[:] = old_group


def adapt_runtime(frozen):
    """Checked, local-copy-only changes; keep inference/thresholds untouched."""
    changes = []
    def change(name, old, new):
        path = frozen / name
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError(f"Frozen runtime seam changed: {name}: {old[:70]}")
        path.write_text(text.replace(old, new))
        compile(path.read_text(), str(path), "exec")
        changes.append(dict(file=name, old=old, new=new))
    name = "simulation/color_generalization.py"
    change(name, "self.model.checkpoint['step']!=50000", "self.model.checkpoint['step']!=self.cfg['expected_checkpoint_step']")
    change(name, "mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_flex(m,d);mujoco.mj_camlight(m,d)", "mujoco.mj_forward(m,d)  # recompute contacts for the migrated initial/resumed state")
    change(name, "rec.frame(-settle,-1,0);commit()", "rec.sample(0.0);batch_quality(rec,evalcfg['evaluation']);rec.frame(0.0,0,2);commit()")
    change(name, "source='live_recolored_scene' if group=='B' else 'live_original_scene'", "source='live_reconstructed_scene'")
    change(name, "group+(' 原色' if group=='A' else ' 桌面/地板换色')", "'重建场景 / 最佳48500步模型'")
    change(name, "frames_include_reset_and_settle=True", "frames_include_reset_and_settle=False")
    change("simulation/apple_batch_evaluate.py", "g=placed_geometry(q[ix],geo,e)", "g=placed_geometry(q[ix],geo,e,plate_qpos=a['plate_qpos'][ix])")
    return changes


def prepare(config_path):
    import torch
    from PIL import Image
    cfg = read(config_path)
    root = Path(cfg["run_root"])
    receipt = root / "prepared.json"
    if receipt.exists():
        prior = read(receipt)
        if prior["requested_config_sha256"] != sha(config_path):
            raise ValueError("Existing task config differs; preserve it rather than overwrite")
        for name, value in prior["bound_inputs"].items():
            if sha(name) != value:
                raise ValueError("Input changed: " + name)
        for name, value in prior["generated_files"].items():
            if sha(root/name) != value:
                raise ValueError("Prepared file changed: " + name)
        return cfg
    if root.exists() and any(root.iterdir()):
        raise ValueError("Uncommitted preparation exists; preserve it before retry")
    root.mkdir(parents=True)
    write(root / "requested_config.json", cfg)
    reconstruction = Path(cfg["reconstruction"])
    source = Path(cfg["robot_source"])
    original = Path(cfg["original_candidate"])
    training = Path(cfg["training_run"])
    runtime = Path(cfg["test_runtime"])
    best = read(training / "formal/best_val_metrics.json")
    vals = [json.loads(line) for line in (training / "formal/metrics.jsonl").read_text().splitlines()]
    vals = [x for x in vals if "validation_selection_score" in x]
    selected = min(vals, key=lambda x: x["validation_selection_score"])
    if best["step"] != cfg["expected_checkpoint_step"] or selected["step"] != best["step"]:
        raise ValueError("Configured checkpoint does not match saved validation minimum")
    checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["step"] != best["step"]:
        raise ValueError("Checkpoint metadata differs from best marker")
    cfg["hand_action_policy"] = checkpoint["config"]["hand_action_policy"]
    cfg["checkpoint_sha256"] = sha(cfg["checkpoint"])
    write(root / "checkpoint_selection.json", dict(
        checkpoint=cfg["checkpoint"], sha256=cfg["checkpoint_sha256"], step=best["step"],
        saved_validation_records=len(vals), selection=best,
        source_checkpoint_step=checkpoint.get("source_checkpoint_step"),
        step_semantics="New optimizer updates in the 30000+10000+10000 wrist fine-tuning run; not the inherited source step",
        base_checkpoint=cfg["base_checkpoint"], base_sha256=checkpoint.get("base_sha256"),
        closed_loop_success_previously_measured=checkpoint.get("closed_loop_success_measured")))
    for key in ("config", "body_normalization", "hand_normalization"):
        write(root / ("checkpoint_" + key + ".json"), checkpoint[key])
    del checkpoint
    frozen = root / "frozen_code"
    shutil.copytree(runtime / "frozen_code", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("movens_interact/model.py", "movens_interact/hand.py", "configs/body.json"):
        shutil.copy2(training / "frozen_code" / name, frozen / name)
    changes = adapt_runtime(frozen)
    write(root / "runtime_adaptations.json", changes)
    prior = read(reconstruction / "camera_prior.json")
    case = root / "test"
    scene = case / "scene"
    shutil.copytree(reconstruction / "physics/portable", scene)
    robot_assets = merge_robot(scene / "scene.xml", source / "scene.xml", scene / "scene.xml", prior)
    robot = read(source / "robot.json")
    names = robot["body_joint_names"] + robot["hand_joint_names"]
    if set(names) != set(prior["observed_joints_rad"]) or len(names) != 43:
        raise ValueError("Frame-0 observed joint map is incomplete")
    q = [prior["observed_joints_rad"][n] for n in names]
    reset = robot["reset_protocol"]
    reset.update(joint_names=names, qpos43=q, ctrl43=q, settle_steps=0)
    reset.pop("apple_initial_pose", None)
    write(scene / "robot.json", robot)
    env = mujoco.MjModel.from_xml_path(str(reconstruction / "physics/portable/scene.xml"))
    ed = mujoco.MjData(env)
    with np.load(reconstruction / "physics/portable/scene.state.npz", allow_pickle=False) as state:
        ed.qpos[:] = state["qpos"]; ed.qvel[:] = state["qvel"]; ed.time = float(state["time"])
    ed.time = 0.0
    mujoco.mj_forward(env, ed)
    model = mujoco.MjModel.from_xml_path(str(scene / "scene.xml"))
    data = mujoco.MjData(model)
    migrated = migrate_environment(env, ed, model, data)
    for name, value in zip(names, q):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        data.ctrl[model.actuator(name).id] = value
    mujoco.mj_forward(model, data)
    initializer = root / "initialization"
    initializer.mkdir()
    np.save(initializer / "initial_state.npy", snapshot(model, data), allow_pickle=False)
    np.savez(scene / "scene.state.npz", qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time)
    protocol = read(original / "run/report.json")
    original_settle = protocol["settle_duration_s"]
    protocol["settle_duration_s"] = 0.0
    protocol["initialization_source"] = "Reconstructed environment state + frame-0 observed robot joints; no additional settle"
    write(initializer / "report.json", protocol)
    duration = protocol["original_trajectory_end_s"] + protocol["tail_verification_seconds"]
    frames = video_schedule(protocol["original_trajectory_end_s"], duration)
    write(initializer / "frame_map.json", dict(diagnostic_frames=frames, fps=30, source_index_kind="video_frame_30fps"))
    estimate = read(original.parent / "estimate.json")
    estimate["apple_xyz_m"] = data.xpos[model.body("apple_root").id].tolist()
    estimate["plate_xyz_m"] = data.xpos[model.body("plate_root").id].tolist()
    estimate["source_estimate_sha256"] = sha(original.parent / "estimate.json")
    estimate["policy_test_note"] = "Object/camera scene is reconstructed; event hints never control policy execution"
    write(root / "estimate.json", estimate)
    for name in ("softpad.json", "batch_construction.json"):
        shutil.copy2(source / name, scene / name)
    info = read(source / "scene.json")
    info.update(scene="scene.xml", scene_sha256=sha(scene/"scene.xml"), robot_config="robot.json",
                apple_xyz=estimate["apple_xyz_m"], plate_xyz=estimate["plate_xyz_m"],
                camera=read(reconstruction/"camera_actual.json"),
                initialization="scene.state.npz / name-mapped environment + observed frame-0 robot",
                source_reconstruction=str(reconstruction), source_scene_sha256=sha(reconstruction/"physics/portable/scene.xml"))
    info["timing"]["physics_fps"] = 1/model.opt.timestep
    write(scene/"scene.json", info)
    # Verify every original body and compiled geom, not merely apple centers.
    errors = {}
    body_map = {0: 0}
    for j in range(1, env.nbody):
        name = env.body(j).name
        if name:
            k = model.body(name).id
        else:
            # MuJoCo allows anonymous static cloth/display bodies. Preserve
            # their ordinal among anonymous siblings under the mapped parent.
            parent = int(env.body_parentid[j])
            siblings = [i for i in range(1,j+1) if env.body_parentid[i] == parent and not env.body(i).name]
            targets = [i for i in range(1,model.nbody) if model.body_parentid[i] == body_map[parent] and not model.body(i).name]
            k = targets[len(siblings)-1]
        body_map[j] = k
        errors["body:"+(name or str(j))] = max(float(abs(ed.xpos[j]-data.xpos[k]).max()), float(abs(ed.xmat[j]-data.xmat[k]).max()))
    for g in range(env.ngeom):
        name = env.geom(g).name
        if not name:
            continue
        k = model.geom(name).id
        errors["geom:"+name] = max(float(abs(x-y).max()) for x,y in zip(geom_bounds(env,ed,g),geom_bounds(model,data,k)))
        if model.geom_type[k] != env.geom_type[g] or model.geom_contype[k] != env.geom_contype[g] or model.geom_conaffinity[k] != env.geom_conaffinity[g]:
            raise ValueError("Environment collision changed: " + name)
    camera = "approx_ego_camera"
    a, b = env.camera(camera).id, model.camera(camera).id
    camera_error = max(float(abs(ed.cam_xpos[a]-data.cam_xpos[b]).max()), float(abs(ed.cam_xmat[a]-data.cam_xmat[b]).max()), float(abs(env.cam_fovy[a]-model.cam_fovy[b])))
    flex_error = float(abs(ed.flexvert_xpos-data.flexvert_xpos).max())
    reload = mujoco.MjModel.from_xml_path(str(scene/"scene.xml")); rd = mujoco.MjData(reload)
    mujoco.mj_setState(reload, rd, np.load(initializer/"initial_state.npy", allow_pickle=False), mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(reload, rd)
    reload_error = max(float(abs(data.qpos-rd.qpos).max()), float(abs(data.qvel-rd.qvel).max()), float(abs(data.ctrl-rd.ctrl).max()))
    original_image = render_check(env, ed)
    merged_hidden = render_check(model, data, hide_robot=True)
    actual_image = render_check(model, data)
    reloaded_image = render_check(reload, rd)
    Image.fromarray(original_image).save(root/"environment_before_robot.png")
    Image.fromarray(merged_hidden).save(root/"environment_after_robot_hidden.png")
    Image.fromarray(actual_image).save(root/"initial_ego.png")
    rgb_error = int(abs(original_image.astype(int)-merged_hidden.astype(int)).max())
    rgb_reload = int(abs(actual_image.astype(int)-reloaded_image.astype(int)).max())
    passed = max(errors.values()) < 1e-9 and camera_error < 1e-10 and flex_error < 1e-10 and reload_error == 0 and rgb_error == 0 and rgb_reload == 0
    write(root/"scene_adaptation_check.json", dict(passed=passed, environment_joint_count=len(migrated), robot_joints=names,
        maximum_environment_error=max(errors.values()), camera_error=camera_error, flex_vertex_error_m=flex_error,
        reload_state_max_error=reload_error, environment_rgb_max_error=rgb_error, reload_rgb_max_error=rgb_reload,
        removed_old_robot_camera=True, world_camera_fixed=True, camera_fovy_deg=float(model.cam_fovy[b]),
        camera_world_position_m=data.cam_xpos[b].tolist(), camera_world_rotation_matrix=data.cam_xmat[b].reshape(3,3).tolist(),
        original_settle_skipped_s=original_settle, additional_settle_s=0, robot_velocity_assumption="zero; frame-0 measured velocities unavailable",
        initial_contacts=int(data.ncon), robot_assets=robot_assets,
        reconstructed_quality=read(reconstruction/"result.json")["physical_passed"],
        episode_split="external_to_saved_250_episode_split"))
    if not passed:
        raise ValueError("Scene adaptation or reload verification failed; see scene_adaptation_check.json")
    cfg["episodes"] = [dict(episode_index=cfg["episode"], source_dir=str(initializer))]
    cfg["gpu_uuid"] = "selected_cuda_visible_device_" + str(cfg["gpu"])
    cfg["egl_device_id"] = cfg["gpu"]
    write(root/"config.json", cfg)
    shutil.copy2(runtime/"evaluation_config.json", root/"evaluation_config.json")
    shutil.copy2(config_path, root/"input_config.json")
    (root/"adapter_code").mkdir()
    shutil.copy2(Path(__file__), root/"adapter_code/policy_test.py")
    shutil.copy2(PROJECT/"scripts/test_reconstructed_policy.py", root/"adapter_code/test_reconstructed_policy.py")
    paths = [reconstruction/"physics/portable/scene.xml", reconstruction/"physics/portable/scene.state.npz",
             reconstruction/"parameters_final.json", reconstruction/"camera_prior.json", source/"scene.xml",
             Path(cfg["checkpoint"]), training/"formal/best_val_metrics.json", training/"formal/metrics.jsonl"]
    bound = {str(p): sha(p) for p in paths}
    generated = {str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file()}
    write(receipt, dict(requested_config_sha256=sha(config_path), bound_inputs=bound, generated_files=generated,
                        source_project_read_only=True, status="ready_for_one_closed_loop_test"))
    return cfg


def install_adapters(root):
    """Reuse existing per-case runtime; all overrides are confined to this process."""
    sys.path.insert(0, str(root/"frozen_code"))
    import simulation.color_generalization as runtime
    import simulation.apple_batch_evaluate as evaluator
    from simulation.softpad import TABLE
    base_guard = runtime.SceneGuard
    class MeshGuard(base_guard):
        def __init__(self, sim, policy, state=None):
            super().__init__(sim, policy, state)
            lo, hi = geom_bounds(sim.model, sim.data, sim.model.geom(TABLE).id)
            self.xy, self.half, self.top = (lo[:2]+hi[:2])/2, (hi[:2]-lo[:2])/2, float(hi[2])
    runtime.SceneGuard = MeshGuard
    recorder = runtime.BatchRecorder
    class MovingPlateRecorder(recorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            m = self.sim.model
            joints = np.flatnonzero((m.jnt_bodyid == self.sim.plate_id) & (m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
            if len(joints) != 1:
                raise ValueError("Expected exactly one free plate joint")
            self.plate_qpos_address = int(m.jnt_qposadr[joints[0]])
            self.plate_qvel_address = int(m.jnt_dofadr[joints[0]])
        def sample(self, t):
            result = super().sample(t)
            d = self.sim.data
            a,b = self.plate_qpos_address,self.plate_qvel_address
            self.metrics[-1]["plate_qpos"] = d.qpos[a:a+7].copy()
            self.metrics[-1]["plate_qvel"] = d.qvel[b:b+6].copy()
            return result
    runtime.BatchRecorder = MovingPlateRecorder
    original_geometry = evaluator.geometry
    def geometry(model):
        geo = original_geometry(model)
        data = mujoco.MjData(model); mujoco.mj_forward(model,data)
        lo,hi = geom_bounds(model,data,model.geom(TABLE).id)
        geo.update(table_center=(lo+hi)/2,table_half=(hi-lo)/2)
        return geo
    evaluator.geometry = geometry
    def placed_geometry(q, geo, e, plate_qpos=None):
        if plate_qpos is None:
            raise ValueError("Dynamic reconstructed plate requires recorded pose per timestep")
        output=[]
        for first in range(0,len(q),32):
            qq=q[first:first+32];pq=plate_qpos[first:first+32]
            points=np.einsum('tij,vj->tvi',evaluator.rotations(qq[:,3:]),geo['apple_vertices'])+qq[:,None,:3]
            local=np.einsum('tvi,tij->tvj',points-pq[:,None,:3],evaluator.rotations(pq[:,3:]))
            margin=evaluator.hull_margin(local,geo['plate_hull']).min(axis=1)
            surface=evaluator.plate_surface_z(local.reshape(-1,3),geo['plate_triangles'],e['plate_partition_seam_tolerance_m']).reshape(local.shape[:2])
            gap=np.min(local[:,:,2]-surface,axis=1);covered=np.isfinite(surface).all(axis=1)
            bottom=local[:,:,2].min(axis=1)
            passed=covered&(margin>=e['plate_footprint_margin_m'])&(gap>=-e['plate_support_height_tolerance_m'])&(gap<=e['plate_support_height_tolerance_m'])
            passed &= bottom>=geo['plate_vertices'][:,2].min()-e['plate_support_height_tolerance_m']
            output.extend(zip(passed.tolist(),margin.tolist(),gap.tolist(),bottom.tolist()))
        return np.array(output)
    evaluator.placed_geometry = placed_geometry
    return runtime


def summarize(root, wall):
    case=root/"test";report=read(case/"run/report.json");evaluation=read(case/"evaluation.json")
    events=[json.loads(x) for x in (case/"run/predictions.jsonl").read_text().splitlines()] if (case/"run/predictions.jsonl").exists() else []
    with np.load(case/"run/metrics.npz",allow_pickle=False) as z:
        q=z["apple_qpos"];p=z["plate_qpos"];t=z["t"]
        motion=dict(apple_final_displacement_m=float(np.linalg.norm(q[-1,:3]-q[0,:3])),
                    apple_max_xy_displacement_m=float(np.linalg.norm(q[:,:2]-q[0,:2],axis=1).max()),
                    apple_final_world_xyz_m=q[-1,:3].tolist(),plate_final_displacement_m=float(np.linalg.norm(p[-1,:3]-p[0,:3])),
                    recorded_initial_snapshots=int(np.sum(t==0)),native_physics_steps=int(np.sum(t>0)))
    selection=read(root/"checkpoint_selection.json")
    result=dict(status="complete" if report["tail_completed"] else "stopped", episode=0,
        source_xml=str(Path(read(root/"config.json")["reconstruction"])/"physics/portable/scene.xml"),
        tested_xml=str(case/"scene/scene.xml"), checkpoint=selection["checkpoint"], checkpoint_step=selection["step"],
        closed_loop_predictions=len(events),last_sim_time_s=report["last_sim_time_s"],stop_reason=report["stop_reason"],
        task_success=evaluation["pick_and_place_success"],physics_valid=evaluation["physics_valid"],
        accepted=evaluation["accepted"],render_valid=evaluation["render_valid"],wall_seconds=wall,
        motion=motion,scene_optimization_performed=False,training_performed=False,
        source_scene_quality_passed=False,split="external_to_saved_250_episode_split",
        videos={n:str(case/"videos"/n) for n in ("ego.mp4","side.mp4")})
    write(root/"result.json",result)
    failure="、".join(evaluation["failure_reason"]) or "无"
    text=f"""# 重建场景单条策略测试\n\n状态：{result['status']}；任务成功：{result['task_success']}；物理有效：{result['physics_valid']}。\n\n实际场景：`{result['tested_xml']}`\n\n重建源：`{result['source_xml']}`；仅补回 G1，环境尺度、位置、材质、物理属性和固定世界相机保持不变。源场景静置检查已失败，不代表真实动力学。\n\nCheckpoint：`{selection['checkpoint']}`，新增训练 {selection['step']} 步。依据：{selection['saved_validation_records']} 条已有验证记录最低选择分数 {selection['selection']['score']:.9f}。加载冻结代码和内嵌配置/归一化。\n\n运行到 {result['last_sim_time_s']:.5f} 秒；模型预测 {len(events)} 次；原生物理步 {motion['native_physics_steps']}；额外初始观测 {motion['recorded_initial_snapshots']} 次。停止原因：{result['stop_reason']}。本次执行/导出耗时 {wall:.2f} 秒。失败项：{failure}。\n\n苹果最终位移 {motion['apple_final_displacement_m']*1000:.2f} mm；最大 XY 位移 {motion['apple_max_xy_displacement_m']*1000:.2f} mm；盘子最终位移 {motion['plate_final_displacement_m']*1000:.2f} mm。细分抓取、承托、接触和穿透证据见 `test/evaluation.json` 与 `test/run/report.json`。\n\n实时当前图像输入、固定基座、50Hz 控制、30/20/10 预测执行协议，seed42。没有重训、拟合、教师动作、换色或第二条样本。机器人速度缺少测量而设为零；root 沿用未实测 Mount 假设。frame0 的 43 个关节为观测状态。该 episode 不属于原 250 条划分，不能称为标准测试集成绩。\n\n兼容修改见 `runtime_adaptations.json`；初始状态/相机/环境保持与重载检查见 `scene_adaptation_check.json`。动态盘子承托评价使用每步记录的盘子位姿；所有原验收阈值保持不变。初始布局的旧名义软垫间隙判据仅作诊断，不通过移动场景来满足。\n\n视频：`test/videos/ego.mp4`、`test/videos/side.mp4`（640×480、30FPS）。\n\n复现命令：\n\n```bash\ncd /root/autodl-tmp/Movens-Sim\n/root/autodl-tmp/Movens-Interact/.local/train-env/bin/python -B scripts/test_reconstructed_policy.py --config configs/policy_test_wrist48500_v1.json --resume\n```\n\n已完成输出用 --resume 校验/读取，不再发起测试。中断的同一试跑沿用已有检查点，不改变 seed 或输入身份。\n"""
    (root/"report.md").write_text(text)
    import html
    (root/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>Reconstructed policy test</title><h1>episode 000000 / wrist 48500</h1><video controls width='640' src='test/videos/ego.mp4'></video><video controls width='640' src='test/videos/side.mp4'></video><p><a href='report.md'>报告</a> · <a href='test/evaluation.json'>指标</a></p><pre>"+html.escape(json.dumps(result,indent=2,ensure_ascii=False))+"</pre>")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--phase",choices=("prepare","run","render","evaluate","all"),default="all")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();cfg=read(args.config);root=Path(cfg["run_root"])
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg["gpu"])
    os.environ["MUJOCO_EGL_DEVICE_ID"]=str(cfg["gpu"])
    lock_path=root.parent/(root.name+".lock");lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        cfg=prepare(args.config)
        if args.phase=="prepare":
            print(json.dumps(dict(status="prepared",root=str(root))),flush=True);return
        if (root/"result.json").exists():
            if not args.resume:
                raise ValueError("Completed one-case test exists; use --resume to verify/read it")
            result=read(root/"result.json")
            for name,digest in read(root/"completion.json")["files"].items():
                if sha(root/name)!=digest:raise ValueError("Completed output changed: "+name)
            print(json.dumps(result,ensure_ascii=False),flush=True);return
        runtime=install_adapters(root);case=root/"test";ep=cfg["episode"]
        if args.phase in ("run","all"):
            if (case/"run/report.json").exists():
                if not args.resume:raise ValueError("Simulation exists; use --resume")
            else:
                policy=runtime.Policy(root)
                runtime.simulate(root,ep,"reconstructed",policy,out=case)
                del policy
        if args.phase in ("render","all"):
            runtime.render_case(root,ep,"reconstructed",out=case)
        if args.phase in ("evaluate","all"):
            runtime.evaluate_case(root,ep,"reconstructed",out=case)
            result=summarize(root,time.monotonic()-started)
            files=[p for p in case.rglob("*") if p.is_file() and not p.name.endswith(".partial.mp4")]+[root/"result.json",root/"model_loaded.json",root/"scene_adaptation_check.json",root/"report.md"]
            write(root/"completion.json",dict(files={str(p.relative_to(root)):sha(p) for p in files},status="one_test_finished"))
            print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
