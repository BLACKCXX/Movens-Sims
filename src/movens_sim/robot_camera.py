"""Observed-state robot FK and explicitly approximate optical mounting priors."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .scene import quat_from_matrix
from .util import ROOT, read, sha, write


def download_pinned(url, path, git_blob=None):
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        if git_blob and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() != git_blob:
            raise ValueError('Official robot Git blob mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix+'.partial')
        temp.write_bytes(content)
        temp.replace(path)
    content = path.read_bytes()
    if git_blob and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() != git_blob:
        raise ValueError('Cached official robot Git blob mismatch')
    return dict(url=url, path=str(path), sha256=sha(path), git_blob=git_blob)


def observed_joint_values(parquet, mapping, robot, frame, timestamp, tolerance):
    table = pq.read_table(parquet, columns=['frame_index','timestamp','observation.state'])
    indices = np.flatnonzero(np.asarray(table['frame_index']) == frame)
    if len(indices) != 1:
        raise ValueError('Reference frame must map to one observed state')
    row = table.slice(int(indices[0]),1).to_pylist()[0]
    if abs(row['timestamp']-timestamp) > tolerance:
        raise ValueError('Observed state/video timestamp mismatch')
    state = np.asarray(row['observation.state'], dtype=float)
    if state.shape != (43,) or not np.isfinite(state).all():
        raise ValueError('Expected 43 finite observed joints')
    values = dict(zip(mapping['source_body_names'],state[:29]))
    hand = state[29:][mapping['hand_state_to_canonical']]
    values.update(zip(robot['hand_joint_names'],hand))
    if len(values) != 43:
        raise ValueError('Joint-name mapping is not one-to-one')
    return values, row


def camera_transform(torso_position, torso_rotation, translation, relative_pitch_deg, optical_offset):
    """Mount +X forward => MuJoCo -Z forward. No extra world pitch."""
    mount_rotation = torso_rotation @ Rotation.from_euler('y',relative_pitch_deg,degrees=True).as_matrix()
    mount_position = torso_position + torso_rotation @ translation
    # Columns are MuJoCo camera right, up, backward in mounting-link axes.
    link_from_mujoco = np.array([[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]])
    rotation = mount_rotation @ link_from_mujoco
    position = mount_position + mount_rotation @ optical_offset
    return dict(xyz_m=position, quaternion_wxyz=quat_from_matrix(rotation),
                rotation_camera_to_world=rotation, mount_reference_world_m=mount_position,
                mount_rotation_to_world=mount_rotation,
                down_angle_deg=float(np.degrees(np.arcsin(np.clip(rotation[2,2],-1,1)))),
                forward_world=-rotation[:,2], link_from_mujoco=link_from_mujoco)


def prepare_prior(entry, config, run, imported=None, table_top=None):
    run=Path(run); folder=run/'inputs'; folder.mkdir(parents=True,exist_ok=True)
    imported=Path(imported) if imported is not None else ROOT/'data'/f"episode_{entry['episode']:06}"
    raw=Path(config['source_project'])/'data/apple_raw'/entry['data_revision']
    source=raw/'data'/f"chunk-{entry['episode']//1000:03}"/f"episode_{entry['episode']:06}.parquet"
    target=folder/source.name
    if not target.exists(): shutil.copy2(source,target)
    if sha(source)!=sha(target): raise ValueError('Observed-state copy checksum mismatch')
    rc=config['robot']; official=folder/'official_g1_rev10.urdf'
    url=f"https://raw.githubusercontent.com/{rc['repository']}/{rc['revision']}/{rc['path']}"
    official_source=download_pinned(url,official,rc['git_blob'])
    doc_url=f"https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/{rc['mount_doc_revision']}/hardware/camera_mount/README.md"
    doc=download_pinned(doc_url,folder/'official_mount_README.md')
    robot=read(imported/'source/robot.json'); mapping=read(imported/'source/prepared_manifest.json')['mapping']
    joints,row=observed_joint_values(target,mapping,robot,rc['reference_frame'],rc['reference_timestamp_s'],rc['timestamp_tolerance_s'])
    model=mujoco.MjModel.from_xml_path(str(imported/'source/scene.xml')); data=mujoco.MjData(model)
    root=model.body('pelvis').id
    model.body_pos[root]=robot['reset_protocol']['root_xyz']
    model.body_quat[root]=robot['reset_protocol']['root_quaternion_wxyz']
    for name,value in joints.items():
        j=model.joint(name).id
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE: raise ValueError('Observed joint must be hinge')
        data.qpos[model.jnt_qposadr[j]]=value
    mujoco.mj_forward(model,data)
    urdf=ET.parse(official).getroot(); transforms=[]
    for joint in ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']:
        node=urdf.find(f"./joint[@name='{joint}']"); child=node.find('child').get('link')
        xyz=np.fromstring(node.find('origin').get('xyz'),sep=' ')
        axis=np.fromstring(node.find('axis').get('xyz'),sep=' ')
        np.testing.assert_allclose(model.body_pos[model.body(child).id],xyz,atol=1e-9)
        np.testing.assert_allclose(model.jnt_axis[model.joint(joint).id],axis,atol=1e-9)
        transforms.append(dict(joint=joint,body=child,origin_m=xyz,axis=axis))
    mount=urdf.find(f"./joint[@name='{rc['mount_joint']}']")
    if mount.find('parent').get('link')!='torso_link': raise ValueError('Unexpected mount parent')
    origin=mount.find('origin'); translation=np.fromstring(origin.get('xyz'),sep=' ')
    rpy=np.fromstring(origin.get('rpy'),sep=' ')
    if abs(rpy[0])+abs(rpy[2])>1e-9:raise ValueError('Mount profile implementation expects pitch-only reference')
    np.testing.assert_allclose(np.degrees(rpy[1]),rc['mount_profiles']['stock_reference'],atol=1e-6)
    torso=model.body('torso_link').id
    priors={name:camera_transform(data.xpos[torso],data.xmat[torso].reshape(3,3),translation,pitch,np.array(rc['optical_center_offset_in_mount_m'])) for name,pitch in rc['mount_profiles'].items()}
    top=float(table_top) if table_top is not None else read(ROOT/config['comparison_root']/f"episode_{entry['episode']:06}/scene/scene_context.json")['table_top_z_m']
    for p in priors.values(): p['height_above_nominal_cloth_m']=float(p['xyz_m'][2]-top)
    result=dict(version='robot_mount_prior_v1',episode=entry['episode'],data_revision=entry['data_revision'],
        frame_index=row['frame_index'],timestamp_s=row['timestamp'],observed_joints_rad=joints,
        observed_state_source=dict(path=str(source),sha256=sha(target),field='observation.state'),
        joint_mapping=mapping,official_model=official_source,mount_document=doc,
        source_model=dict(path=str(imported/'source/scene.xml'),sha256=sha(imported/'source/scene.xml')),
        root=dict(xyz_m=model.body_pos[root],quaternion_wxyz=model.body_quat[root],source=rc['root_source'],measured=False),
        verified_kinematic_chain=transforms,mount_translation_in_torso_m=translation,mount_stock_rpy_rad=rpy,
        optical_center_offset_in_mount_m=rc['optical_center_offset_in_mount_m'],optical_center_status=rc['optical_center_offset_status'],
        torso_world=dict(xyz_m=data.xpos[torso],rotation=data.xmat[torso].reshape(3,3)),priors=priors,
        hardware_confirmed=False,real_root_available=False,
        state_policy='Observed row aligned to real video frame0; no action targets, no reset pose or post-settle robot joints used.',
        soft_surface_policy='Environment retains explicitly saved old post-settle flex geometry, independently of observed robot FK.',
        conventions='World meters; wxyz; MuJoCo -Z forward +Y up; mounting-link +X forward is an explicit assumption.')
    write(run/'camera_prior.json',result)
    return result
