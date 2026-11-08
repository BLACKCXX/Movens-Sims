"""Separate physical/visual repair from acceptance-semantic and tolerance changes."""
from pathlib import Path
import mujoco
import numpy as np
from .diagnostics import members_and_groups,terminal_pass
from ..util import read


def physical_attribution(old_run,current_physical):
    old_run=Path(old_run);xml=old_run/'physics/portable/scene.xml';m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
    with np.load(xml.with_suffix('.state.npz')) as z:d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel']
    mujoco.mj_forward(m,d);initial={k:(d.xpos[m.body(k+'_root').id].copy(),d.xquat[m.body(k+'_root').id].copy()) for k in ('apple','plate')}
    with np.load(old_run/'diagnostic_final_state.npz') as z:d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel']
    mujoco.mj_forward(m,d);groups=members_and_groups(m)[1];peaks=read(old_run/'physical_diagnostic.json')['objects'];rows={}
    original=dict(translation_m_max=.005,rotation_deg_max=5.,penetration_m_max=.001);relaxed=dict(translation_m_max=.01,rotation_deg_max=10.,penetration_m_max=.002)
    for label in ('apple','plate'):
        b=m.body(label+'_root').id;p,q=initial[label];relevant=np.any(np.isin(d.contact.geom,groups[label]),axis=1)
        old=dict(translation_final_m=float(np.linalg.norm(d.xpos[b]-p)),rotation_final_deg=float(np.degrees(2*np.arccos(np.clip(abs(d.xquat[b]@q),0,1)))),penetration_final_m=float(max(0.,-d.contact.dist[relevant].min())) if relevant.any() else 0.)
        new=current_physical['objects'][label];old_peak=bool(peaks[label]['translation_max_m']<=.005 and peaks[label]['rotation_max_deg']<=5 and peaks[label]['penetration_max_m']<=.001)
        old_terminal=terminal_pass(old,original);old_relaxed=terminal_pass(old,relaxed);new_original=terminal_pass(new,original);new_relaxed=terminal_pass(new,relaxed)
        rows[label]=dict(old_peak_metrics=peaks[label],unchanged_old_terminal_metrics=old,new_terminal_metrics={k:new[k] for k in old},old_peak_under_old_limits=old_peak,unchanged_old_terminal_under_old_limits=old_terminal,unchanged_old_terminal_under_new_limits=old_relaxed,new_terminal_under_old_limits=new_original,new_terminal_under_new_limits=new_relaxed,physical_metric_improvements={k:old[k]-new[k] for k in old},criterion_change_alone_suffices=not old_peak and old_terminal,tolerance_change_alone_suffices=not old_terminal and old_relaxed,repair_required_for_new_pass=not old_relaxed and new_relaxed)
    return dict(objects=rows,source='Read-only original XML/initial state and saved one-second terminal state; no old trajectory rerun or changed reference',interpretation='Flags may overlap across metrics; compare raw terminal values to distinguish repair, peak-to-terminal semantics and tolerance changes.')
