#!/usr/bin/env python3
"""Independent finite-ray comparison of visible plate and collision bearing surfaces."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import mujoco
import numpy as np
from movens_sim.util import ROOT,read,write,sha

def audit(run):
    path=run/'scene/scene_physics.xml';m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);mujoco.mj_forward(m,d);root=m.body('plate_root').id;bodies={root}
    for b in range(m.nbody):
        if int(m.body_parentid[b]) in bodies:bodies.add(b)
    geoms=np.where(np.isin(m.geom_bodyid,list(bodies)))[0];m.geom_group[:]=5
    collision=[g for g in geoms if m.geom_contype[g] or m.geom_conaffinity[g]];visual=[g for g in geoms if not(m.geom_contype[g] or m.geom_conaffinity[g])]
    m.geom_group[collision]=0;m.geom_group[visual]=1;rows=[]
    for radius in [0,.03,.06,.08,.087]:
        samples=[]
        for angle in ([0] if radius==0 else np.linspace(0,2*np.pi,16,endpoint=False)):
            point=d.xpos[root]+[radius*np.cos(angle),radius*np.sin(angle),.1];hit=[]
            for group in [0,1]:
                groups=np.zeros(6,np.uint8);groups[group]=1;geom=np.array([-1],np.int32);distance=mujoco.mj_ray(m,d,point,np.array([0.,0.,-1.]),groups,True,-1,geom);hit.append(None if distance<0 else float(point[2]-distance))
            samples.append(dict(angle_rad=float(angle),collision_z_m=hit[0],visual_z_m=hit[1],collision_minus_visual_m=hit[0]-hit[1] if None not in hit else None))
        rows.append(dict(radius_m=radius,samples=samples))
    gaps=[s['collision_minus_visual_m'] for row in rows for s in row['samples'] if s['collision_minus_visual_m'] is not None]
    return dict(method='Vertical MuJoCo ray casts separately through source convex collision pieces and visual mesh, at center plus four radii; tests authored mesh surfaces, not a drop test',scene_sha256=sha(path),collision_mesh_count=len(collision),radii=rows,max_collision_above_visual_m=max(gaps),max_abs_collision_visual_gap_m=max(abs(g) for g in gaps),free_plate_dynamics_validated=False,limitation='Finite ray samples cannot certify all convex seams or real physical contact properties')

def attach(run,p):
    samples=[(row['radius_m'],s) for row in p['radii'] for s in row['samples'] if s['collision_minus_visual_m'] is not None];radius,worst=max(samples,key=lambda t:abs(t[1]['collision_minus_visual_m']))
    review=dict(max_abs_collision_visual_gap_m=p['max_abs_collision_visual_gap_m'],worst_radius_m=radius,check_file='plate_support_surface.json',needs_review=p['max_abs_collision_visual_gap_m']>.001,interpretation='Finite ray samples compare inherited collision pieces and visible concave plate; not a free-dynamics or real-contact validation')
    result=read(run/'result.json');result['plate_surface_review']=review
    if review['needs_review']:
        reason='Inherited plate collision/visual support surfaces differ by up to 7.26mm in sampled rays; concave bearing geometry needs review'
        if reason not in result['review_reasons']:result['review_reasons'].append(reason)
    write(run/'result.json',result);return dict(episode=result['episode'],**review)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');a=p.parse_args();root=ROOT/'runs/batch_v1';reviews=[]
    for run in sorted(root.glob('episode_*')):
        if not (run/'result.json').exists():continue
        saved=run/'plate_support_surface.json';path=run/'scene/scene_physics.xml'
        if a.reuse and saved.exists():
            result=read(saved);assert sha(path)==read(run/'physical_diagnostic.json')['scene_sha256']
            assert result.get('scene_sha256',sha(path))==sha(path)
            result['scene_sha256']=sha(path);result['script_sha256']=sha(__file__)
        else:result=audit(run);result['script_sha256']=sha(__file__)
        write(saved,result);reviews.append(attach(run,result))
    write(root/'plate_surface_review.json',reviews);print('Plate surfaces checked:',len(reviews),flush=True)
