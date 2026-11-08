"""Source-relative table extents and a separate non-contact cloth continuation."""
from __future__ import annotations
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from ..scene import numbers


def obj_bytes(vertices,faces):
    lines=['v '+numbers(v) for v in vertices]
    lines+=['f '+' '.join(str(int(i)+1) for i in f) for f in faces]
    return ('\n'.join(lines)+'\n').encode()


def capture_environment(scene):
    m,d=scene.model,scene.data;records=[];boxes=[]
    for g in scene.geom_sets['table']:
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mi=int(m.geom_dataid[g]);a=int(m.mesh_faceadr[mi]);n=int(m.mesh_facenum[mi])
            v=scene.geom_vertices(g).copy();faces=m.mesh_face[a:a+n].copy()
        elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            local=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
            v=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            faces=ConvexHull(v).simplices.copy()
            for f in faces:
                if np.dot(np.cross(v[f[1]]-v[f[0]],v[f[2]]-v[f[0]]),v[f].mean(0)-v.mean(0))<0:f[[1,2]]=f[[2,1]]
            boxes.append((np.prod(np.sort(m.geom_size[g])[-2:]),v))
        else:continue
        body=int(m.geom_bodyid[g])
        records.append(dict(name=m.geom(g).name,vertices=v,faces=faces,body_pos=d.xpos[body].copy(),body_rot=d.xmat[body].reshape(3,3).copy()))
    if not boxes:raise ValueError('A physical tabletop box is required for bounded extent edits')
    v=max(boxes,key=lambda x:x[0])[1]
    scene.environment_geometry=records;scene.table_original_bounds=np.array([v[:,:2].min(0),v[:,:2].max(0)])


def apply_environment(scene,root,virtual,p):
    bounds=scene.table_original_bounds;anchor=bounds[0,0];depth=bounds[1,0]-anchor
    def transform(v):
        v=v.copy();v[:,0]=anchor-p['table.front_extension']+(v[:,0]-anchor)*(p['table.depth_scale']+p['table.front_extension']/depth)
        v[:,1]*=p['table.width_scale'];return v
    if any(p[k]!=v for k,v in [('table.depth_scale',1.),('table.width_scale',1.),('table.front_extension',0.)]):
        for i,record in enumerate(scene.environment_geometry):
            node=root.find(f".//geom[@name='{record['name']}']")
            if node is None:raise ValueError('Missing named table geom '+str(record['name']))
            world=transform(record['vertices']);v=(world-record['body_pos'])@record['body_rot']
            name=f'v4_table_{i}';filename=name+'.obj';virtual[filename]=obj_bytes(v,record['faces'])
            ET.SubElement(root.find('asset'),'mesh',name=name,file=filename)
            for key in ('size','quat','euler','axisangle','xyaxes','zaxis','fromto'):node.attrib.pop(key,None)
            node.set('pos','0 0 0');node.set('type','mesh');node.set('mesh',name)
    coverage=p['cloth.coverage']
    if coverage<=0:return
    outer=transform(np.c_[bounds,[0.,0.]])[:,:2];inner=scene.support.bounds
    outer=inner+(outer-inner)*coverage
    if np.any(outer[0]>inner[0]+1e-9) or np.any(outer[1]<inner[1]-1e-9):raise ValueError('Cloth continuation must surround the physical patch')
    z=float(scene.support.height_range[1])-.00001
    def rectangle(b):return [[b[0,0],b[0,1],z],[b[1,0],b[0,1],z],[b[1,0],b[1,1],z],[b[0,0],b[1,1],z]]
    v=np.array(rectangle(outer)+rectangle(inner));v=np.concatenate([v,v-[0,0,.0001]])
    faces=[]
    for i in range(4):
        j=(i+1)%4
        faces.extend([(i,j,j+4),(i,j+4,i+4),(i+8,j+12,j+8),(i+8,i+12,j+12),(i,i+8,j+8),(i,j+8,j),(i+4,j+4,j+12),(i+4,j+12,i+12)])
    virtual['v4_cloth_continuation.obj']=obj_bytes(v,faces)
    ET.SubElement(root.find('asset'),'mesh',name='v4_cloth_continuation',file='v4_cloth_continuation.obj')
    ET.SubElement(root.find('worldbody'),'geom',name='cloth_continuation',type='mesh',mesh='v4_cloth_continuation',group='1',contype='0',conaffinity='0',rgba=numbers([p['cloth.gray']]*3+[1.]))


def coverage_seed(scene,observation):
    """Ray-plane extent evidence, not a screen-space cover or support expansion."""
    yy,xx=np.where(observation['target']['cloth']&observation['valid'])
    f=240/np.tan(np.radians(scene.locked_camera['fovy_deg']/2));r=np.asarray(scene.locked_camera['rotation_camera_to_world']);eye=np.asarray(scene.locked_camera['xyz_m'])
    rays=np.c_[(xx-320)/f,-(yy-240)/f,-np.ones(len(xx))]@r.T
    distance=(scene.support.height_range[1]-eye[2])/rays[:,2];good=(distance>0)&np.isfinite(distance)
    xy=eye[:2]+distance[good,None]*rays[good,:2]
    wanted=np.array([xy.min(0),xy.max(0)]);b=scene.table_original_bounds;depth=b[1,0]-b[0,0]
    p={'cloth.coverage':1.,'table.depth_scale':float(np.clip((wanted[1,0]+.01-b[0,0])/depth,1,2.5)),
       'table.width_scale':float(np.clip(max(abs(wanted[:,1]))/max(abs(b[:,1]))+.01,1,1.6)),
       'table.front_extension':float(np.clip(b[0,0]-wanted[0,0]+.01,0,.15))}
    return dict(updates=p,ray_plane_required_bounds_m=wanted.tolist(),original_table_bounds_m=b.tolist(),source='Real cloth pixels intersected with fixed h0 plane under the approximate camera; hidden extents remain ambiguous',physical_support_changed=False)
