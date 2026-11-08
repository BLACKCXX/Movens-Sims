"""One authoritative parameter registry and source-MJCF modification backend."""
from __future__ import annotations

import copy
import io
import json
from collections import OrderedDict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..ingest import portable_copy
from ..render import Renderer
from ..scene import Scene, numbers, matrix_from_quat, quat_from_matrix
from ..util import digest, read, sha, write


class SceneError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def absolute_assets(tree, directory):
    for node in tree.findall('./asset/*'):
        if node.get('file'):
            node.set('file', str((Path(directory)/node.get('file')).resolve()))
    for node in tree.findall('compiler'):
        for key in ('meshdir', 'texturedir', 'assetdir'):
            node.attrib.pop(key, None)
    return tree


def scale_object(root, label, scale, depth=1.):
    """Apply an absolute scale to a fresh source tree, cloning shared meshes.

    The imported rigid object subtrees currently have aligned child frames.
    Nonuniform plate depth is explicitly rejected for rotated child frames.
    """
    body = root.find(f".//body[@name='{label}_root']")
    asset = root.find('asset')
    a = np.array([scale, scale, scale*depth], float)
    det = float(np.prod(a))
    if body is None or not np.isfinite(a).all() or np.min(a) <= 0:
        raise SceneError('invalid_scale', label)
    for b in body.iter('body'):
        if depth != 1. and b is not body and any(k in b.attrib for k in ('quat','euler','axisangle','xyaxes','zaxis')):
            raise SceneError('unsupported_shape_transform', 'Nonuniform depth requires aligned rigid child frames')
        if b is not body and b.get('pos'):
            b.set('pos', numbers(np.fromstring(b.get('pos'), sep=' ')*a))
        for joint in list(b.findall('joint')):
            if joint.get('type', 'hinge') != 'free':
                raise SceneError('articulated_object', label)
        inertial = b.find('inertial')
        if inertial is not None:
            inertial.set('pos', numbers(np.fromstring(inertial.get('pos','0 0 0'),sep=' ')*a))
            inertial.set('mass', str(float(inertial.get('mass'))*det))
            if inertial.get('fullinertia'):
                v=np.fromstring(inertial.get('fullinertia'),sep=' ')
                inertia=np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]])
            else:
                r=matrix_from_quat(np.fromstring(inertial.get('quat','1 0 0 0'),sep=' '))
                inertia=r@np.diag(np.fromstring(inertial.get('diaginertia'),sep=' '))@r.T
            moment=.5*np.trace(inertia)*np.eye(3)-inertia
            moment=det*np.diag(a)@moment@np.diag(a)
            inertia=np.trace(moment)*np.eye(3)-moment
            for key in ('diaginertia','quat','euler','axisangle','xyaxes','zaxis'):
                inertial.attrib.pop(key,None)
            inertial.set('fullinertia',numbers([*np.diag(inertia),inertia[0,1],inertia[0,2],inertia[1,2]]))
    clones={}
    for geom in body.iter('geom'):
        if geom.get('type','sphere') != 'mesh':
            raise SceneError('unsupported_geometry','Object scaling currently supports mesh geoms')
        name=geom.get('mesh')
        if name not in clones:
            mesh=asset.find(f"mesh[@name='{name}']")
            if mesh is None:raise SceneError('missing_mesh',name)
            clone=copy.deepcopy(mesh);clones[name]=label+'__absolute__'+name
            clone.set('name',clones[name]);clone.set('scale',numbers(np.fromstring(mesh.get('scale','1 1 1'),sep=' ')*a));asset.append(clone)
        geom.set('mesh',clones[name])
        if geom.get('pos'):geom.set('pos',numbers(np.fromstring(geom.get('pos'),sep=' ')*a))
        if geom.get('mass'):geom.set('mass',str(float(geom.get('mass'))*det))
    for site in body.iter('site'):
        if site.get('pos'):site.set('pos',numbers(np.fromstring(site.get('pos'),sep=' ')*a))
        if site.get('size'):site.set('size',numbers(np.fromstring(site.get('size'),sep=' ')*scale))


class SupportSurface:
    """Barycentric heights on the actual saved visible triangles, not a flat Z."""
    def __init__(self, vertices, triangles, top_only=False):
        tri=vertices[triangles]
        if top_only:tri=tri[np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])[:,2]>1e-14]
        u=tri[:,1,:2]-tri[:,0,:2];v=tri[:,2,:2]-tri[:,0,:2]
        det=u[:,0]*v[:,1]-u[:,1]*v[:,0];keep=np.abs(det)>1e-14
        self.tri,self.u,self.v,self.det=tri[keep],u[keep],v[keep],det[keep]
        self.tree=cKDTree(self.tri[:,:,:2].mean(1))
        self.bounds=np.array([vertices[:,:2].min(0),vertices[:,:2].max(0)])
        height_vertices=self.tri.reshape(-1,3) if top_only else vertices
        self.height_range=[float(height_vertices[:,2].min()),float(height_vertices[:,2].max())]

    def __call__(self, xy):
        xy=np.atleast_2d(xy);_,idx=self.tree.query(xy,k=min(24,len(self.tri)))
        if idx.ndim==1:idx=idx[:,None]
        delta=xy[:,None,:]-self.tri[idx,0,:2]
        u,v,det=self.u[idx],self.v[idx],self.det[idx]
        b=(delta[:,:,0]*v[:,:,1]-delta[:,:,1]*v[:,:,0])/det
        c=(u[:,:,0]*delta[:,:,1]-u[:,:,1]*delta[:,:,0])/det
        ok=(b>=-1e-7)&(c>=-1e-7)&(b+c<=1+1e-7)
        hit=ok.argmax(1);rows=np.arange(len(xy));chosen=idx[rows,hit]
        z=(1-b[rows,hit]-c[rows,hit])*self.tri[chosen,0,2]+b[rows,hit]*self.tri[chosen,1,2]+c[rows,hit]*self.tri[chosen,2,2]
        z[~ok.any(1)]=np.nan
        return z


class SceneBackend:
    def __init__(self, static_xml, physics_xml=None, physics_state=None, initial=None, camera=None, support_top_only=False, initialize_supported=False, config=None):
        self.config=config or {};self.v5=self.config.get('workflow') in ('terra_closed_loop_v5','ablation_no_terra_v5');self.classified=self.config.get('workflow') in ('classified_reconstruction_v4','terra_closed_loop_v5','ablation_no_terra_v5')
        self.support_limits=self.config.get('initial_support',dict(gap_min_m=-.001,gap_max_m=.002,pair_penetration_m_max=.001))
        self.path=Path(static_xml);self.physics_path=Path(physics_xml) if physics_xml else None
        self.physics_state=Path(physics_state) if physics_state else None
        self.template=absolute_assets(ET.parse(self.path).getroot(),self.path.parent)
        self.physics_template=absolute_assets(ET.parse(self.physics_path).getroot(),self.physics_path.parent) if physics_xml else None
        authored=ET.parse(self.path).getroot()
        self.asset_hashes={node.get('file'):sha((self.path.parent/node.get('file')).resolve()) for node in authored.findall('./asset/*') if node.get('file')}
        self.source_hash=digest(dict(xml=sha(self.path),assets=self.asset_hashes));self.cache=OrderedDict();self.stage='geometry';self.shape_enabled=False
        if self.classified:
            contract_files=[Path(__file__).parent/name for name in ('backend.py','environment.py','appearance.py')]
            if self.v5:contract_files += [Path(__file__).parent/'v5_material.py',Path(__file__).parent/'v5_metrics.py',Path(__file__).parent.parent/'render.py']
            self.render_contract=digest({p.name:sha(p) for p in contract_files})
            self.source_hash=digest(dict(source=self.source_hash,render_contract=self.render_contract))
        self.render_passes=0;self.initial_poses={};self.locked_camera=camera
        self.params={}
        for label in ('apple','plate'):
            b=self.template.find(f".//body[@name='{label}_root']")
            xyz=np.fromstring(b.get('pos','0 0 0'),sep=' ');q=np.fromstring(b.get('quat','1 0 0 0'),sep=' ')
            self.initial_poses[label]=dict(xyz_m=xyz,quaternion_wxyz=q)
            self.params.update({label+'.x':float(xyz[0]),label+'.y':float(xyz[1]),label+'.scale':1.,label+'.rx':0.,label+'.ry':0.})
        self.params.update({'apple.rz':0.,'plate.depth_ratio':1.,'light.ambient':.35,'light.diffuse':.6,'cloth.gray':.36,'plate.albedo':.8,'material.specular':1.,'material.shininess':1.,'apple.tint_r':1.,'apple.tint_g':1.,'apple.tint_b':1.,'apple.yellow_mix':0.})
        if self.v5:
            from .v5_material import defaults
            self.params.update(defaults(self.template))
        if initial:self.params.update(initial)
        self._activate(self.params, solve=False)
        vertices=[];faces=[];offset=0
        for g in self.geom_sets['cloth']:
            if not self.model.geom(g).name.startswith('frozen_'):continue
            mi=int(self.model.geom_dataid[g]);points=self.geom_vertices(g);a=int(self.model.mesh_faceadr[mi]);n=int(self.model.mesh_facenum[mi])
            vertices.append(points);faces.append(self.model.mesh_face[a:a+n]+offset);offset+=len(points)
        if not vertices:raise SceneError('missing_support','No frozen physical soft surface')
        self.support=SupportSurface(np.concatenate(vertices),np.concatenate(faces),top_only=support_top_only)
        if self.classified:
            from .environment import capture_environment
            capture_environment(self)
            self.params.update({'table.depth_scale':1.,'table.width_scale':1.,'table.front_extension':0.,'cloth.coverage':0.})
        self.registry_entries=self._registry()
        self.initial_support_seed={}
        if initialize_supported:
            # FK is a prior and may overhang the original support. Start the
            # numerical search at the nearest feasible XY, without growing the
            # table or changing any immutable FK input/asset/size.
            for label in ('apple','plate'):
                pts=np.concatenate([self.vertices(label,collision=True),self.vertices(label)])[:,:2]
                lo,hi=self.support.bounds;minimum,maximum=pts.min(0),pts.max(0);margin=.0001
                if np.any(maximum-minimum>hi-lo-2*margin):raise SceneError('initial_object_too_wide',label)
                delta=np.maximum(lo+margin-minimum,0.)+np.minimum(hi-margin-maximum,0.)
                if np.any(delta):
                    before=[self.params[label+'.x'],self.params[label+'.y']]
                    for i,axis in enumerate('xy'):self.params[label+'.'+axis]+=float(delta[i])
                    self.initial_support_seed[label]=dict(raw_fk_xy_m=before,delta_xy_m=delta.tolist(),search_seed_xy_m=[self.params[label+'.x'],self.params[label+'.y']],reason='Nearest complete visual/collision footprint inside fixed support; numerical initialization, not hand-direction compensation')
            self._activate(self.params)
            if 'apple_plate.intersection' in self.geometry_metrics()['failures']:
                saved=self.params.copy();a=self.sample_collision('apple');p=self.sample_collision('plate');candidates=[]
                for axis in (0,1):
                    for delta in (a[:,axis].min()-.001-p[:,axis].max(),a[:,axis].max()+.001-p[:,axis].min()):
                        proposal=saved.copy();proposal['plate.'+'xy'[axis]]+=float(delta)
                        try:
                            self._activate(proposal)
                            if self.geometry_metrics()['physical_valid']:candidates.append((abs(float(delta)),proposal))
                        except SceneError:pass
                if not candidates:raise SceneError('no_feasible_initial_layout','No supported, nonintersecting initial XY seed within the fixed table')
                _,chosen=min(candidates,key=lambda item:item[0]);self._activate(chosen)
                raw=self.initial_poses['plate']['xyz_m'][:2]
                current=np.array([chosen['plate.x'],chosen['plate.y']])
                self.initial_support_seed['plate']=dict(raw_fk_xy_m=raw.tolist(),delta_xy_m=(current-raw).tolist(),search_seed_xy_m=current.tolist(),reason='Nearest tested AABB-separated, supported geometric search seed for an intersecting FK prior; no image claim, no size/table/hand-direction change')
        self.apply({}, internal=True)

    def _registry(self):
        entries=[]
        def add(id,unit,lo,hi,stages,meaning,source='engineering search range; not measured',prior=None):
            entries.append(dict(id=id,unit=unit,coordinate_frame='world XYZ meters; world rotation vector degrees relative to immutable object quaternion wxyz',default=self.params[id],lower=lo,upper=hi,stages=stages,source=source,prior=prior,meaning=meaning,linked_updates=['support_z','collision','inertia','renderer'] if id.split('.')[0] in ('apple','plate') else ['appearance']))
        for label in ('apple','plate'):
            for i,a in enumerate('xy'):add(label+'.'+a,'m',float(self.support.bounds[0,i]),float(self.support.bounds[1,i]),['geometry'],'Absolute XY; complete scaled footprint must remain on physical support')
            add(label+'.scale','ratio',.5,2.,['geometry'],'Absolute overall scale relative to immutable imported source geometry',prior=dict(type='weak_log_normal',center=1.,weight=.002))
            for a in 'xy':add(label+'.r'+a,'deg',-45. if label=='apple' else -5.,45. if label=='apple' else 5.,['geometry'],'Supported orientation; plate tilt requires explicit residual evidence')
        add('apple.rz','deg',-180.,180.,['geometry'],'Apple orientation around world Z')
        add('plate.depth_ratio','ratio',.7,1.3,['geometry'],'Additional vertical profile scale, only after recorded shape evidence')
        if self.classified:
            for key,lo,hi in [('table.depth_scale',1.,2.5),('table.width_scale',1.,1.6),('table.front_extension',0.,.15),('cloth.coverage',0.,1.)]:
                add(key,'m' if key.endswith('extension') else 'ratio',lo,hi,['geometry'],'Absolute world-plane extent; physical soft patch is never stretched; display cloth cannot support objects')
        for id,lo,hi in [('light.ambient',.1,.8),('light.diffuse',.1,1.),('cloth.gray',.1,.65),('plate.albedo',.45,1.),('material.specular',0.,1.6),('material.shininess',.2,2.),('apple.tint_r',.3,1.6),('apple.tint_g',.3,1.6),('apple.tint_b',.3,1.6),('apple.yellow_mix',0.,.8)]:add(id,'ratio' if 'tint' in id or 'mix' in id or 'material' in id else 'linear_rgb_or_phong',lo,hi,['appearance'],'Renderer-supported appearance control; material values multiply their immutable authored baselines; geometry locked')
        if self.v5:
            from .v5_material import registry
            entries += registry(self.params)
            for p in entries:
                p['dependencies']=['geometry','physical','appearance'] if 'geometry' in p['stages'] else ['appearance']
        return entries

    def registry(self):return copy.deepcopy(self.registry_entries)

    def validate(self, values, stage=None):
        reg={x['id']:x for x in self.registry_entries}
        for key,value in values.items():
            if key not in reg:raise SceneError('unknown_parameter',key)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value):raise SceneError('nonfinite_parameter',key)
            spec=reg[key]
            if not spec['lower']<=value<=spec['upper']:raise SceneError('out_of_bounds',key)
            if stage and stage not in spec['stages']:raise SceneError('wrong_stage',key)
            if key=='plate.depth_ratio' and value!=1. and not self.shape_enabled:raise SceneError('shape_evidence_required',key)
            if key.startswith('plate.r') and value!=0. and not getattr(self,'plate_tilt_enabled',False):raise SceneError('orientation_evidence_required',key)

    def _xml(self, params, physics=False):
        root=copy.deepcopy(self.physics_template if physics else self.template);virtual={}
        if self.classified and hasattr(self,'environment_geometry'):
            from .environment import apply_environment
            apply_environment(self,root,virtual,params)
        for label in ('apple','plate'):scale_object(root,label,params[label+'.scale'],params['plate.depth_ratio'] if label=='plate' else 1.)
        cam=root.find(".//camera[@name='approx_ego_camera']")
        if self.locked_camera:
            for key in ('euler','xyaxes','zaxis','axisangle'):cam.attrib.pop(key,None)
            cam.set('pos',numbers(self.locked_camera['xyz_m']));cam.set('quat',numbers(self.locked_camera['quaternion_wxyz']));cam.set('fovy',str(self.locked_camera['fovy_deg']))
        visual=root.find('visual')
        if visual is None:visual=ET.SubElement(root,'visual')
        head=visual.find('headlight')
        if head is None:head=ET.SubElement(visual,'headlight')
        head.set('ambient',numbers([params['light.ambient']]*3));head.set('diffuse',numbers([params['light.diffuse']]*3))
        for node in root.findall('.//geom')+root.findall('./deformable/flex'):
            if node.get('name','').startswith('frozen_') or node.get('name')=='cloth_continuation' or node.tag=='flex':node.set('rgba',numbers([params['cloth.gray']]*3+[1.]))
        for label in ('apple','plate'):
            b=root.find(f".//body[@name='{label}_root']");materials={g.get('material') for g in b.iter('geom') if g.get('material')}
            for name in materials:
                node=root.find(f"./asset/material[@name='{name}']")
                if node is None:continue
                original=np.fromstring(node.get('rgba','1 1 1 1'),sep=' ')
                color=np.array([params['plate.albedo']]*3) if label=='plate' else np.minimum(1.,original[:3]*[params['apple.tint_'+a] for a in 'rgb'])
                node.set('rgba',numbers([*color,original[3]]));node.set('specular',str(float(node.get('specular','.5'))*params['material.specular']));node.set('shininess',str(float(node.get('shininess','.5'))*params['material.shininess']))
                if label=='apple' and params['apple.yellow_mix']>0:
                    tex=root.find(f"./asset/texture[@name='{node.get('texture')}']")
                    if tex is not None and tex.get('file'):
                        if self.classified:
                            from .appearance import apple_texture
                            visual=next(g for g in b.iter('geom') if g.get('material')==name and g.get('mesh'))
                            mesh=root.find(f"./asset/mesh[@name='{visual.get('mesh')}']")
                            filename='v4_height_apple_texture.png';virtual[filename]=apple_texture(tex.get('file'),mesh.get('file'),params['apple.yellow_mix']);tex.set('file',filename)
                            continue
                        img=np.array(Image.open(tex.get('file')).convert('RGB'),float)/255.;yy=np.linspace(1.,0.,img.shape[0])[:,None,None]
                        mix=params['apple.yellow_mix']*yy;lum=img.max(2,keepdims=True);yellow=lum*np.array([1.,.83,.12]);rgb=np.clip((1-mix)*img+mix*yellow,0,1)
                        buf=io.BytesIO();Image.fromarray((rgb*255).astype(np.uint8)).save(buf,format='PNG');filename='v3_apple_texture.png';virtual[filename]=buf.getvalue();tex.set('file',filename)
        if self.v5:
            from .v5_material import apply_material
            apply_material(self,root,virtual,params)
        return root,virtual

    def _activate(self, params, solve=True):
        key=digest(dict(geometry_and_appearance={k:v for k,v in params.items() if not k.endswith(('.x','.y','.rx','.ry','.rz'))},camera=self.locked_camera))
        if key not in self.cache:
            root,virtual=self._xml(params);model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets=virtual)
            self.cache[key]=dict(model=model,data=mujoco.MjData(model),renderers={})
            while len(self.cache)>4:
                _,entry=self.cache.popitem(last=False)
                for renderer in entry['renderers'].values():renderer.renderer.close()
        entry=self.cache[key];self.cache.move_to_end(key);self.active=entry;self.model=entry['model'];self.data=entry['data'];self.cam=self.model.camera('approx_ego_camera').id
        self.descendants={};self.geom_sets={};self.joints={}
        for label in ('apple','plate','table'):
            body=self.model.body(label+'_root').id;members={body}
            for b in range(self.model.nbody):
                if int(self.model.body_parentid[b]) in members:members.add(b)
            self.descendants[label]=members;self.geom_sets[label]=np.flatnonzero(np.isin(self.model.geom_bodyid,list(members)))
            if label!='table':self.joints[label]=self.model.joint(label+'_freejoint').id
        self.apple=self.model.body('apple_root').id;self.plate=self.model.body('plate_root').id
        self.geom_sets['cloth']=np.array([g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith('frozen_') or self.model.geom(g).name=='cloth_continuation'])
        mujoco.mj_resetData(self.model,self.data)
        for label in ('apple','plate'):
            q=int(self.model.jnt_qposadr[self.joints[label]]);initial=self.initial_poses[label]
            self.data.qpos[q:q+3]=[params[label+'.x'],params[label+'.y'],initial['xyz_m'][2]]
            rv=[params[label+'.rx'],params[label+'.ry'],params.get(label+'.rz',0.)]
            self.data.qpos[q+3:q+7]=quat_from_matrix(Rotation.from_rotvec(rv,degrees=True).as_matrix()@matrix_from_quat(initial['quaternion_wxyz']))
            mujoco.mj_forward(self.model,self.data)
            if solve:
                pts=self.sample_collision(label);height=self.support(pts[:,:2])
                if not np.isfinite(height).all():raise SceneError('outside_support',label)
                self.data.qpos[q+2]-=float(np.min(pts[:,2]-height));mujoco.mj_forward(self.model,self.data)
        self.params=params.copy()

    def apply(self, values, stage=None, internal=False):
        if not internal:
            if self.stage not in ('geometry','appearance'):raise SceneError('frozen_scene',self.stage)
            self.validate(values,stage or self.stage)
        old=self.params.copy();params={**old,**values}
        try:
            if np.linalg.norm([params['plate.rx'],params['plate.ry']])>5.+1e-10:raise SceneError('orientation_norm_bound','Plate tilt exceeds 5 degrees')
            self._activate(params)
            geometry=self.geometry_metrics()
            if not geometry['physical_valid']:raise SceneError('invalid_support_or_intersection',json.dumps(geometry['failures']))
        except BaseException:
            self._activate(old)
            raise
        return self.inspect()

    def geom_vertices(self,g):
        mi=int(self.model.geom_dataid[g]);a=int(self.model.mesh_vertadr[mi]);n=int(self.model.mesh_vertnum[mi])
        return self.model.mesh_vert[a:a+n]@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]

    def vertices(self,label,collision=False):return Scene.vertices(self,label,collision)

    def sample_collision(self,label):
        samples=[]
        for g in self.geom_sets[label]:
            if not(self.model.geom_contype[g] or self.model.geom_conaffinity[g]):continue
            points=self.geom_vertices(g);mi=int(self.model.geom_dataid[g]);a=int(self.model.mesh_faceadr[mi]);n=int(self.model.mesh_facenum[mi]);faces=self.model.mesh_face[a:a+n]
            samples.extend([points,points[faces].mean(1)])
        return np.concatenate(samples)

    def geometry_metrics(self):
        result={};failures=[]
        for label in ('apple','plate'):
            pts=self.sample_collision(label);h=self.support(pts[:,:2]);gap=pts[:,2]-h;visual=self.vertices(label);vg=visual[:,2]-self.support(visual[:,:2]);finite=bool(np.isfinite(gap).all())
            minimum=float(np.min(gap)) if finite else None
            body=self.model.body(label+'_root').id
            result[label]=dict(xyz_m=self.data.xpos[body].tolist(),quaternion_wxyz=self.data.xquat[body].tolist(),signed_support_gap_m=minimum,penetration_m=max(0.,-minimum) if finite else None,visual_support_gap_m=float(np.nanmin(vg)) if np.isfinite(vg).any() else None,footprint_inside_physical_support=finite,dimensions_world_m=np.ptp(visual,axis=0).tolist(),aabb_world_m=[pts.min(0).tolist(),pts.max(0).tolist()],mass_kg=float(sum(self.model.body_mass[list(self.descendants[label])])),absolute_scale=self.params[label+'.scale'])
            if not finite or minimum is None or not self.support_limits['gap_min_m']<=minimum<=self.support_limits['gap_max_m']:failures.append(label+'.support')
            if not np.isfinite(vg).all() or not self.support_limits['gap_min_m']<=float(np.nanmin(vg))<=self.support_limits['gap_max_m']:failures.append(label+'.visual_support')
        bounds=[np.array(result[k]['aabb_world_m']) for k in ('apple','plate')]
        overlap=bool(np.all(np.minimum(bounds[0][1],bounds[1][1])>np.maximum(bounds[0][0],bounds[1][0])))
        distance=None
        if overlap:
            distances=[]
            for a in self.geom_sets['apple']:
                if not(self.model.geom_contype[a] or self.model.geom_conaffinity[a]):continue
                for b in self.geom_sets['plate']:
                    if self.model.geom_contype[b] or self.model.geom_conaffinity[b]:distances.append(float(mujoco.mj_geomDistance(self.model,self.data,int(a),int(b),.05,None)))
            distance=min(distances)
            if distance < -self.support_limits['pair_penetration_m_max']:failures.append('apple_plate.intersection')
        result.update(physical_valid=not failures,failures=failures,object_pair_distance_m=distance,support_policy='Barycentric interpolation on saved deformed visible triangles; collision vertices and face centroids; flex thickness/dynamics checked separately',support_height_range_m=self.support.height_range,physical_support_xy_bounds_m=self.support.bounds.tolist())
        return result

    def inspect(self):
        return dict(params=self.params.copy(),geometry=self.geometry_metrics(),registry=self.registry(),stage=self.stage,camera=self.locked_camera,source_hash=self.source_hash,candidate_id=self.candidate_id())

    def candidate_id(self):return digest(dict(source=self.source_hash,params=self.params,camera=self.locked_camera))[:20]

    def render(self,width=640,height=480):
        size=(width,height)
        if size not in self.active['renderers']:self.active['renderers'][size]=Renderer(self.model,width,height)
        renderer=self.active['renderers'][size];rgb=renderer.rgb(self.data);masks=renderer.masks(self.data,self.geom_sets);self.render_passes+=2
        return rgb,masks

    def export(self,out,physics=False):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);root,virtual=self._xml(self.params,physics)
        for label in ('apple','plate'):
            b=self.model.body(label+'_root').id;node=root.find(f".//body[@name='{label}_root']")
            for key in ('euler','axisangle','xyaxes','zaxis'):node.attrib.pop(key,None)
            node.set('pos',numbers(self.data.xpos[b]));node.set('quat',numbers(self.data.xquat[b]))
        staging=out/'authored';staging.mkdir(exist_ok=True)
        for filename,content in virtual.items():(staging/filename).write_bytes(content)
        source=staging/'scene.xml';ET.ElementTree(root).write(source,encoding='utf-8',xml_declaration=True)
        closure=portable_copy(source,out/'portable');path=out/'portable/scene.xml'
        model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
        if physics and self.physics_state:
            with np.load(self.physics_state) as z:
                for name,adr,n in zip(z['joint_names'],z['qpos_addresses'],z['qpos_sizes']):
                    if str(name) in ('apple_freejoint','plate_freejoint'):continue
                    j=model.joint(str(name)).id;a=int(model.jnt_qposadr[j]);data.qpos[a:a+int(n)]=z['qpos'][int(adr):int(adr)+int(n)]
        if self.v5 and physics and self.physics_state:
            with np.load(self.physics_state) as z:
                if 'dof_addresses' in z:
                    for name,adr,n in zip(z['joint_names'],z['dof_addresses'],z['dof_sizes']):
                        j=model.joint(str(name)).id;a=int(model.jnt_dofadr[j]);data.qvel[a:a+int(n)]=z['qvel'][int(adr):int(adr)+int(n)]
                data.time=float(z['time']) if 'time' in z else 0.
        else:data.qvel[:]=0
        mujoco.mj_forward(model,data)
        np.savez_compressed(path.with_suffix('.state.npz'),qpos=data.qpos,qvel=data.qvel,time=np.array(data.time))
        write(out/'parameters.json',self.params);write(out/'provenance.json',dict(candidate_id=self.candidate_id(),source_sha256=self.source_hash,xml_sha256=sha(path),resource_closure=closure,mass_policy='Source density preserved: uniform mass s^3, inertia s^5; anisotropic depth uses affine second moments or compiler inferred density; unmeasured',physics=physics))
        return path

    def close(self):
        for entry in self.cache.values():
            for renderer in entry['renderers'].values():renderer.renderer.close()
        self.cache.clear()
