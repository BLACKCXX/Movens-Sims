"""Native Phong controls and deterministic object/surface-attached textures."""
from functools import lru_cache
import io
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from .appearance import height_atlas
from ..scene import numbers


def defaults(root):
    p={'apple.color_strength':0.,'apple.color_center':.42,'apple.color_width':.10,
       'apple.red_r':1.,'apple.red_g':.12,'apple.red_b':.035,
       'apple.yellow_r':1.,'apple.yellow_g':.88,'apple.yellow_b':.10,
       'light.key_rx':0.,'light.key_ry':0.,'light.key_strength':1.,
       'cloth.texture_amplitude':0.,'cloth.texture_scale':.08}
    for label in ('apple','plate'):
        b=root.find(f".//body[@name='{label}_root']")
        name=next(g.get('material') for g in b.iter('geom') if g.get('material'))
        mat=root.find(f"./asset/material[@name='{name}']")
        for key,default in [('specular',.5),('shininess',.5)]:p[label+'.'+key]=float(mat.get(key,str(default)))
    return p


def registry(params):
    ranges={
        **{f'apple.{color}_{c}':(0.,1.,'native_rgb') for color in ('red','yellow') for c in 'rgb'},
        'apple.color_strength':(0.,1.,'ratio'),'apple.color_center':(.12,.85,'normalized_mesh_height'),
        'apple.color_width':(.025,.30,'normalized_mesh_height'),
        'apple.specular':(0.,1.,'native_phong'),'apple.shininess':(.005,1.,'native_phong'),
        'plate.specular':(0.,1.,'native_phong'),'plate.shininess':(.005,1.,'native_phong'),
        'light.key_rx':(-60.,60.,'deg'),'light.key_ry':(-60.,60.,'deg'),
        'light.key_strength':(0.,1.25,'ratio'),'cloth.texture_amplitude':(0.,.12,'native_rgb'),
        'cloth.texture_scale':(.015,.25,'m')}
    return [dict(id=k,unit=u,lower=lo,upper=hi,default=params[k],stages=['appearance'],coordinate_frame='Immutable source mesh height / world XY surface UV; shared world light',meaning='Effective MuJoCo native-renderer control; absolute, not cumulative',source='Engineering bounds, not measured',linked_updates=['material','texture','renderer','export']) for k,(lo,hi,u) in ranges.items()]


@lru_cache(maxsize=32)
def recolor(base_bytes,mesh_path,values):
    strength,center,width,*colors=values
    if strength==0:return base_bytes
    img=np.asarray(Image.open(io.BytesIO(base_bytes)).convert('RGB'),np.float32)/255.
    h,w=img.shape[:2];z=height_atlas(mesh_path,w,h)
    mix=1/(1+np.exp(np.clip(-(z-center)/width,-40,40)))
    red,yellow=np.asarray(colors[:3]),np.asarray(colors[3:])
    # Preserve the authored fine luminance modulation, with bounded contrast.
    lum=img.max(2);smooth=gaussian_filter(lum,7)
    detail=np.clip(lum/np.maximum(smooth,.08),.65,1.35)
    color=((1-mix[:,:,None])*red+mix[:,:,None]*yellow)*detail[:,:,None]
    out=np.clip((1-strength)*img+strength*color,0,1)
    # Authored dark upper UV islands are the stalk and recessed cavity, not
    # yellow skin. Preserve their original color instead of painting them.
    protected=(z>.60)&(lum<.45)
    out[protected]=img[protected]
    buf=io.BytesIO();Image.fromarray(np.rint(out*255).astype(np.uint8)).save(buf,format='PNG');return buf.getvalue()


@lru_cache(maxsize=32)
def cloth_texture(gray,amplitude,scale):
    rng=np.random.default_rng(20261108);a=rng.normal(size=(512,512))
    # One-meter periodic surface tile, not a projected photograph or fake folds.
    a=gaussian_filter(a,max(1.,scale*512/3),mode='wrap');a/=max(a.std(),1e-8)
    value=np.clip(gray+amplitude*a,0,1);rgb=np.repeat(value[:,:,None],3,2)
    buf=io.BytesIO();Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(buf,format='PNG');return buf.getvalue()


def apply_material(scene,root,virtual,p):
    for label in ('apple','plate'):
        b=root.find(f".//body[@name='{label}_root']")
        names={g.get('material') for g in b.iter('geom') if g.get('material')}
        for name in names:
            mat=root.find(f"./asset/material[@name='{name}']")
            mat.set('specular',str(p[label+'.specular']));mat.set('shininess',str(p[label+'.shininess']))
            if label!='apple' or p['apple.color_strength']==0:continue
            tex=root.find(f"./asset/texture[@name='{mat.get('texture')}']")
            if tex is None or not tex.get('file'):continue
            base=virtual.get(tex.get('file'))
            if base is None:
                from pathlib import Path
                base=Path(tex.get('file')).read_bytes()
            g=next(g for g in b.iter('geom') if g.get('material')==name and g.get('mesh'))
            mesh=root.find(f"./asset/mesh[@name='{g.get('mesh')}']")
            values=tuple(p[k] for k in ['apple.color_strength','apple.color_center','apple.color_width']+[f'apple.{color}_{c}' for color in ('red','yellow') for c in 'rgb'])
            virtual['v5_apple.png']=recolor(base,mesh.get('file'),values);tex.set('file','v5_apple.png')
            # V5 colors replace legacy material tint; blend preserves the exact v4 endpoint.
            old=np.fromstring(mat.get('rgba','1 1 1 1'),sep=' ')
            old[:3]=old[:3]*(1-p['apple.color_strength'])+p['apple.color_strength'];mat.set('rgba',numbers(old))
    key=root.find(".//light[@name='probe_light']")
    if key is not None:
        rv=np.array([p['light.key_rx'],p['light.key_ry'],0.])
        if np.linalg.norm(rv)>60.+1e-9:raise ValueError('Shared key-light rotation exceeds 60 degrees')
        direction=Rotation.from_rotvec(rv,degrees=True).apply([0.,0.,-1.])
        key.set('dir',numbers(direction));key.set('diffuse',numbers([.8*p['light.key_strength']]*3));key.set('specular',numbers([.3*p['light.key_strength']]*3))
    if p['cloth.texture_amplitude']<=0:return
    virtual['v5_cloth.png']=cloth_texture(p['cloth.gray'],p['cloth.texture_amplitude'],p['cloth.texture_scale'])
    ET.SubElement(root.find('asset'),'texture',name='v5_cloth_tex',type='2d',file='v5_cloth.png')
    ET.SubElement(root.find('asset'),'material',name='v5_cloth_mat',texture='v5_cloth_tex',rgba='1 1 1 1',specular='0',shininess='.1',texrepeat='1 1')
    from pathlib import Path
    for node in root.findall('.//geom'):
        if not (node.get('name','').startswith('frozen_') or node.get('name')=='cloth_continuation'):continue
        mesh=root.find(f"./asset/mesh[@name='{node.get('mesh')}']")
        if mesh is None:continue
        content=virtual.get(mesh.get('file'))
        content=content.decode() if content is not None else Path(mesh.get('file')).read_text()
        vertices=[list(map(float,s.split()[1:4])) for s in content.splitlines() if s.startswith('v ')]
        # Source frozen cloth and continuation are world-aligned; use their world XY.
        uv=['vt '+numbers(v[:2]) for v in vertices];lines=[]
        for s in content.splitlines():
            if s.startswith('vt '):continue
            if s.startswith('f '):s='f '+' '.join(f'{x.split("/")[0]}/{x.split("/")[0]}' for x in s.split()[1:])
            lines.append(s)
        filename='v5_uv_'+mesh.get('name')+'.obj';virtual[filename]=('\n'.join(lines+uv)+'\n').encode();mesh.set('file',filename)
        node.set('material','v5_cloth_mat');node.attrib.pop('rgba',None)
    for flex in root.findall('./deformable/flex'):
        if int(flex.get('group','0'))==5:continue
        xyz=np.fromstring(flex.get('vertex',''),sep=' ').reshape(-1,3)
        if flex.get('body'):
            positions=[np.fromstring(root.find(f".//body[@name='{name}']").get('pos','0 0 0'),sep=' ') for name in flex.get('body').split()]
            xyz=xyz+np.asarray(positions)
        flex.set('texcoord',numbers(xyz[:,:2].ravel()));flex.set('material','v5_cloth_mat');flex.attrib.pop('rgba',None)
