"""Fresh, disk-only diagnostic views and photometric plots."""
from pathlib import Path
import cv2
import numpy as np
from .diagnostics import load_state,spend_render,finish_render
from ..render import Renderer
from ..util import save_rgb,write


def color_pipeline_probe(budget=None):
    """Measure the installed native texture transfer, rather than assume gamma."""
    import io
    import mujoco
    from PIL import Image
    colors=[(40,80,120),(128,128,128),(220,180,60)];assets={};nodes=[]
    for i,color in enumerate(colors):
        buf=io.BytesIO();Image.fromarray(np.tile(np.array(color,np.uint8),(16,16,1))).save(buf,format='PNG');assets[f'c{i}.png']=buf.getvalue()
        nodes.append(f'<texture name="t{i}" type="2d" file="c{i}.png"/><material name="m{i}" texture="t{i}" specular="0" rgba="1 1 1 1"/>')
    geoms=''.join(f'<geom type="box" name="c{i}" pos="{(i-1)*.5} 0 0" size=".18 .18 .02" material="m{i}" group="1" contype="0" conaffinity="0"/>' for i in range(3))
    xml='<mujoco><visual><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual><asset>'+''.join(nodes)+'</asset><worldbody><camera name="approx_ego_camera" pos="0 0 2" quat="1 0 0 0" fovy="42"/>'+geoms+'</worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml,assets);d=mujoco.MjData(m);mujoco.mj_forward(m,d);events=spend_render(budget,'render_diagnostic',dict(kind='native_color_pipeline_probe'))
    with Renderer(m) as r:rgb=r.rgb(d)
    finish_render(budget,events,dict(rgb=1,view='native_color_pipeline_probe'))
    f=240/np.tan(np.radians(21));actual=[rgb[240,round(320+f*(i-1)*.5/1.98)].tolist() for i in range(3)]
    return dict(source_rgb=colors,rendered_rgb=actual,maximum_channel_error=int(np.max(np.abs(np.array(colors)-actual))),native_passthrough=bool(np.max(np.abs(np.array(colors)-actual))<=1),extra_gamma_applied=False,independent_object_exposure=False)


def extra_views(xml,folder,budget,pool='render_export'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);m,d=load_state(xml)
    for name in ('specular_off','neutral_all_lights_control'):
        m.mat_specular[:]=0
        if name=='neutral_all_lights_control':
            m.light_diffuse[:]=0;m.light_specular[:]=0;m.light_ambient[:]=0
            m.vis.headlight.ambient[:]=.55;m.vis.headlight.diffuse[:]=.35;m.vis.headlight.specular[:]=0
        events=spend_render(budget,pool,dict(kind='final_v5_auxiliary',view=name))
        with Renderer(m) as r:rgb=r.rgb(d)
        save_rgb(folder/(name+'.png'),rgb);finish_render(budget,events,dict(rgb=1,view=name))
    events=spend_render(budget,pool,dict(kind='final_v5_depth_normals'))
    with Renderer(m) as r:
        r.update(d);r.renderer.enable_depth_rendering();depth=r.renderer.render().copy()
    cam=m.camera('approx_ego_camera').id;f=240/np.tan(np.radians(m.cam_fovy[cam]/2));y,x=np.indices(depth.shape)
    pos=np.stack([(x-320)*depth/f,-(y-240)*depth/f,-depth],2);normal=np.cross(np.gradient(pos,axis=1),np.gradient(pos,axis=0));normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-9)
    save_rgb(folder/'depth_normals.png',np.clip((normal+1)*127.5,0,255).astype(np.uint8));finish_render(budget,events,dict(depth=1,view='depth_normals'))
    write(folder/'v5_diagnostic_views.json',dict(fresh_xml_load=True,normal_kind='Camera-space depth-derived surface normals; source authored-normal statistics reported separately',real_geometry_verified=False,neutral_control='All explicit lights and all specular disabled, only declared headlight ambient/diffuse'))


def plots(metrics,folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p=metrics['v5']['plate_shading']['radial_profile']
    fig,ax=plt.subplots(figsize=(8,4));ax.plot([v['radius'] for v in p],[v['real'] for v in p],label='REAL');ax.plot([v['radius'] for v in p],[v['sim'] for v in p],label='SIM');ax.set(xlabel='Fixed reference ellipse normalized radius',ylabel='OpenCV L (0..255)',title='Plate photometric profile, not real geometry');ax.legend();fig.tight_layout();fig.savefig(folder/'plate_luminance_curve.png',dpi=140);plt.close(fig)
    write(folder/'plate_luminance_profile.json',metrics['v5']['plate_shading']);write(folder/'apple_highlight.json',metrics['v5']['apple_highlight']);write(folder/'apple_color.json',metrics['v5']['apple_color'])
