from __future__ import annotations
import subprocess
from pathlib import Path
import cv2
import mujoco
import numpy as np
from .util import save_rgb, video_info, write

class Renderer:
    def __init__(self,model,width=640,height=480):
        self.model=model;self.width=width;self.height=height
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,width);model.vis.global_.offheight=max(model.vis.global_.offheight,height)
        model.vis.quality.offsamples=0
        self.renderer=mujoco.Renderer(model,height=height,width=width)
        self.mesh_copy=model.mesh_vert.copy()
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=[1,1,1,0,0,0];self.option.sitegroup[:]=0;self.option.flexgroup[5]=0
        debug=(model.geom_group==0)&((model.geom_contype|model.geom_conaffinity)!=0)&(model.geom_type!=mujoco.mjtGeom.mjGEOM_PLANE)
        model.geom_group[debug]=3  # Same RGB display convention as the original batch renderer.
        self.option.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT]=False;self.option.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE]=False
        self.option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE]=True
    def __enter__(self):return self
    def __exit__(self,*args):self.renderer.close()
    def update(self,data):
        if self.renderer._gl_context is not None:self.renderer._gl_context.make_current()
        # Model vertex changes need explicit GPU upload; changing an array alone does
        # not update MuJoCo's cached OpenGL mesh buffers.
        if not np.array_equal(self.mesh_copy,self.model.mesh_vert):
            for mesh in range(self.model.nmesh):
                a=int(self.model.mesh_vertadr[mesh]);n=int(self.model.mesh_vertnum[mesh])
                if not np.array_equal(self.mesh_copy[a:a+n],self.model.mesh_vert[a:a+n]):mujoco.mjr_uploadMesh(self.model,self.renderer._mjr_context,mesh)
            self.mesh_copy[:]=self.model.mesh_vert
        self.renderer.update_scene(data,camera='approx_ego_camera',scene_option=self.option)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    def rgb(self,data):
        self.renderer.disable_segmentation_rendering();self.update(data);return self.renderer.render().copy()
    def segmentation(self,data):
        self.renderer.enable_segmentation_rendering();self.update(data);im=self.renderer.render().copy();self.renderer.disable_segmentation_rendering();return im
    def masks(self,data,geom_sets):
        im=self.segmentation(data);return {k:np.isin(im[:,:,0],list(gs))&(im[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) for k,gs in geom_sets.items()}

def overlay(reference,masks,real_masks):
    out=reference.copy()
    for k,color in [('apple',(255,80,0)),('plate',(0,220,255))]:
        for mask,c in [(real_masks[k],(60,255,60)),(masks[k],color)]:
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contours,-1,c,1)
    return out

def video_writer(path,frames,fps=30):
    iterator=iter(frames);first=next(iterator);h,w=first.shape[:2];path=Path(path);temp=path.with_name(path.stem+'.partial.mp4')
    p=subprocess.Popen(['ffmpeg','-v','error','-y','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    count=0
    try:
        p.stdin.write(first.tobytes());count+=1
        for frame in iterator:p.stdin.write(np.ascontiguousarray(frame).tobytes());count+=1
        p.stdin.close();error=p.stderr.read();rc=p.wait()
        if rc:raise RuntimeError('ffmpeg failed: '+error.decode()[:200])
        temp.replace(path)
    except BaseException:
        p.kill();p.wait();raise
    info=video_info(path)
    if info['frames']!=count or info['width']!=w or info['height']!=h or abs(info['fps']-fps)>1e-6:raise ValueError('Encoded video failed frame/FPS/size validation')
    return info

def static_video(scene,run,config,width,height):
    with Renderer(scene.model,width,height) as r:frame=r.rgb(scene.data)
    n=int(round(config['duration_s']*config['fps']));info=video_writer(Path(run)/'ego_static.mp4',(frame for _ in range(n)),config['fps'])
    write(Path(run)/'ego_static_definition.json',dict(type='fixed_camera_initial_static_simulation',real_capture=False,interaction_replay=False,dynamics_validated=False,method='One MuJoCo render of the complete initial 3D scene repeated at the requested FPS, no integration',state='scene/scene_optimized.state.npz',scene='scene/scene_optimized.xml',video=info))
    return info
