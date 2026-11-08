"""Bake the apple's upper-yellow hypothesis in mesh UVs, not atlas image rows."""
from functools import lru_cache
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


@lru_cache(maxsize=2)
def height_atlas(mesh_path,width,height):
    vertices=[];uv=[];faces=[]
    for line in Path(mesh_path).read_text().splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='v':vertices.append([float(x) for x in parts[1:4]])
        elif parts[0]=='vt':uv.append([float(x) for x in parts[1:3]])
        elif parts[0]=='f':
            refs=[p.split('/') for p in parts[1:]]
            if all(len(r)>1 and r[1] for r in refs):
                for i in range(1,len(refs)-1):faces.append([(int(r[0])-1,int(r[1])-1) for r in (refs[0],refs[i],refs[i+1])])
    v=np.asarray(vertices);uv=np.asarray(uv);result=np.full((height,width),.5,np.float32)
    if not faces:raise ValueError('Height-mapped apple color requires authored mesh UV coordinates')
    z=(v[:,2]-v[:,2].min())/max(1e-12,np.ptp(v[:,2]));uv=uv*np.array([width-1,-(height-1)])+np.array([0,height-1])
    for face in faces:
        vi,ti=np.asarray(face).T;p=uv[ti];lo=np.maximum(np.floor(p.min(0)).astype(int),0);hi=np.minimum(np.ceil(p.max(0)).astype(int),[width-1,height-1])
        if np.any(hi<lo):continue
        a=p[1]-p[0];b=p[2]-p[0];det=a[0]*b[1]-a[1]*b[0]
        if abs(det)<1e-8:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];dx=xx-p[0,0];dy=yy-p[0,1]
        u=(dx*b[1]-dy*b[0])/det;w=(a[0]*dy-a[1]*dx)/det;inside=(u>=-.01)&(w>=-.01)&(u+w<=1.01)
        values=(1-u-w)*z[vi[0]]+u*z[vi[1]]+w*z[vi[2]];roi=result[lo[1]:hi[1]+1,lo[0]:hi[0]+1];roi[inside]=np.clip(values[inside],0,1)
    return result


@lru_cache(maxsize=16)
def apple_texture(texture_path,mesh_path,mix):
    img=np.asarray(Image.open(texture_path).convert('RGB'),np.float32)/255.;h,w=img.shape[:2]
    height=height_atlas(mesh_path,w,h)
    # The red lower half stays attached to the source mesh. UV seams retain
    # source texture detail; this is an explicit appearance hypothesis only.
    weight=mix*np.clip((height-.2)/.55,0,1)[:,:,None]
    lum=img.max(2,keepdims=True);yellow=lum*np.array([1.,.88,.18],np.float32)
    rgb=np.clip((1-weight)*img+weight*yellow,0,1);buf=io.BytesIO();Image.fromarray((rgb*255).astype(np.uint8)).save(buf,format='PNG')
    return buf.getvalue()
