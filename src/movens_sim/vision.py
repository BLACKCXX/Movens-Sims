"""Auditable color-component masks for these simple, unoccluded initial frames."""
from pathlib import Path
import io
import json
import subprocess
import cv2
import numpy as np
from .util import read, save_rgb, write

def frame_at(video,index=0):
    from PIL import Image
    # System FFmpeg/libdav1d decodes the raw AV1 source; OpenCV's bundled FFmpeg does not.
    p=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{int(index)})','-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True,check=True)
    if not p.stdout:raise ValueError(f'Cannot read frame {index} of {video}')
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-of','json',str(video)],capture_output=True,text=True,check=True)
    a,b=json.loads(probe.stdout)['streams'][0]['avg_frame_rate'].split('/')
    return np.array(Image.open(io.BytesIO(p.stdout)).convert('RGB')),float(a)/float(b)

def largest(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if n<2:raise ValueError('No segmented component; manual pixel-mask review required')
    j=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);out=(labels==j).astype(np.uint8)
    contours,_=cv2.findContours(out,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,contours,-1,1,-1);return out.astype(bool)

def segment(rgb):
    h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    region=np.zeros((h,w),bool);region[int(.12*h):,:]=True
    apple=largest(region&(((hsv[:,:,0]<38)|(hsv[:,:,0]>170))&(hsv[:,:,1]>95)&(hsv[:,:,2]>75)))
    plate_region=region.copy();plate_region[:,:int(.25*w)]=False
    plate=largest(plate_region&(hsv[:,:,1]<55)&(hsv[:,:,2]>155))
    for label,mask in [('apple',apple),('plate',plate)]:
        fraction=mask.mean()
        if not .002<fraction<.35:raise ValueError(f'{label} mask area implausible: {fraction}')
    return dict(apple=apple,plate=plate)

def describe(mask):
    y,x=np.where(mask);contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);c=max(contours,key=cv2.contourArea)
    ellipse=cv2.fitEllipse(c) if len(c)>=5 else None
    visible=1.0;cropped=bool(np.any(y==0)|np.any(y==mask.shape[0]-1)|np.any(x==0)|np.any(x==mask.shape[1]-1))
    if ellipse:
        area=np.pi*ellipse[1][0]*ellipse[1][1]/4
        visible=min(1.0,float(len(x)/area))
    return dict(pixels=int(len(x)),centroid_xy_px=[float(x.mean()),float(y.mean())],bbox_xyxy_px=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],ellipse=ellipse,visible_fraction_estimate=visible,border_cropped=cropped,occluded_by_robot=False,visible_fraction_note='Ellipse-based image-border visibility estimate; does not measure hidden 3D surface area')

def prepare_observation(imported,run):
    imported,run=Path(imported),Path(run);rgb,fps=frame_at(imported/'real_ego.mp4');save_rgb(run/'reference.png',rgb)
    masks=segment(rgb);folder=run/'masks';folder.mkdir(exist_ok=True)
    valid=np.ones(rgb.shape[:2],bool)
    # No hand/arm intersects either object in the reviewed first frames of the frozen batch.
    # Future scenes with occlusion require an explicit reviewed mask; never use a GPT box.
    occ=np.zeros_like(valid)
    np.savez_compressed(folder/'observations.npz',**masks,valid=valid,occlusion=occ)
    for k,mask in masks.items():
        cv2.imwrite(str(folder/(k+'.png')),mask.astype(np.uint8)*255)
        cv2.imwrite(str(folder/(k+'_edge.png')),cv2.Canny(mask.astype(np.uint8)*255,80,180))
    cv2.imwrite(str(folder/'valid.png'),valid.astype(np.uint8)*255);cv2.imwrite(str(folder/'occlusion.png'),occ.astype(np.uint8)*255)
    preview=rgb.copy()
    for k,c in [('apple',(255,80,0)),('plate',(0,255,120))]:
        cs,_=cv2.findContours(masks[k].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,cs,-1,c,2)
    save_rgb(run/'reference_masks_review.png',preview)
    temporal=[]
    for idx in (1,3,6):
        frame,_=frame_at(imported/'real_ego.mp4',idx)
        try:
            other=segment(frame);changes={}
            for k in masks:
                a,b=masks[k],other[k];changes[k]=dict(iou=float((a&b).sum()/max(1,(a|b).sum())),centroid_shift_px=float(np.linalg.norm(np.array(describe(a)['centroid_xy_px'])-describe(b)['centroid_xy_px'])))
            temporal.append(dict(frame_index=idx,time_s=idx/fps,changes=changes))
        except ValueError as e:temporal.append(dict(frame_index=idx,time_s=idx/fps,error=str(e)))
    write(run/'reference_selection.json',dict(frame_index=0,timestamp_s=0,fps=fps,width=rgb.shape[1],height=rgb.shape[0],selection='First decoded original real ego frame; no object manipulation; objects clear and hands absent',method_version='hsv_connected_components_v1',segmentation='Pixel threshold, connected component and hole fill; manually inspected contour image',objects={k:describe(v) for k,v in masks.items()},valid_pixels=int(valid.sum()),robot_occlusion_pixels=0,temporal_initial_window=temporal,fit_frame_indices=[0],heldout_image_fitting=False,ambiguity='Single view with no metric depth. Nearby frames only check early-window image stability; camera/objects may move slightly. No claim of triangulation or new 3D parallax.',camera_model='MuJoCo centered pinhole. Original 640x480, no crop; fitting downsampled proportionally; distortion not calibrated.'))
    return rgb,masks,valid

def load_observation(run,width=None,height=None):
    run=Path(run);rgb=cv2.cvtColor(cv2.imread(str(run/'reference.png')),cv2.COLOR_BGR2RGB)
    with np.load(run/'masks/observations.npz') as z:masks={k:z[k].copy() for k in ('apple','plate')};valid=z['valid'].copy()
    if width is not None and rgb.shape[1]!=width:
        rgb=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
        masks={k:cv2.resize(v.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool) for k,v in masks.items()};valid=cv2.resize(valid.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)
    return rgb,masks,valid
