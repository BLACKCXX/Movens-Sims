"""Fixed real-image evaluation regions; no candidate-dependent pixel exclusion."""
from functools import lru_cache
import cv2
import numpy as np
from ..util import digest

OBJECTIVES=('apple_color_v5','apple_highlight_v5','plate_shading_v5','cloth_coarse_v5','scene_lighting_v5','coverage_v5','geometry_v5','support_v5')
POLICY='cloth_optional_details_v5'


def core(mask,valid,n=5):return cv2.erode(mask.astype(np.uint8),np.ones((n,n),np.uint8)).astype(bool)&valid
def mean(a,w):return float(np.sum(a*w)/max(float(np.sum(w)),1e-8))
def lab(rgb):return cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(float)


def color_membership(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(float)
    # Hue around zero is red; yellow rises smoothly around 15 OpenCV hue units.
    hue=hsv[:,:,0];hue=np.where(hue>150,hue-180,hue)
    return 1/(1+np.exp(np.clip(-(hue-14)/2.5,-30,30)))


def highlight(rgb):
    x=rgb.astype(np.float32)/255.;v=x.max(2);chroma=x.max(2)-x.min(2)
    excess=np.maximum(0.,v-cv2.GaussianBlur(v,(0,0),4))
    # Bright yellow alone is not specularity. Local excess and reduced chroma
    # are both required; these remain soft, uncertain photometric evidence.
    return excess*np.clip(1-chroma/.6,0,1)


def appearance_metrics(rgb,observation):
    real=observation['rgb'];valid=observation['valid'];target=observation['target'];a,b=lab(rgb),lab(real)
    regions={k:core(target[k],valid) for k in ('apple','plate','cloth')}
    apple=regions['apple'];yy,xx=np.where(apple);diam=max(1.,np.sqrt(apple.sum()))
    # A fixed dark, upper fruit-core mask excludes the stalk/cavity from base-color fitting.
    if len(xx):
        upper=np.indices(apple.shape)[0]<(yy.min()+.45*np.ptp(yy));dark=(real.max(2)<70)&upper
        apple=apple&~dark
    truth_y=color_membership(real);pred_y=color_membership(rgb)
    red=apple*(1-truth_y);yellow=apple*truth_y
    err=np.abs(a-b).mean(2)
    red_error=mean(err,red);yellow_error=mean(err,yellow)
    distribution=mean(np.abs(pred_y-truth_y),apple)
    area=abs(mean(pred_y,apple)-mean(truth_y,apple))
    hr=highlight(real)*apple;hp=highlight(rgb)*apple
    def center(w):
        y,x=np.indices(w.shape);return [mean(x,w),mean(y,w)]
    highlight_confidence=float(np.clip(hr.sum()/max(1,apple.sum())/.008,0,1))
    position=float(np.linalg.norm(np.array(center(hp))-center(hr))/diam) if hp.sum()>1e-7 and hr.sum()>1e-7 else 1.
    harea=float(abs(mean(hp,apple)-mean(hr,apple)))
    hmap=mean(np.abs(hp-hr),apple)
    hscore=(min(position,2.)/2+harea*10+hmap*10)/3
    plate=regions['plate'];contours,_=cv2.findContours(target['plate'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    y,x=np.indices(plate.shape);profiles=[];regions_l={};sector=[]
    if contours and len(max(contours,key=cv2.contourArea))>=5:
        (cx,cy),(dx,dy),angle=cv2.fitEllipse(max(contours,key=cv2.contourArea));theta=np.radians(angle)
        u=((x-cx)*np.cos(theta)+(y-cy)*np.sin(theta))/max(dx/2,1)
        v=(-(x-cx)*np.sin(theta)+(y-cy)*np.cos(theta))/max(dy/2,1)
        radius=np.hypot(u,v);az=np.arctan2(v,u)
        for lo,hi,name in [(0,.4,'central'),(.4,.8,'wall'),(.8,1.01,'rim')]:
            region=plate&(radius>=lo)&(radius<hi);regions_l[name]=dict(real=mean(b[:,:,0],region),sim=mean(a[:,:,0],region),pixels=int(region.sum()))
        for i in range(32):
            region=plate&(radius>=i/32)&(radius<(i+1)/32)
            profiles.append(dict(radius=(i+.5)/32,real=mean(b[:,:,0],region),sim=mean(a[:,:,0],region),pixels=int(region.sum())))
        for i in range(8):
            region=plate&(az>=-np.pi+i*np.pi/4)&(az<-np.pi+(i+1)*np.pi/4)
            sector.append(dict(real=mean(b[:,:,0],region),sim=mean(a[:,:,0],region)))
    valid_profile=[p for p in profiles if p['pixels']>0]
    rp=np.array([p['real'] for p in valid_profile]);sp=np.array([p['sim'] for p in valid_profile]);rd=np.diff(rp);sd=np.diff(sp)
    radial=float(np.abs(sp-rp).mean()) if len(rp) else 255.
    gradient=float(np.abs(sd-rd).mean()) if len(rd) else 255.
    region_error=float(np.mean([abs(v['sim']-v['real']) for v in regions_l.values()])) if regions_l else 255.
    sector_error=float(np.mean([abs(v['sim']-v['real']) for v in sector])) if sector else 255.
    boundary_confidence=float(np.clip(np.max(abs(rd)) / 20,0,1)) if len(rd) else 0.
    cloth=regions['cloth'];contrast_r=cv2.GaussianBlur(b[:,:,0].astype(np.float32),(0,0),2)-cv2.GaussianBlur(b[:,:,0].astype(np.float32),(0,0),12)
    contrast_s=cv2.GaussianBlur(a[:,:,0].astype(np.float32),(0,0),2)-cv2.GaussianBlur(a[:,:,0].astype(np.float32),(0,0),12)
    overall={k:mean(err,r) for k,r in regions.items()}
    scores=dict(apple_color_v5=(red_error/255+yellow_error/255+distribution+area)/4,
                apple_highlight_v5=float(hscore*highlight_confidence),
                plate_shading_v5=(radial/255+region_error/255+sector_error/255+boundary_confidence*gradient/32)/4,
                cloth_coarse_v5=overall['cloth']/255)
    scores['scene_lighting_v5']=float(np.mean([overall[k]/255 for k in overall]))
    return dict(version='fixed_real_regions_v5.1',region_fingerprint=digest({k:np.packbits(v).tolist() for k,v in regions.items()}),
        apple_color=dict(red_lab_mae=red_error,yellow_lab_mae=yellow_error,yellow_fraction_real=mean(truth_y,apple),yellow_fraction_sim=mean(pred_y,apple),area_fraction_error=area,spatial_distribution_mae=distribution,pixels=int(apple.sum())),
        apple_highlight=dict(confidence=highlight_confidence,center_real_px=center(hr),center_sim_px=center(hp),position_error_normalized=position,soft_area_error=harea,soft_distribution_mae=hmap,real_weight_sum=float(hr.sum()),sim_weight_sum=float(hp.sum()),interpretation='Soft local bright, reduced-chroma evidence; bright yellow is not an automatic highlight label'),
        plate_shading=dict(radial_luminance_mae=radial,regional_luminance_mae=region_error,sector_luminance_mae=sector_error,gradient_mae=gradient,gradient_max_real=float(max(abs(rd),default=0)),gradient_max_sim=float(max(abs(sd),default=0)),boundary_confidence=boundary_confidence,regions=regions_l,sectors=sector,radial_profile=profiles,geometry_truth=False),
        cloth=dict(lab_mae=overall['cloth'],contrast_real=float(np.std(contrast_r[cloth])) if cloth.any() else 0.,contrast_sim=float(np.std(contrast_s[cloth])) if cloth.any() else 0.,details_policy=POLICY),
        objectives=scores,overall_fixed_lab=overall)


def objective(metrics,objective_id,config):
    if objective_id=='coverage_v5':
        t=config['acceptance'];m=metrics['table']
        return float(metrics['image_loss']+10*(max(0,t['cloth_iou_min']-m['cloth_iou'])+max(0,t['top_cloth_iou_min']-m['top_cloth_iou'])+max(0,m['false_background_on_real_cloth_fraction']-t['false_background_max'])))
    if objective_id in ('geometry_v5','support_v5'):return metrics['image_loss']
    return metrics['v5']['objectives'][objective_id]
