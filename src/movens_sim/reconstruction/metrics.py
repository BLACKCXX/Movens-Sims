"""Versioned, object-balanced geometry and appearance objectives."""
from __future__ import annotations
import cv2
import numpy as np
from ..evaluate import whole_scene_metrics
from ..vision import describe


def measure(rgb, masks, observation, config):
    target,valid,reference=observation['target'],observation['valid'],observation['rgb']
    result=whole_scene_metrics(masks,target,valid,config)
    total=0.
    for label in ('apple','plate'):
        obj=result['objects'][label];a=masks[label]&valid;b=target[label]&valid
        features=[]
        for mask in (a,b):
            try:features.append(describe(mask))
            except (ValueError,cv2.error):features.append(None)
        if all(x is not None for x in features):
            pred,truth=features;ax=np.sort(pred['ellipse'][1]) if pred['ellipse'] else np.ones(2);bx=np.sort(truth['ellipse'][1]) if truth['ellipse'] else np.ones(2)
            axes=np.abs(np.log(np.maximum(ax,1)/np.maximum(bx,1)))
            obj.update(projected_bbox_xyxy_px=pred['bbox_xyxy_px'],target_bbox_xyxy_px=truth['bbox_xyxy_px'],projected_axes_px=ax.tolist(),target_axes_px=bx.tolist(),axis_ratio_error=float(abs(ax[0]/max(ax[1],1)-bx[0]/max(bx[1],1))))
        else:axes=np.ones(2)*5
        diameter=max(1.,float(np.sqrt(b.sum())))
        area_ratio=float(a.sum()/max(1,b.sum()));obj['signed_area_relative_error']=area_ratio-1
        obj['centroid_error_normalized']=obj['centroid_error_px']/diameter;obj['edge_error_normalized']=obj['symmetric_edge_distance_px']/diameter
        obj['objective']=float(.30*(1-obj['iou'])+.20*obj['edge_error_normalized']+.15*obj['centroid_error_normalized']+.15*abs(np.log(max(area_ratio,1e-5)))+.20*axes.mean())
        total+=.5*obj['objective']
    table=result['table'];total+=.1*(1-table['cloth_iou'])+.1*(1-table['top_cloth_iou'])
    result['image_loss']=float(total);result['evaluator_version']='balanced_geometry_v3';result['mask_version']=observation['version']
    # Inner edges are photometric, not independent geometry truth. Report only.
    inner=cv2.erode(target['plate'].astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)&valid
    er=cv2.Canny(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),30,90).astype(bool)&inner
    ep=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),30,90).astype(bool)&inner
    distance=None
    if er.any() and ep.any():
        dr=cv2.distanceTransform((~er).astype(np.uint8),cv2.DIST_L2,5);dp=cv2.distanceTransform((~ep).astype(np.uint8),cv2.DIST_L2,5)
        distance=float((dr[ep].mean()+dp[er].mean())/2)
    result['plate_inner_structure']=dict(real_edge_pixels=int(er.sum()),render_edge_pixels=int(ep.sum()),symmetric_edge_distance_px=distance,confidence='low',used_for_acceptance=False,note='RGB edges confounded by illumination; no reliable annotated inner ring. Shape changes need explicit diagnostic evidence.')
    color={}
    for label in ('apple','plate','cloth'):
        region=cv2.erode(target[label].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&valid&masks[label]
        if region.any():
            lab_a=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(float);lab_b=cv2.cvtColor(reference,cv2.COLOR_RGB2LAB).astype(float)
            color[label]=dict(lab_mean_absolute_error=float(np.abs(lab_a[region]-lab_b[region]).mean()),rgb_mean_absolute_error=float(np.abs(rgb.astype(float)[region]-reference.astype(float)[region]).mean()),pixels=int(region.sum()))
        else:color[label]=dict(lab_mean_absolute_error=255.,rgb_mean_absolute_error=255.,pixels=0)
    result['appearance']=color;result['appearance_loss']=float(np.mean([x['lab_mean_absolute_error'] for x in color.values()])/255.)
    return result


def gate(metrics, geometry, baseline, config):
    t=config['acceptance'];fail=[]
    for label,obj in metrics['objects'].items():
        checks={'iou':obj['iou']>=t['iou_min'],'center':obj['centroid_error_px']<=t['center_px_max'],'edge':obj['symmetric_edge_distance_px']<=t['edge_px_max'],'area':obj['area_relative_error']<=t['area_relative_max'],'axes':max(obj['axis_relative_errors'])<=t['axes_relative_max']}
        fail.extend(label+'.'+k for k,v in checks.items() if not v)
    if baseline:
        for key,drop in [('cloth_iou',t.get('cloth_iou_degradation_max',.01)),('top_cloth_iou',t.get('top_cloth_iou_degradation_max',.02))]:
            if metrics['table'][key]<baseline['table'][key]-drop:fail.append('table.'+key+'.degraded')
        if metrics['table']['false_background_on_real_cloth_fraction']>max(t.get('false_background_max',.01),baseline['table']['false_background_on_real_cloth_fraction']+t.get('false_background_degradation_max',.005)):fail.append('table.false_background')
    if 'cloth_iou_min' in t:
        for key in ('cloth_iou','top_cloth_iou'):
            if metrics['table'][key]<t[key+'_min']:fail.append('table.'+key+'.absolute')
        if metrics['table']['false_background_on_real_cloth_fraction']>t['false_background_max']:fail.append('table.false_background.absolute')
    return dict(numeric_passed=not fail,initial_support_passed=geometry['physical_valid'],geometry_numeric_passed=not fail and geometry['physical_valid'],failures=fail+geometry['failures'])


def nondegrading(current, previous, config):
    t=config['acceptance'];reasons=[]
    for label,a in current['objects'].items():
        b=previous['objects'][label]
        if a['iou']<b['iou']-t['iou_degradation_max']:reasons.append(label+'.iou_degradation')
        if a['centroid_error_px']>b['centroid_error_px']+t['center_degradation_px']:reasons.append(label+'.center_degradation')
        if a['symmetric_edge_distance_px']>b['symmetric_edge_distance_px']+t['edge_degradation_px']:reasons.append(label+'.edge_degradation')
    for key,drop in [('cloth_iou',t.get('cloth_iou_degradation_max',.01)),('top_cloth_iou',t.get('top_cloth_iou_degradation_max',.02))]:
        if current['table'][key]<previous['table'][key]-drop:reasons.append('table.'+key)
    return reasons
