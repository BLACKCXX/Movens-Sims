"""Versioned additive camera-sensitivity experiment; never edits the strict pilot.

Run after head-camera-pilot. Uses only existing bounded fit/scene/evaluation code.
The extra scheduler tests a shared rotation hypothesis, which a same-direction
centroid residual rule cannot detect. Probe evaluations count in the 600 budget.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import cv2
import numpy as np

from movens_sim.head_camera import (LayoutFit,make_scene,observations,render_metrics,
    save_candidate,rest_diagnostic,auxiliary_views,labelled_comparison,load_case_scene,case_outputs)
from movens_sim.pipeline import complete_receipt,receipt_valid
from movens_sim.render import Renderer,video_writer
from movens_sim.util import ROOT,read,write,sha,digest,code_identity,save_rgb,now


def object_loss(metrics,label):
    o=metrics['objects'][label];d=np.hypot(metrics['width'],metrics['height'])
    return 1-o['iou']+.6*o['symmetric_edge_distance_px']/d+.5*o['centroid_error_px']/d+.2*o['area_relative_error']


def run(config,episode,resume=False):
    if episode not in config['allowed_episodes']:raise ValueError('Episode outside authorized pilot')
    parent=ROOT/config['run_root']/f'episode_{episode:06}';out=parent/'mount_refinement';out.mkdir(exist_ok=True)
    parent_identity=read(parent/'run_identity.json')
    if parent_identity['code_hashes']!=code_identity():raise ValueError('Core fit/scene/evaluator changed since strict pilot')
    if not receipt_valid(parent,'render',parent_identity):raise ValueError('Complete strict pilot with valid receipts required')
    identity=dict(version='shared_camera_sensitivity_v1',parent_identity_sha256=digest(parent_identity),script_sha256=sha(__file__),config_sha256=digest(config),selection_sha256=sha(parent/'selection.json'),candidate_index_sha256=sha(parent/'candidate_index.json'),code_hashes=code_identity())
    if (out/'run_identity.json').exists() and read(out/'run_identity.json')!=identity:raise ValueError('Incompatible refinement identity')
    write(out/'run_identity.json',identity);write(out/'resolved_config.json',config)
    selection=read(parent/'selection.json');cases={r['name']:r for r in read(parent/'candidate_index.json')};geometry=read(parent/'geometry_inputs.json');priors=read(parent/'camera_prior.json')['priors'];baseline=read(parent/'batch_final_metrics.json');targets,valid=observations(parent);results=[]
    for variant,sel in selection.items():
        folder=out/variant;folder.mkdir(exist_ok=True);seed=cases[sel['best_strict']]
        if resume and receipt_valid(folder,'experiment',identity):
            results.append(read(folder/'result.json'));print(variant+': verified refinement resume',flush=True);continue
        start=time.monotonic();scene=make_scene(geometry[variant]);scene.set_camera(priors[seed['mount_profile']],seed['fovy_deg']);scene.registry()
        with Renderer(scene.model,640,480) as renderer:save_rgb(folder/'camera_only.png',renderer.rgb(scene.data))
        local=copy.deepcopy(config);local['fit']['max_evaluations']=config['refinement']['max_evaluations']
        fitter=LayoutFit(scene,priors[seed['mount_profile']],seed['fovy_deg'],folder,parent,local,refine=True,resume=resume)
        try:
            if not (resume and (folder/'checkpoint.json').exists()):
                fitter.best.update(read(Path(seed['folder'])/'parameters_optimized.json'));fitter.completed=['xy','orientation'];fitter.evaluate(fitter.best,'strict_seed')
            if (folder/'sensitivity.json').exists():sensitivity=read(folder/'sensitivity.json')
            else:
                original=fitter.best.copy();base_loss=fitter.best_loss
                _,_,base_metrics=render_metrics(scene,targets,valid,config)
                probes=[];best=original.copy();best_loss=base_loss
                for prefix,step in [('mount.d',.005),('mount.r',2.)]:
                    for axis in 'xyz':
                        for direction in (-1,1):
                            fitter.best=original.copy();fitter.best_loss=base_loss
                            proposal=original.copy();proposal[prefix+axis]=direction*step
                            loss=fitter.evaluate(proposal,'sensitivity_'+prefix+axis)
                            # Re-evaluate pixels at the accepted proposal only if it improved.
                            accepted=loss<base_loss-1e-8;both=False;changes={}
                            if accepted:
                                _,_,metrics=render_metrics(scene,targets,valid,config)
                                changes={k:object_loss(base_metrics,k)-object_loss(metrics,k) for k in ('apple','plate')}
                                both=all(value>=.002 for value in changes.values())
                                if loss<best_loss:best=proposal.copy();best_loss=loss
                            probes.append(dict(parameter=prefix+axis,step=direction*step,objective=loss,improved=accepted,both_objects_improved=both,object_loss_improvements=changes))
                triggered=any(p['both_objects_improved'] for p in probes)
                sensitivity=dict(version='shared_camera_sensitivity_v1',triggered=triggered,seed=seed['name'],base_objective=base_loss,probes=probes,rule='At least one <=5mm or <=2deg camera-only perturbation improves each object loss by >=0.002. This tests shared rotational as well as translational residuals; all probes count in the 600 evaluation budget.',review_reason='The two object vertical residuals have opposite signs, consistent with a camera-roll hypothesis; same-direction centroid rule did not trigger.',thresholds_changed=False)
                write(folder/'sensitivity.json',sensitivity)
                fitter.best=best if triggered else original;fitter.best_loss=best_loss if triggered else base_loss;fitter.apply(fitter.best);fitter.checkpoint()
            if sensitivity['triggered']:fitter.run_fit()
            else:fitter.completed=['xy','orientation','mount','xy_refine'];fitter.run_fit()
            metadata=dict(name='bounded_refinement__'+variant,geometry_variant=variant,mount_profile=seed['mount_profile'],fovy_deg=seed['fovy_deg'],refinement=True,folder=str(folder),strict_parent=seed['name'],sensitivity_triggered=sensitivity['triggered'])
            result=save_candidate(scene,folder,parent,config,baseline,metadata)
            fitter.checkpoint(complete=True)
        finally:fitter.close()
        diagnostics=rest_diagnostic(scene,folder,config);auxiliary_views(scene,folder)
        with Renderer(scene.model,640,480) as renderer:frame=renderer.rgb(scene.data)
        video=video_writer(folder/'ego_static.mp4',(frame for _ in range(int(round(config['render']['fps']*config['render']['duration_s'])))),config['render']['fps'])
        write(folder/'ego_static_definition.json',dict(type='fixed_camera_initial_static_3d_simulation',real_capture=False,dynamics_validated=False,video=video,camera='approx_ego_camera',fovy_deg=seed['fovy_deg'],scene='scene/scene_optimized.xml',state='scene/scene_optimized.state.npz'))
        result.update(physical_diagnostic=diagnostics,parent_strict_metrics=seed['metrics']);write(folder/'result.json',result)
        complete_receipt(folder,'experiment',identity,[p for p in folder.rglob('*') if p.is_file() and 'receipts' not in p.parts],time.monotonic()-start)
        print(variant+f": sensitivity={sensitivity['triggered']}, apple IoU={result['metrics']['objects']['apple']['iou']:.4f}, plate IoU={result['metrics']['objects']['plate']['iou']:.4f}, static={result['gates']['static_passed']}, physical={diagnostics['physics_passed']}",flush=True)
        results.append(result)
    reference=cv2.cvtColor(cv2.imread(str(parent/'reference.png')),cv2.COLOR_BGR2RGB);batch=cv2.cvtColor(cv2.imread(str(parent/'inputs/batch_optimized_ego.png')),cv2.COLOR_BGR2RGB);cards=[]
    for result in results:
        folder=Path(result['folder']);variant=result['geometry_variant'];camera_only=cv2.cvtColor(cv2.imread(str(folder/'camera_only.png')),cv2.COLOR_BGR2RGB);optimized=cv2.cvtColor(cv2.imread(str(folder/'optimized_ego.png')),cv2.COLOR_BGR2RGB)
        comparison=labelled_comparison([reference,batch,camera_only,optimized],['Real frame0','batch_v1 final',f"Nominal mount; VFOV {result['fovy_deg']:g}",'Small mount correction + layout'])
        save_rgb(out/(variant+'_comparison.png'),comparison)
        cards.append(f'<h2>{variant}</h2><img src="{variant}_comparison.png"><p>静态判据 {result["gates"]["static_passed"]}；边界 {result["near_bounds"]}；物理 {result["physical_diagnostic"]["physics_passed"]}</p><video controls src="{variant}/ego_static.mp4"></video><video controls src="{variant}/settle_diagnostic.mp4"></video><p><a href="{variant}/sensitivity.json">小扰动依据</a> · <a href="{variant}/overlay.png">轮廓</a> · <a href="{variant}/scene/scene_optimized.xml">XML</a></p>')
    evaluations=sum(read(Path(r['folder'])/'fit_summary.json')['evaluations'] for r in results)
    summary=dict(version='shared_camera_sensitivity_v1',parent=str(parent),results=results,numeric_evaluations=evaluations,total_numeric_evaluations=read(parent/'result.json')['numeric_evaluations']+evaluations,static_passed=any(r['gates']['static_passed'] for r in results),recommended=False,gpt_calls=0,thresholds=config['thresholds'],completed_utc=now())
    write(out/'result.json',summary)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:16px system-ui;background:#181d23;color:#eee;margin:30px}img{width:100%}video{width:48%}a{color:#8cf}</style><h1>有界安装修正 · 独立实验</h1><p>原严格先验结果保留。两组几何各最多 600 次，包含诊断探针；平移范数 ≤20mm，旋转角 ≤5°，FOV/尺寸/桌面固定。评价阈值不变。</p><p><a href="../index.html">严格先验报告</a> · <a href="result.json">完整结果</a></p>'+''.join(cards)+'</html>')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/head_camera_pilot_v1.json');parser.add_argument('--episode',type=int,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    result=run(read(ROOT/args.config),args.episode,args.resume)
    print({k:v for k,v in result.items() if k!='results'},flush=True)
