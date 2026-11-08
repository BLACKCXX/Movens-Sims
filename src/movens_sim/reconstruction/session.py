"""Durable session used by Python, optimizer and MCP tools."""
from __future__ import annotations
import re
import time
from pathlib import Path
import numpy as np
import copy
from contextlib import nullcontext
from scipy.spatial.transform import Rotation

from .backend import SceneBackend,SceneError
from .budget import BudgetManager,BudgetExceeded
from .optimize import Evaluator,fit_geometry
from .prepare import load_observation
from ..util import ROOT,read,write,digest,append,sha
from ..scene import quat_from_matrix
from .metrics import nondegrading


class ReconstructionSession:
    def __init__(self,run,config):
        from .modes import install_initial_xml_guard
        install_initial_xml_guard(config)
        self.run=Path(run);self.config=config
        self.validate_inputs()
        task=config['task_group_id']
        budget_id=f"{task}/episode_{read(self.run/'source_manifest.json')['episode']:06}" if config.get('batch',{}).get('enabled') else task
        self.budget=BudgetManager(ROOT/'runs/_task_budgets'/f'{task}.sqlite',budget_id,config['budget'])
        with self.budget.computation(dict(kind='v5_session_compile')) if config.get('workflow') in ('terra_closed_loop_v5','ablation_no_terra_v5') else nullcontext():
            self.scene=SceneBackend(self.run/'inputs/template_static/scene.xml',self.run/'inputs/template_physics/scene.xml',self.run/'inputs/physics_state_by_name.npz',read(self.run/'initial_parameters.json'),read(self.run/'camera.json'),support_top_only=config.get('initialization',{}).get('mode')=='fk_h0',initialize_supported=config.get('initialization',{}).get('mode')=='fk_h0',config=config)
        if self.scene.v5:
            # The constructor captures environment geometry after its first
            # provisional compile. Never reuse that pre-capture model when a
            # warm start restores the same camera and nonzero cloth extents.
            for cached in self.scene.cache.values():
                for renderer in cached['renderers'].values():renderer.renderer.close()
            self.scene.cache.clear()
        self.scene.allow_camera_revision=config['camera'].get('allow_revision',True)
        if self.scene.initial_support_seed and not (self.run/'support_search_seed.json').exists():
            write(self.run/'support_search_seed.json',dict(method='geometric_feasible_XY_seed_v1',updates=self.scene.initial_support_seed,immutable_fk_preserved=True,no_table_or_scale_change=True))
        self.observation=load_observation(run)
        if self.scene.classified and not self.scene.v5:self.observation['version']+='_render_'+self.scene.render_contract[:12]
        self.state_path=self.run/'session.json'
        self.state=read(self.state_path) if self.state_path.exists() else dict(stage='geometry',shape_enabled=False,plate_tilt_enabled=False,rounds=[],operations=[],geometry_review_passed=False)
        if self.state.get('camera_override'):self.scene.locked_camera=self.state['camera_override']
        if 'params' in self.state:self.scene.apply(self.state['params'],internal=True)
        self.scene.stage=self.state['stage'];self.scene.shape_enabled=self.state['shape_enabled'];self.scene.plate_tilt_enabled=self.state['plate_tilt_enabled']
        self.evaluator=Evaluator(self.scene,self.observation,self.budget,self.run,config)
        self.save()

    def validate_inputs(self):
        manifest=self.run/'source_manifest.json'
        if manifest.exists():
            for name,expected in read(manifest).get('inputs',{}).items():
                if not (self.run/name).is_file() or sha(self.run/name)!=expected:raise SceneError('input_identity_mismatch',name)

    def save(self):
        if self.scene.classified and self.state.get('candidate_id')!=self.scene.candidate_id():
            from .review_state import invalidate
            invalidate(self.state,self.state.get('params',{}),self.scene.params,self.state.get('camera_override'),self.scene.locked_camera)
            if self.scene.v5:
                from .review_state import dependencies
                for issue in self.state.get('issues',{}).values():
                    current=dependencies(issue['category'],self.scene.params,self.scene.locked_camera,self.observation['version'])
                    if issue.get('status')=='resolved' and issue.get('dependency_hash')!=current:
                        issue.update(status='open',invalidation_reason='scene_dependency_changed')
            self.state['scene_revision']=self.state.get('scene_revision',0)+1
        self.state.update(params=self.scene.params.copy(),camera_override=self.scene.locked_camera,candidate_id=self.scene.candidate_id(),stage=self.scene.stage,shape_enabled=self.scene.shape_enabled,plate_tilt_enabled=getattr(self.scene,'plate_tilt_enabled',False))
        write(self.state_path,self.state)

    def inspect_scene(self):
        result=self.scene.inspect();result.update(budget={k:v for k,v in self.budget.summary().items() if k!='events'},camera_provenance=read(self.run/'camera_prior.json'),capabilities=self.capabilities())
        return result

    def capabilities(self):
        if self.scene.v5:
            from .v5_tools import capabilities
            return capabilities(self)
        if self.scene.classified:
            from .operations import capabilities
            return capabilities(self)
        return dict(object_absolute_scale=True,joint_xy_size_fit=True,local_saved_triangle_support=True,shape_parameters=['plate.depth_ratio'],plate_tilt='evidence_required',arbitrary_mesh_deformation=False,renderer_material='Phong specular/shininess; no ineffective PBR roughness',camera_policy='LOCKED for this batch; no camera/FOV edits' if self.config['camera'].get('allow_revision') is False else 'fixed by default; shared-residual bounded independent hypothesis only',terminal_read_only=self.scene.stage=='final_readonly')

    def check_id(self,candidate_id):
        self.validate_inputs()
        if candidate_id!=self.scene.candidate_id():raise SceneError('stale_candidate','Expected current candidate_id')

    def propose_or_apply_object_update(self,updates,candidate_id,apply=False):
        self.check_id(candidate_id)
        if self.scene.stage!='geometry':raise SceneError('wrong_stage','object update')
        if self.scene.classified:
            from .operations import current_evaluation,allowed_parameters
            if set(updates)-set(allowed_parameters(self,'objects')):raise SceneError('invalid_object_parameter','Use update_environment for table/cloth geometry')
            current_evaluation(self,'geometry')
        self.scene.validate(updates,'geometry');old=self.scene.params.copy()
        try:
            with self.budget.computation(dict(kind='tool_physical_trial')):
                self.scene.apply(updates);inspection=self.scene.inspect()
            if apply:
                # A proposal alone is never accepted without the common evaluator.
                self.evaluator.stage='geometry';self.evaluator.pool='render_repair' if self.state['rounds'] else 'render_initial'
                loss,metrics=self.evaluator.evaluate(self.scene.params.copy(),'tool_object_candidate',640,480)
                inspection.update(metrics=metrics,loss=loss,accepted=self.scene.params!=old)
                self.save()
            return inspection
        finally:
            if not apply:self.scene.apply(old,internal=True)

    def fit_geometry(self,parameter_ids=None,candidate_id=None,evidence='',confidence=0.):
        if candidate_id:self.check_id(candidate_id)
        if self.scene.stage!='geometry':raise SceneError('wrong_stage','fit_geometry')
        if self.scene.classified:
            from .operations import current_evaluation
            current_evaluation(self,'geometry')
            if parameter_ids is None:parameter_ids=['apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale']
        if parameter_ids is not None:
            allowed={p['id'] for p in self.scene.registry() if 'geometry' in p['stages']}
            if not parameter_ids or len(set(parameter_ids))!=len(parameter_ids) or set(parameter_ids)-allowed:raise SceneError('invalid_parameter_subset','Need distinct supported geometry parameter IDs')
            if 'plate.depth_ratio' in parameter_ids and not self.scene.shape_enabled:raise SceneError('shape_evidence_required','Use update_shape_or_asset to authorize the supported profile with evidence before fitting it')
        if parameter_ids and any(k in ('plate.rx','plate.ry') for k in parameter_ids) and not self.scene.plate_tilt_enabled:
            if not evidence or not np.isfinite(confidence) or not .7<=confidence<=1.:raise SceneError('orientation_evidence_required','Plate tilt fitting needs explicit evidence and finite confidence in [0.7,1] before a fit round is reserved')
            self.scene.plate_tilt_enabled=True
            self.state['operations'].append(dict(tool='fit_geometry_orientation_permission',evidence=evidence,confidence=confidence))
            self.save()
        summary=self.budget.summary();pending=[e for e in summary['events'] if e['kind']=='geometry_rounds' and e['status']=='reserved']
        index=pending[-1]['detail']['round_index'] if pending else int(summary['used'].get('geometry_rounds',0))
        try:
            with self.budget.computation(dict(kind='complete_geometry_round',round_index=index),whole_task_slice=True):
                environment_fit=self.scene.classified and parameter_ids and all(k.startswith('table.') or k=='cloth.coverage' for k in parameter_ids)
                if environment_fit:
                    from .optimize import environment_hypotheses
                    self.evaluator.stage='geometry';self.evaluator.pool='render_repair';self.evaluator.objective_mode='coverage_feasibility';environment_hypotheses(self.evaluator)
                if self.scene.classified and parameter_ids and any(k in ('plate.rx','plate.ry') for k in parameter_ids):
                    from .optimize import plate_orientation_hypotheses
                    self.evaluator.stage='geometry';self.evaluator.pool='render_repair';plate_orientation_hypotheses(self.evaluator)
                result=fit_geometry(self.evaluator,parameter_ids,index,pending[-1]['id'] if pending else None)
        except BudgetExceeded as exc:
            best=self.evaluator.best.get('geometry_640x480')
            if not best:raise
            result=dict(round_index=index,parameter_ids=parameter_ids,stop_reason=str(exc),best_candidate_id=best['candidate_id'],best_loss=best['loss'])
        if self.scene.classified and getattr(self.evaluator,'objective_mode',None):
            self.evaluator.objective_mode=None
            for key,best in self.evaluator.best.items():
                if key.startswith('geometry_'):best['loss']=best['metrics']['image_loss']+.002*sum((np.log(best['params'][label+'.scale'])/np.log(2))**2 for label in ('apple','plate'))
            self.evaluator.checkpoint();result['selection_objective']='Absolute coverage feasibility plus the common image objective'
        self.state['rounds'].append(result);self.save();return result

    def update_shape_or_asset(self,updates,evidence,confidence,candidate_id):
        self.check_id(candidate_id)
        if self.scene.stage!='geometry':raise SceneError('wrong_stage','shape')
        if not evidence or not np.isfinite(confidence) or not .7<=confidence<=1.:raise SceneError('insufficient_shape_evidence','Need explicit residual evidence with finite confidence in [0.7,1]')
        supported={'plate.depth_ratio','plate.rx','plate.ry'}
        if set(updates)-supported:raise SceneError('capability_missing','Only plate depth and evidence-backed small plate tilt are supported')
        self.scene.shape_enabled=True;self.scene.plate_tilt_enabled=True
        try:
            result=self.propose_or_apply_object_update(updates,candidate_id,apply=True)
        except BaseException:
            self.scene.shape_enabled=self.state['shape_enabled'];self.scene.plate_tilt_enabled=self.state['plate_tilt_enabled'];raise
        self.state['operations'].append(dict(tool='update_shape_or_asset',evidence=evidence,confidence=confidence));self.save();return result

    def update_appearance(self,updates,candidate_id):
        self.check_id(candidate_id)
        if self.scene.stage!='appearance' or (not self.scene.classified and not self.state['geometry_review_passed']):raise SceneError('wrong_stage','Valid Terra geometry review required')
        self.scene.validate(updates,'appearance');params={**self.scene.params,**updates};self.evaluator.stage='appearance';self.evaluator.pool='render_appearance'
        loss,metrics=self.evaluator.evaluate(params,'tool_appearance',640,480);self.save();return dict(loss=loss,metrics=metrics,candidate_id=self.scene.candidate_id())

    def review_camera_hypothesis(self,evidence,confidence,candidate_id):
        """Independent bounded hypotheses, only for common unexplained projection error."""
        if self.scene.classified:
            from .operations import camera_hypotheses
            return camera_hypotheses(self,evidence,confidence,candidate_id)
        self.check_id(candidate_id)
        if self.config['camera'].get('allow_revision') is False:raise SceneError('camera_locked_for_batch','This batch fixes the selected Mount prior and FOV')
        if self.scene.stage!='geometry' or not self.state['rounds']:raise SceneError('wrong_stage','Object fitting must precede camera review')
        base=self.evaluator.best['geometry_640x480'];objects=base['metrics']['objects'];settings=self.config['camera']['conditional_revision']
        residuals=[np.asarray(objects[k]['centroid_residual_xy_px']) for k in ('apple','plate')]
        shared_center=all(np.linalg.norm(v)>=settings['minimum_shared_center_error_px'] for v in residuals) and float(residuals[0]@residuals[1]/max(1e-12,np.linalg.norm(residuals[0])*np.linalg.norm(residuals[1])))>=settings['shared_centroid_cosine_min']
        sizes=[objects[k]['signed_area_relative_error'] for k in ('apple','plate')]
        shared_size=min(abs(x) for x in sizes)>.05 and sizes[0]*sizes[1]>0
        global_error=base['metrics']['table']['top_cloth_iou']<.9 or base['metrics']['table']['false_background_on_real_cloth_fraction']>.01
        if not np.isfinite(confidence) or not .7<=confidence<=1. or not evidence or not global_error or not(shared_center or shared_size):raise SceneError('camera_evidence_insufficient','Need finite confidence and two-object common residual plus global discrepancy after joint fitting')
        round_event=self.budget.reserve('geometry_rounds',detail=dict(round_index=int(self.budget.summary()['used'].get('geometry_rounds',0)),kind='camera_hypothesis'))
        e=self.evaluator;e.stage='geometry';e.pool='render_repair';saved_best=copy.deepcopy(e.best);saved_camera=copy.deepcopy(self.scene.locked_camera);saved_params=self.scene.params.copy()
        prior=read(self.run/'camera.json');fk=read(self.run/'camera_prior.json')['priors'][prior['mount_profile']];mount=np.asarray(fk['mount_rotation_to_world']);r=np.asarray(prior['rotation_camera_to_world']);rows=[];chosen=None
        try:
            for name,delta,angle in [('mount_y_minus_5mm',[0,-.005,0],0.),('mount_y_plus_5mm',[0,.005,0],0.),('pitch_minus_1deg',[0,0,0],-1.),('pitch_plus_1deg',[0,0,0],1.)]:
                rotation=Rotation.from_rotvec(mount[:,1]*np.radians(angle)).as_matrix()@r
                camera={**prior,'xyz_m':(np.array(prior['xyz_m'])+mount@delta).tolist(),'rotation_camera_to_world':rotation.tolist(),'quaternion_wxyz':quat_from_matrix(rotation).tolist(),'down_angle_deg':float(np.degrees(np.arcsin(rotation[2,2]))),'hypothesis':dict(name=name,evidence=evidence,translation_norm_m=float(np.linalg.norm(delta)),rotation_deg=abs(angle),fov_fixed=True,table_fixed=True)}
                if np.linalg.norm(delta)>settings['translation_norm_max_m'] or abs(angle)>settings['rotation_norm_max_deg']:raise SceneError('camera_bounds','Hypothesis exceeds mount uncertainty')
                self.scene.locked_camera=camera;self.scene.apply(saved_params,internal=True);e.best={}
                e.evaluate(saved_params,'camera_hypothesis_'+name,640,480)
                e.optimize(['apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale'],30,'camera_hypothesis_layout_'+name,640,480)
                candidate=e.best.get('geometry_640x480')
                if candidate:
                    reasons=nondegrading(candidate['metrics'],base['metrics'],self.config);improved=candidate['loss']<base['loss']-self.config['acceptance']['minimum_objective_improvement'] and not reasons
                    rows.append(dict(name=name,camera=camera,candidate=candidate,acceptable=improved,reasons=reasons))
                    if improved and (chosen is None or candidate['loss']<chosen['candidate']['loss']):chosen=rows[-1]
        finally:
            e.best=saved_best;self.scene.locked_camera=chosen['camera'] if chosen else saved_camera
            if chosen:e.best['geometry_640x480']=chosen['candidate']
            self.scene.apply(chosen['candidate']['params'] if chosen else saved_params,internal=True);e.checkpoint();self.budget.finish(round_event)
            experiment=dict(fixed_mount_control=base,fixed_camera=saved_camera,hypotheses=rows,accepted=chosen['name'] if chosen else None,real_calibration=False)
            write(self.run/('camera_hypotheses_'+str(len(self.state['rounds']))+'.json'),experiment)
            self.state['rounds'].append(dict(kind='independent_bounded_camera_hypothesis',accepted=experiment['accepted']));self.state['operations'].append(dict(tool='review_camera_hypothesis',evidence=evidence,confidence=confidence));self.save()
        return experiment

    def render_and_evaluate(self,candidate_id):
        self.check_id(candidate_id)
        if self.scene.stage not in ('geometry','appearance'):raise SceneError('frozen_scene','Optimization tools cannot search after freeze')
        self.evaluator.stage=self.scene.stage;self.evaluator.pool='render_appearance' if self.scene.stage=='appearance' else 'render_repair' if len(self.state['rounds']) else 'render_initial'
        loss,metrics=self.evaluator.evaluate(self.scene.params.copy(),'tool_render_evaluate',640,480);self.save();return dict(loss=loss,metrics=metrics,candidate_id=self.scene.candidate_id())

    def checkpoint(self,label,candidate_id):
        self.check_id(candidate_id)
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',label):raise SceneError('invalid_checkpoint_label',label)
        self.save();folder=self.run/'checkpoints'/label;folder.mkdir(parents=True,exist_ok=True)
        write(folder/'session.json',self.state);write(folder/'binding.json',dict(source_sha256=self.scene.source_hash,observation=self.observation['version'],config_sha256=digest(self.config),camera_sha256=sha(self.run/'camera.json')))
        write(folder/'search_best.json',self.evaluator.best)
        return dict(label=label,candidate_id=self.scene.candidate_id(),budget_rollback=False)

    def restore(self,label,candidate_id):
        self.check_id(candidate_id)
        if self.scene.stage not in ('geometry','appearance'):raise SceneError('frozen_scene','Terminal/frozen state cannot be restored for editing')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',label):raise SceneError('invalid_checkpoint_label',label)
        folder=self.run/'checkpoints'/label;binding=read(folder/'binding.json')
        expected=dict(source_sha256=self.scene.source_hash,observation=self.observation['version'],config_sha256=digest(self.config),camera_sha256=sha(self.run/'camera.json'))
        if binding!=expected:raise SceneError('checkpoint_identity_mismatch',label)
        for name,expected_hash in self.scene.asset_hashes.items():
            if sha((self.scene.path.parent/name).resolve())!=expected_hash:raise SceneError('asset_identity_mismatch',name)
        state=read(folder/'session.json')
        if state['stage']!=self.scene.stage:raise SceneError('wrong_stage','Cannot restore checkpoint from a different stage')
        self.scene.shape_enabled=state['shape_enabled'];self.scene.plate_tilt_enabled=state['plate_tilt_enabled'];self.scene.locked_camera=state.get('camera_override',self.scene.locked_camera);self.scene.apply(state['params'],internal=True)
        if (folder/'search_best.json').exists():
            self.evaluator.best=read(folder/'search_best.json');self.evaluator.checkpoint()
        self.save()
        return dict(candidate_id=self.scene.candidate_id(),budget_rollback=False)

    def inspect_environment(self,candidate_id):
        from .operations import inspect_environment
        return inspect_environment(self,candidate_id)

    def update_environment(self,updates,evidence,confidence,candidate_id):
        from .operations import update_environment
        return update_environment(self,updates,evidence,confidence,candidate_id)

    def render_auxiliary(self,candidate_id):
        from .operations import render_auxiliary
        return render_auxiliary(self,candidate_id)

    def diagnose_support(self,candidate_id):
        from .operations import diagnose_support
        return diagnose_support(self,candidate_id)

    def repair_support(self,updates,evidence,confidence,candidate_id):
        from .operations import repair_support
        return repair_support(self,updates,evidence,confidence,candidate_id)

    def fit_appearance(self,parameter_ids=None,candidate_id=None):
        if self.scene.v5:
            from .v5_tools import optimize_issue,GROUPS,group_stage
            group=next((g for g,ids in GROUPS.items() if parameter_ids is not None and set(ids)==set(parameter_ids) and group_stage(g)=='appearance'),None) if parameter_ids is not None else 'apple_color'
            if group is None:raise SceneError('invalid_parameter_subset','Choose an executable v5 appearance group')
            issue='plate_shading' if group.startswith('plate') else 'cloth_color' if group.startswith('cloth') else 'apple_highlight' if group=='apple_highlight' else 'apple_color'
            objective='scene_lighting_v5' if group in ('shared_lighting','plate_and_lighting') else dict(plate_shading='plate_shading_v5',cloth_color='cloth_coarse_v5',apple_highlight='apple_highlight_v5',apple_color='apple_color_v5')[issue]
            return optimize_issue(self,issue,objective,group,128,candidate_id or self.scene.candidate_id())
        from .operations import fit_appearance
        return fit_appearance(self,parameter_ids,candidate_id)

    def close(self):self.scene.close()
