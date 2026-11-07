"""Read-only discovery and portable XML resource closure with dual hashes."""
from __future__ import annotations
import copy
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from .util import ROOT, digest, now, read, sha, write

FILE_ATTRS = ('file','fileleft','fileright','fileup','filedown','filefront','fileback')

def resource_closure(xml_path):
    """Resolve includes recursively; preserve the declaring XML base and compiler dirs."""
    records = {}; visiting = set()
    def visit(path, inherited=None):
        path=Path(path).resolve()
        if path in visiting: raise ValueError(f'Include cycle: {path}')
        if path in records: return
        if not path.is_file(): raise FileNotFoundError(str(path))
        visiting.add(path); tree=ET.parse(path); compiler=tree.find('compiler')
        dirs=dict(inherited or {})
        if compiler is not None:
            for k in ('meshdir','texturedir','assetdir'):
                if compiler.get(k): dirs[k]=(path.parent/compiler.get(k)).resolve()
        refs=[]
        for elem in tree.iter():
            for attr in FILE_ATTRS:
                name=elem.get(attr)
                if not name: continue
                if elem.tag=='include': base=path.parent
                elif elem.tag=='mesh': base=dirs.get('meshdir',dirs.get('assetdir',path.parent))
                elif elem.tag=='texture': base=dirs.get('texturedir',dirs.get('assetdir',path.parent))
                else: base=dirs.get('assetdir',path.parent)
                target=(base/name).resolve()
                if not target.is_file(): raise FileNotFoundError(f'{path}: {elem.tag}.{attr}={name} -> {target}')
                refs.append((elem,attr,target))
                if elem.tag=='include': visit(target,dirs)
        records[path]=(tree,refs); visiting.remove(path)
    visit(xml_path)
    return records

def portable_copy(xml_path, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    records=resource_closure(xml_path); mapping={}
    for src in records:
        mapping[src]=out/('scene.xml' if src==Path(xml_path).resolve() else 'includes/'+sha(src)[:16]+'_'+src.name)
    for _,refs in records.values():
        for _,_,src in refs:
            if src not in mapping: mapping[src]=out/'assets'/(sha(src)[:16]+'_'+src.name)
    evidence=[]
    for src,dst in mapping.items():
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src not in records: shutil.copy2(src,dst)
        else:
            tree,refs=records[src]
            for elem,attr,target in refs: elem.set(attr,os.path.relpath(mapping[target],dst.parent))
            comp=tree.find('compiler')
            if comp is not None:
                for attr in ('meshdir','texturedir','assetdir','strippath'): comp.attrib.pop(attr,None)
            tree.write(dst,encoding='utf-8',xml_declaration=True)
        evidence.append(dict(source=str(src),portable=str(dst.relative_to(out)),source_sha256=sha(src),portable_sha256=sha(dst),bytes=dst.stat().st_size))
    # Compile outside the source tree later, not only while original absolute paths work.
    return evidence

def discover(source, limit=5):
    source=Path(source).resolve()
    active=read(source/'outputs/appletoplate_tableup005mm_active.json')
    active_root=Path(active['root']); registry=read(active_root/'baseline_video_index.json')
    training=read(source/'data/appletoplate_tableup005mm_training_v1/index.json')
    revision=training['data_revision'];raw=source/'data/apple_raw'/revision
    if training['source_batch']!=str(active_root): raise ValueError('Active batch and paired index disagree; source review needed')
    # Record all scene families, including incomplete experiments. Registration determines priority.
    inventory=[]
    for p in sorted((source/'outputs').rglob('scene.json')):
        try:
            d=read(p)
            if 'source_episode_index' in d:
                inventory.append(dict(episode=int(d['source_episode_index']),scene_setup=str(p),family=p.relative_to(source/'outputs').parts[0],mtime_ns=p.stat().st_mtime_ns))
        except (ValueError,OSError): continue
    # The latest registered active batch has a complete video/state-to-source mapping.
    by_ep={}; missing=[]
    for row in registry: by_ep.setdefault(int(row['episode']),[]).append(row)
    entries=[]
    for ep,rows in sorted(by_ep.items()):
        valid=[]
        for row in rows:
            c=Path(row['path']);s=c/'scene'; reasons=[]
            real=Path(row.get('source_rgb') or '__missing_real_video__')
            for p in [real,Path(row.get('main_rgb') or '__missing_sim_video__'),s/'scene.xml',s/'scene.json',s/'robot.json',s/'reset.json',c/'run/states.npz',c/'run/frame_map.json']:
                if not p.is_file(): reasons.append(f'missing {p}')
            if not reasons:
                try:resource_closure(s/'scene.xml')
                except Exception as e: reasons.append(str(e))
            if reasons: missing.append(dict(episode=ep,candidate=str(c),reasons=reasons));continue
            meta=read(s/'scene.json')
            if meta.get('source_episode_index')!=ep or row['source_revision']!=revision: raise ValueError('Episode/revision identity mismatch')
            if meta.get('scene_sha256')!=sha(s/'scene.xml'): raise ValueError(f'XML hash mismatch {s}')
            valid.append((s.stat().st_mtime_ns,row))
        if valid:
            _,row=max(valid,key=lambda x:x[0]);entries.append(dict(episode=ep,candidate=row['path'],scene_dir=str(Path(row['path'])/'scene'),real_video=row['source_rgb'],old_sim_video=row['main_rgb'],data_revision=revision,registration=str(active_root/'baseline_video_index.json')))
    chosen=entries[:limit]
    if len(chosen)<limit: raise RuntimeError(f'Only {len(chosen)} complete episodes; see discovery evidence')
    result=dict(version='manifest_v1',created_utc=now(),frozen=True,source_project=str(source),data_revision=revision,dataset='nvidia/GR00T-N1.7-AppleToPlate',source_batch=str(active_root),selection_rule='Ascending distinct episode ID among complete registered active-batch inputs. Grasp success is not used. Latest registered valid candidate per episode.',entries=chosen)
    write(ROOT/'configs/discovery.json',dict(scene_inventory=inventory,complete_episodes=[e['episode'] for e in entries],missing=missing,registry_sha256=sha(active_root/'baseline_video_index.json'),older_families='Historical, paused and superseded candidates retained in inventory, not used as the active registered result.'))
    manifest=ROOT/'configs/manifest.json'
    if manifest.exists():
        old=read(manifest)
        if old['entries']!=chosen: raise RuntimeError('Frozen manifest differs: use a new named manifest, never silently replace episodes')
        return old
    result['selection_sha256']=digest(chosen);write(manifest,result);return result

def import_episode(entry, source_project):
    ep=entry['episode'];out=ROOT/'data'/f'episode_{ep:06}';receipt=out/'source_manifest.json'
    if receipt.exists():
        d=read(receipt)
        if any(d.get(k)!=value for k,value in entry.items()):raise ValueError('Manifest candidate differs from immutable imported episode; choose a new data namespace, do not silently reuse it')
        for f in d['imported_files']:
            if sha(out/f['path'])!=f['sha256']: raise ValueError('Immutable imported input changed')
        return out
    out.mkdir(parents=True,exist_ok=True);s=Path(entry['scene_dir']);c=Path(entry['candidate'])
    closure=portable_copy(s/'scene.xml',out/'source')
    for name in ('scene.json','robot.json','reset.json','softpad.json','grid.npz','construction.json','batch_construction.json','prepared_manifest.json'):
        if (s/name).exists():shutil.copy2(s/name,out/'source'/name)
    meta=read(out/'source/scene.json');meta['scene']='scene.xml';meta['scene_sha256']=sha(out/'source/scene.xml');write(out/'source/scene.json',meta)
    for src,dst in [(Path(entry['real_video']),out/'real_ego.mp4'),(Path(entry['old_sim_video']),out/'old_sim.mp4'),(c/'run/frame_map.json',out/'source_frame_map.json'),(c/'run/report.json',out/'source_run_report.json')]:shutil.copy2(src,dst)
    # Only the reset and mapped source-frame-zero state are imported; no action trajectories.
    import numpy as np
    with np.load(c/'run/states.npz') as z:
        idx=int(read(c/'run/frame_map.json')['source_frames'][0]['state_index'])
        states={}
        for k in ('qpos','qvel','ctrl','integration_state'):
            if k in z: states[k]=z[k][idx].copy();states['xml_reset_'+k]=z[k][0].copy()
        states['source_state_index']=np.array(idx);states['source_timestamp_s']=np.array(z['timestamps'][idx])
        np.savez_compressed(out/'source_initial_state.npz',**states)
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    for name in ('simulation/common.py','simulation/apple_batch_render.py','simulation/softpad.py','docs/softpad/tableup005_batch.md','AGENTS.md','agent.md'):
        src=Path(source_project)/name
        if src.exists():shutil.copy2(src,evidence/(name.replace('/','__')))
    # Verify real raw video against immutable upstream source index when LFS identity exists.
    raw=Path(source_project)/'data/apple_raw'/entry['data_revision'];index=read(raw/'source_index.json')
    rel=str(Path(entry['real_video']).relative_to(raw));item=next(x for x in index['files'] if x['path']==rel)
    expected=item.get('lfs',{}).get('oid');actual=sha(out/'real_ego.mp4')
    if expected and expected!=actual:raise ValueError('Real video failed upstream LFS SHA256')
    shutil.copy2(raw/'meta/info.json',out/'dataset_info.json')
    try:commit=subprocess.check_output(['git','-C',str(source_project),'rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError:commit=None
    imported=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    write(receipt,dict(**entry,imported_utc=now(),upstream_repository='https://github.com/wendell0218/Movens-Interact',source_git_commit=commit,source_has_local_changes=True,resource_closure=closure,imported_files=imported,real_video_upstream_sha256=expected,real_video_sha256=actual,source_states_sha256=sha(c/'run/states.npz'),state_choice='source frame zero saved post-settle state; original XML/reset state separately preserved',independent_assets=True,download_needed=False,scale_policy='Existing mesh scales and baked shape already applied. No new table-height compensation or uniform scale at import.'))
    return out
