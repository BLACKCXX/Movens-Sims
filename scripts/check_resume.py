#!/usr/bin/env python3
"""Temporarily hide one final video and verify only the missing stage regenerates."""
import subprocess
from movens_sim.util import ROOT,read,write,sha

run=ROOT/'runs/batch_v1/episode_000004';video=run/'ego_static.mp4';backup=run/'ego_static.mp4.resume_backup'
before=dict(video_sha256=sha(video),fit_receipt_sha256=sha(run/'receipts/fit.json'),physics_receipt_sha256=sha(run/'receipts/validate.json'),candidate_history_sha256=sha(run/'candidates.jsonl'),gpt_summary_sha256=sha(run/'gpt_summary.json'))
if backup.exists():raise RuntimeError('A previous resume check backup needs inspection')
video.rename(backup)
try:
    result=subprocess.run([str(ROOT/'scripts/movens-sim'),'run','--episode','4','--resume','--gpt'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert video.is_file();assert sha(run/'receipts/fit.json')==before['fit_receipt_sha256'];assert sha(run/'receipts/validate.json')==before['physics_receipt_sha256'];assert sha(run/'candidates.jsonl')==before['candidate_history_sha256'];assert sha(run/'gpt_summary.json')==before['gpt_summary_sha256']
    write(ROOT/'runs/batch_v1/resume_checks.json',dict(passed=True,case='Missing completed ego_static.mp4',only_missing_stage_regenerated=True,numerical_candidates_and_api_calls_unchanged=True,video_sha_before=before['video_sha256'],video_sha_after=sha(video),video_bytes_identical=sha(video)==before['video_sha256'],command=result.args,stdout=result.stdout));backup.unlink();print(result.stdout)
except BaseException:
    if not video.exists():backup.rename(video)
    raise
