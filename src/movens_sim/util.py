from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('MOVENS_SIM_PROJECT_ROOT', CODE_ROOT)).resolve()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, Path): return str(obj)
    raise TypeError(type(obj).__name__)

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default, allow_nan=False)+'\n')
    os.replace(temp, path)

def append(path, obj):
    with open(path, 'a') as f: f.write(json.dumps(obj, default=default, allow_nan=False)+'\n')

def now(): return datetime.now(timezone.utc).isoformat()

def code_identity():
    files = sorted((ROOT/'src/movens_sim').glob('*.py')) + [ROOT/'pyproject.toml']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def video_info(path):
    p = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries',
        'stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)],check=True,capture_output=True,text=True)
    v = json.loads(p.stdout)['streams'][0]
    a,b = v['avg_frame_rate'].split('/')
    return dict(width=int(v['width']),height=int(v['height']),fps=float(a)/float(b),frames=int(v['nb_read_frames']),duration_s=float(v.get('duration',0)))

def save_rgb(path, rgb):
    from PIL import Image
    Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(path)
