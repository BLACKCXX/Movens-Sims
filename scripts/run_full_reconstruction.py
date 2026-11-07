#!/usr/bin/env python3
"""Start/status/stop/resume the isolated 402-episode reconstruction."""
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.full_batch import main
if __name__=='__main__':main()
