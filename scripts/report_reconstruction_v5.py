#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read
from movens_sim.reconstruction.v5_delivery import build
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ep012_reconstruction_v5.json');p.add_argument('--package',action='store_true');a=p.parse_args();r=build(read(ROOT/a.config),a.package);print({k:r[k] for k in ('execution_complete','quality_all_passed','full402_stop_present')})
