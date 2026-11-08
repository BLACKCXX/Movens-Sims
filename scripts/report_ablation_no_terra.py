#!/usr/bin/env python3
"""Prepare immutable ablation receipts or compare three completed new results."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read
from movens_sim.reconstruction.ablation_delivery import prepare,build

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ep012_ablation_no_terra.json');p.add_argument('--prepare',action='store_true');p.add_argument('--package',action='store_true');a=p.parse_args();c=read(ROOT/a.config)
    result=prepare(c) if a.prepare else build(c,a.package)
    print({k:result[k] for k in ('search_kernel_unchanged','execution_complete','audit_passed','model_requests') if k in result})
