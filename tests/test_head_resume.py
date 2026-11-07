import importlib.util
from pathlib import Path

from movens_sim.pipeline import complete_receipt,receipt_valid
from movens_sim.util import ROOT,write,read


def test_nested_missing_artifact_invalidates_all_dependent_stages(tmp_path):
    spec=importlib.util.spec_from_file_location('head_launcher',ROOT/'scripts/run_head_camera.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=tmp_path/'run';case=run/'candidates/example';case.mkdir(parents=True);asset=case/'asset.obj';asset.write_text('resource')
    identity={'test':'unchanged'};complete_receipt(case,'fit',identity,[asset],.1)
    write(run/'candidate_index.json',[dict(folder=str(case))]);write(run/'result.json',dict(processing_complete=True))
    for stage in ('fit','validate','render'):complete_receipt(run,stage,identity,[case/'receipts/fit.json'],.1)
    assert module.repair_nested_receipts(run,identity)==[]
    asset.unlink()
    assert module.repair_nested_receipts(run,identity)==[str(case)]
    assert not read(run/'result.json')['processing_complete']
    assert all(not receipt_valid(run,stage,identity) for stage in ('fit','validate','render'))
    assert len(list((run/'repair_history').glob('*/reason.json')))==1
