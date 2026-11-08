"""Workflow selection only; the ablation uses the unchanged v5 scene kernels."""
NO_TERRA = 'ablation_no_terra_v5'
V5_WORKFLOWS = ('terra_closed_loop_v5', NO_TERRA)


def no_terra(config):
    return config.get('workflow') == NO_TERRA


def initial_xml(config):
    return no_terra(config) and config.get('initialization', {}).get('source_mode') == 'initial_xml'


def v5_family(config):
    return config.get('workflow') in V5_WORKFLOWS


def assert_model_allowed(config):
    # Check before credentials, response caches, offline fixtures or transport.
    if no_terra(config):
        raise RuntimeError('model_disabled_ablation: model sending and review-cache reuse are forbidden')


def install_initial_xml_guard(config):
    """Fail closed on historical reconstruction reads in cold-start processes."""
    if not initial_xml(config):return
    import os
    import sys
    from pathlib import Path
    if getattr(sys,'_movens_initial_xml_guard',False):return
    project=Path(os.environ.get('MOVENS_SIM_PROJECT_ROOT',Path(__file__).resolve().parents[3])).resolve()
    forbidden=('ep012_reconstruction_','ep012_ablation_no_terra_','full402_h0_v1','head_camera_pilot_v1','batch_v1')
    def audit(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).absolute()
        for i,part in enumerate(path.parts[:-1]):
            if part=='runs' and any(path.parts[i+1].startswith(x) for x in forbidden):
                raise PermissionError('initial_xml_history_forbidden: '+str(path))
    sys.addaudithook(audit);sys._movens_initial_xml_guard=True
