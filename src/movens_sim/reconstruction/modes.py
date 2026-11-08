"""Workflow selection only; the ablation uses the unchanged v5 scene kernels."""
NO_TERRA = 'ablation_no_terra_v5'
V5_WORKFLOWS = ('terra_closed_loop_v5', NO_TERRA)


def no_terra(config):
    return config.get('workflow') == NO_TERRA


def v5_family(config):
    return config.get('workflow') in V5_WORKFLOWS


def assert_model_allowed(config):
    # Check before credentials, response caches, offline fixtures or transport.
    if no_terra(config):
        raise RuntimeError('model_disabled_ablation: model sending and review-cache reuse are forbidden')
