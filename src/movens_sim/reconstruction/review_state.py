"""Category pass validity follows dependency fingerprints, never a bare boolean."""
from ..util import digest

CATEGORIES=('observation','geometry','initial_support','appearance','rest')


def dependencies(category,params,camera,observation_version):
    geometry={k:v for k,v in params.items() if k.startswith(('apple.','plate.')) and not k.startswith(('apple.tint_','apple.yellow_mix','plate.albedo'))}
    environment={k:v for k,v in params.items() if k.startswith('table.') or k=='cloth.coverage'}
    appearance={k:v for k,v in params.items() if k not in geometry and k not in environment}
    physical=dict(geometry=geometry,table=environment)
    payload={
        'observation':dict(observation=observation_version,camera=camera,geometry=geometry,environment=environment),
        'geometry':dict(observation=observation_version,camera=camera,geometry=geometry),
        'initial_support':physical,
        'appearance':dict(observation=observation_version,camera=camera,params=params),
        'rest':physical,
    }
    return digest(payload[category])


def invalidate(state,previous,current,old_camera,new_camera):
    observation=state.get('observation_version','unknown')
    for category,check in state.get('category_checks',{}).items():
        if dependencies(category,previous,old_camera,observation)!=dependencies(category,current,new_camera,observation):
            check.update(valid=False,invalidation_reason='scene_dependency_changed')


def valid_check(state,category,params,camera,observation_version):
    check=state.get('category_checks',{}).get(category,{})
    return bool(check.get('passed') and check.get('valid') and check.get('dependency_hash')==dependencies(category,params,camera,observation_version))


def protected_slots(state,params,camera,observation_version):
    """One terminal slot plus unspent first/recheck entitlements per category."""
    reserved=1
    for c in CATEGORIES:
        count=state.get('category_attempts',{}).get(c,0)
        if count<1:reserved+=2
        elif count<2 and not valid_check(state,c,params,camera,observation_version):reserved+=1
    return reserved
