"""Local stdio MCP adapter. Tools call the same durable reconstruction session."""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import functools
import json
import os
import sys
import time
from pathlib import Path

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools import Tool

from .session import ReconstructionSession
from ..util import ROOT,CODE_ROOT,append,read


def make_server(session):
    def invoke(name,**kwargs):
        start=time.monotonic()
        with (session.run/'.tool.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                value=getattr(session,name)(**kwargs)
                result=dict(ok=True,result=value)
            except Exception as exc:
                result=dict(ok=False,error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc)[:500]))
            append(session.run/'mcp_tool_calls.jsonl',dict(tool=name,arguments=kwargs,ok=result['ok'],candidate_id=session.scene.candidate_id(),wall_seconds=time.monotonic()-start,transport='real_stdio_mcp',error=result.get('error')))
            return result

    def inspect_scene() -> dict:
        """Inspect actual parameters, dimensions, mount provenance and shared budget."""
        return invoke('inspect_scene')

    def propose_or_apply_object_update(updates:dict[str,float],candidate_id:str,apply:bool=False) -> dict:
        """Absolute source-relative scales and world XY/allowed orientation; automatic support."""
        return invoke('propose_or_apply_object_update',updates=updates,candidate_id=candidate_id,apply=apply)

    def fit_geometry(parameter_ids:list[str]|None=None,candidate_id:str|None=None,evidence:str='',confidence:float=0.) -> dict:
        """Joint numerical fit with the remaining task-global render/time/round budget."""
        return invoke('fit_geometry',parameter_ids=parameter_ids,candidate_id=candidate_id,evidence=evidence,confidence=confidence)

    def update_shape_or_asset(updates:dict[str,float],evidence:str,confidence:float,candidate_id:str) -> dict:
        """Evidence-gated plate profile depth or small plate tilt; no arbitrary mesh replacement."""
        return invoke('update_shape_or_asset',updates=updates,evidence=evidence,confidence=confidence,candidate_id=candidate_id)

    def update_appearance(updates:dict[str,float],candidate_id:str) -> dict:
        """Appearance only after actual Terra geometry approval, geometry stays locked."""
        return invoke('update_appearance',updates=updates,candidate_id=candidate_id)

    def review_camera_hypothesis(evidence:str,confidence:float,candidate_id:str) -> dict:
        """Independent small mount hypotheses only after common two-object and global residual evidence."""
        return invoke('review_camera_hypothesis',evidence=evidence,confidence=confidence,candidate_id=candidate_id)

    def render_and_evaluate(candidate_id:str) -> dict:
        """Render and measure under the common optimization budget, never free previews."""
        return invoke('render_and_evaluate',candidate_id=candidate_id)

    def checkpoint(label:str,candidate_id:str) -> dict:
        """Save a complete scene-parameter checkpoint, excluding irreversible budget ledger."""
        return invoke('checkpoint',label=label,candidate_id=candidate_id)

    def restore(label:str,candidate_id:str) -> dict:
        """Restore a compatible same-stage checkpoint without restoring spent budget."""
        return invoke('restore',label=label,candidate_id=candidate_id)

    funcs=[inspect_scene,propose_or_apply_object_update,fit_geometry,update_shape_or_asset,update_appearance,review_camera_hypothesis,render_and_evaluate,checkpoint,restore]
    tools=[];registry={p['id']:p for p in session.scene.registry()}
    for fn in funcs:
        tool=Tool.from_function(fn)
        schema=tool.parameters;schema['additionalProperties']=False
        props=schema.get('properties',{})
        if 'updates' in props:
            stage='appearance' if fn.__name__=='update_appearance' else 'geometry'
            allowed={k:v for k,v in registry.items() if stage in v['stages']}
            if fn.__name__=='update_shape_or_asset':allowed={k:v for k,v in allowed.items() if k in ('plate.depth_ratio','plate.rx','plate.ry')}
            props['updates']=dict(type='object',properties={k:dict(type='number',minimum=p['lower'],maximum=p['upper']) for k,p in allowed.items()},additionalProperties=False)
        if 'parameter_ids' in props:
            props['parameter_ids']=dict(anyOf=[dict(type='array',items=dict(type='string',enum=[k for k,p in registry.items() if 'geometry' in p['stages']])),dict(type='null')],default=None)
        tools.append(tool)
    return MCPServer('Movens-Sim',version='3.0',tools=tools,log_level='WARNING')


async def call_tools(run, calls, timeout=1900.):
    """Actual MCP transport; no assumption that API models inherit Codex MCP."""
    params=StdioServerParameters(command=str(ROOT/'.venv/bin/python'),args=['-B','-m','movens_sim.reconstruction.mcp_server','--run',str(Path(run).resolve())],cwd=str(ROOT),env={**os.environ,'PYTHONPATH':str(CODE_ROOT/'src'),'MUJOCO_GL':'egl','MUJOCO_EGL_DEVICE_ID':os.environ.get('MUJOCO_EGL_DEVICE_ID','0'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    results=[]
    async with Client(params,read_timeout_seconds=timeout,raise_exceptions=False) as client:
        schemas=await client.list_tools()
        for call in calls:
            arguments=call.get('arguments',{}).copy()
            if arguments.pop('_current_candidate_id',False):
                inspection=await client.call_tool('inspect_scene',{})
                payload=inspection.structured_content
                if not payload:payload=json.loads(inspection.content[0].text)
                arguments['candidate_id']=payload['result']['candidate_id']
            result=await client.call_tool(call['tool'],arguments)
            payload=result.structured_content
            if payload is None:
                try:payload=json.loads(result.content[0].text) if result.content else dict(ok=False,error='empty MCP result')
                except (ValueError,IndexError):payload=dict(ok=False,error=dict(code='mcp_validation_error',message=result.content[0].text[:500] if result.content else 'empty result'))
            results.append(dict(tool=call['tool'],arguments=arguments,result=payload,is_error=result.is_error))
            if result.is_error or not payload.get('ok'):break
    return dict(transport='stdio',sdk_version=__import__('importlib.metadata',fromlist=['version']).version('mcp'),tools=[t.model_dump(mode='json') for t in schemas.tools],calls=results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();run=Path(args.run)
    config=read(run/'resolved_config.json');session=ReconstructionSession(run,config)
    try:make_server(session).run(transport='stdio')
    finally:session.close()


if __name__=='__main__':main()
