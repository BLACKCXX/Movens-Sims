"""Deterministic prefix replay for interrupted existing numerical searches.

Only completed objective values are replayed, never model responses. The original
solver reconstructs its state from the same calls; new work uses the remaining
original allowance and the irreversible task ledger. Replay is separately logged.
"""
from pathlib import Path
from ..util import read,write,digest


class SearchReplay:
    def __init__(self,path,identity,metadata=None):
        self.path=Path(path);self.values=self.path.parent/(self.path.stem+'_values');self.index=0;self.replayed=0
        if self.path.exists():
            self.data=read(self.path)
            if self.data['identity']!=identity:raise ValueError('Interrupted numerical search identity changed; explicit same-budget migration required')
        else:
            self.data=dict(identity=identity,metadata=metadata or {},calls=[]);self.flush(force=True)
        # Small per-slot records are authoritative between periodic index checkpoints.
        for slot in sorted(self.values.glob('slot_*.json')):
            row=read(slot);index=row.pop('index')
            if index>len(self.data['calls']):raise ValueError('Numerical replay journal has a missing slot')
            if index==len(self.data['calls']):self.data['calls'].append(row)
            else:self.data['calls'][index]=row

    def flush(self,force=False):
        n=len(self.data['calls'])
        if force or n<=32 or n%64==0:write(self.path,self.data)

    def slot(self,index):
        write(self.values/f'slot_{index:06}.json',dict(index=index,**self.data['calls'][index]))

    def call(self,key,fn):
        index=self.index;self.index+=1;h=digest(key);calls=self.data['calls']
        if index<len(calls):
            row=calls[index]
            if row['key']!=h:raise ValueError('Interrupted solver prefix diverged at call '+str(index))
            if row['status']=='completed':
                self.replayed+=1;return (row['value'] if 'value' in row else read(self.values/row['value_file'])),True
        else:
            calls.append(dict(key=h,status='reserved'));self.slot(index);self.flush()
        filename=f'{index:06}.json';path=self.values/filename
        if path.exists():
            value=read(path);replayed=True;self.replayed+=1
        else:
            value=fn();write(path,value);replayed=False
        calls[index].update(status='completed',value_file=filename);self.slot(index);self.flush()
        return value,replayed
