"""Platform-only full 64-normal QP with retained actual unforwarded context."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_info
from experiment_review import ROOT, read, sha
import v169_cpu_training_entry_v1 as entry
from v169_cpu_execution_review_v1 import CPU_RESOURCES, platform_identity, require_resources
from v169_dynamic_trial_restoration_v2 import propose


def main(arm):
    identity=platform_identity()
    entry.configure()
    require_resources(CPU_RESOURCES,'initial')
    out=ROOT/f'artifacts/v169_platform_CPU_full_width_{arm}_qualification_v1'
    assert not out.exists()
    out.mkdir()
    run=out/'unforwarded_actual_backend'
    run.mkdir()
    original=read(ROOT/'training/review_policy/v169_prior_pair_execution_contract.json')
    plan=dict(original,resources=CPU_RESOURCES)
    started=time.monotonic()
    with patch.object(entry,'OUT',run):
        backend=entry.ActualBackend(0,arm,plan)
    width=entry.width(arm)
    initial=backend.identity()
    try:
        assert width==1060832+(arm=='B') and sum(p.numel() for p in backend.model.parameters())==width
        if arm=='B':assert float(backend.model.beta.detach())==0.
        vectors=[np.ones(width,np.float64) for _ in range(8)]
        outputs=[np.ones((22546,3),np.float64) for _ in range(16)]
        ledgers=[np.ones(len(backend.ctx['OOF_rows']),bool) for _ in range(3)]
        frames=[]
        for scope in ['OOF','deployment']:
            frame=backend.ctx[scope+'_rows'].copy()
            for cls in [0,1,2]:frame[f'p{cls}']=0.;frame[f'logp{cls}']=0.
            frame['pred']=1;frame['actual_margin']=0.;frame['scope']=scope
            frame['rival']=0;frame['role']=0;frame['protection_kind']='qualification_memory_standin'
            frames.append(frame)
        u=np.empty(width);u.fill(0.);u[:2]=1.
        gm=np.empty(width);gm.fill(0.);gm[0]=-1.
        gs=np.empty(width);gs.fill(0.);gs[1]=-1.
        if arm=='B':gm[-1]=.1;gs[-1]=.2
        normals=np.empty((64,width));normals.fill(0.)
        normals[np.arange(64),np.arange(2,66)]=1.
        normals[0,100]=-0.;gm[100]=-0.;gs[101]=-0.
        origin=np.full(64,.2);current=np.full(64,-.1)
        floors=np.full(64,16*np.finfo(np.float64).eps)
        events=[]
        def callback(event,value):
            events.append(dict(event=event,optimizer_iterations=int(value.nit) if value is not None else None))
        result=propose(callback,u,normals,origin,current,gm,gs,floors)
        assert [event['event'] for event in events]==['call','return']
        assert result['normal_rank']==66 and result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard'
        assert len(result['inequality_reviews'])==66 and all(item['passed'] for item in result['inequality_reviews'])
        assert all(item['resolved_negative'] for item in result['class_reviews'])
        observation={scope+'_'+name:np.load(entry.PRIOR/f'role0/endpoint/{scope}_{name}.npy') for scope in ['OOF','deployment'] for name in ['q','logq']}
        guard=backend._guard(observation)
        assert guard['passed'] and backend.identity()==initial
        counts=backend.counter.counts()
        assert all(value==0 for value in counts.values())
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert peak<=CPU_RESOURCES['qualification_peak_RSS_bound_bytes']
        sources=[Path(__file__).resolve(),Path(entry.__file__),ROOT/'platform_runtime/v169_cpu_runtime_v1.py',ROOT/'training/v169_dynamic_trial_restoration_v2.py',ROOT/'training/v169_working_joint_restoration_v2.py']
        report=dict(status='actual_Linux_aarch64_full_width_rank66_CPU_QP_and_saved_complete_guard_qualified',passed=True,arm=arm,platform=identity,complete_parameters=width,current_functions=64,normal_rank=66,all_coordinates_touched=True,retained_vectors=len(vectors),retained_full_outputs=len(outputs),retained_ledgers=len(ledgers),retained_complete_frames=len(frames),full_original_saved_guard=guard,peak_RSS_bytes=peak,registered_peak_RSS_bound_bytes=CPU_RESOURCES['qualification_peak_RSS_bound_bytes'],elapsed_seconds=time.monotonic()-started,native_pools=threadpool_info(),synthetic_QP_calls=1,original_unit_reviews={key:value for key,value in result.items() if key not in ['displacement','correction']},complete_displacement_sha256=hashlib.sha256(result['displacement'].tobytes()).hexdigest(),all_real_backend_parameters_unchanged=True,official_counters=counts,official_calls=0,fits=0,permanent_updates=0,not_full_official_training_peak=True,source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in sources})
        (out/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(status=report['status'],arm=arm,peak_RSS_bytes=peak,official_calls=0)),flush=True)
    finally:
        backend.close_resources()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A','B'],required=True)
    main(parser.parse_args().arm)
