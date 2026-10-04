"""Full-corpus input replay and finite observable ambiguity, without fitting models."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_v61 import contexts
from v61_common import FIELDS, save, sha
from v61_runtime import load_data


def minimum_errors(keys,y):
    counts=pd.DataFrame({'key':keys,'label':y}).groupby(['key','label']).size().unstack(fill_value=0)
    mixed=counts.gt(0).sum(axis=1)>1
    return {'groups':len(counts),'mixed_groups':int(mixed.sum()),
        'rows_in_mixed_groups':int(counts[mixed].to_numpy().sum()),
        'finite_sample_minimum_errors':int((counts.sum(axis=1)-counts.max(axis=1)).sum())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args();t=time.monotonic()
    df,ctx,config=load_data(a.data)
    meta=pd.read_parquet(a.data/'private_join_audit.parquet').to_dict('records')
    assert sha(a.data/'private_join_audit.parquet')==config['data_bindings']['private_join_audit.parquet']
    renamed=[{k:('rename::'+str(v) if k in ['source','destination','source_port','destination_port','namespace'] and v else v)
              for k,v in m.items()}|{'label':'unavailable_to_context'} for m in meta]
    # Match the frozen preparer's NumPy Unicode representation. Its deterministic
    # neighbor ordering hashes repr(tuple), which differs for Python vs np strings.
    facts=df[FIELDS].to_numpy(dtype=str);replay=contexts(facts,renamed,df.role.to_numpy())
    invariant={name:bool(np.array_equal(ctx[name],value)) for name,value in zip(['stats','neighbors','relation'],replay[:3])}
    assert all(invariant.values())
    tuples=pd.MultiIndex.from_frame(df[['text']+FIELDS]);keys=pd.factorize(tuples,sort=False)[0]
    # The full available symbolic relation structure, BEFORE the 32-state cap or
    # model vocabulary projection, is examined separately from single-event inputs.
    lookup={}
    for i,m in enumerate(meta):lookup.setdefault((df.role.iat[i],m['namespace'],m['source']),[]).append(i)
    relation_keys=[None]*len(df)
    for members in lookup.values():
        for i in members:
            m=meta[i];own=(int(keys[i]),m['destination'],m['source_port'],m['destination_port'])
            states=set((int(keys[j]),meta[j]['destination'],meta[j]['source_port'],meta[j]['destination_port']) for j in members)
            states.discard(own)
            # Equality-only full multiset plus distinct-symbol degrees. Exact
            # tuples are factorized; there is no lossy digest equality assumption.
            rel=tuple(sorted((k,int(dst==m['destination']),int(bool(pt) and pt==m['destination_port'])) for k,dst,sp,pt in states))
            relation_keys[i]=(int(keys[i]),len({q[1] for q in states}),len({q[2] for q in states if q[2]}),len({q[3] for q in states if q[3]}),rel)
    relations=pd.factorize(pd.Series(relation_keys,dtype=object),sort=False)[0]
    audit={}
    for role in ['fit','selection','evaluation']:
        mask=df.role.eq(role).to_numpy();y=df.label.to_numpy()[mask]
        audit[role]={'single_event':minimum_errors(keys[mask],y),
            'uncapped_equality_only_context_diagnostic':minimum_errors(relations[mask],y),
            'context_diagnostic_not_identical_to_C_or_D_model_representation':True}
    save(a.data/'full_input_replay_audit.json',{'full_99398_row_identity_bijection_invariance':invariant,
        'unchanged_source_role_membership':True,'no_model_training':True,'ambiguity':audit,
        'interpretation':'Empirical label conflict for explicitly defined observable views; not a universal Bayes error or attack-truth proof',
        'outer_labels_used_for_input_ambiguity_audit_not_model_selection':True,
        'elapsed_seconds':time.monotonic()-t,'source_sha256':sha(__file__)})
    print({'full_input_invariance':invariant,'ambiguity':audit},flush=True)


if __name__=='__main__':main()
