"""Source-retention and supervision preflight before the registered fits."""
import hashlib
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run_v75 import ROOT, OUT, adapter, read, save, sha
from v79_execute import rows
from v75_views import view, byte_matrix
from v75_corrective import stable
from v82_capacity import DEST, LAST, SEED


def main():
    if (DEST/'supervision_preflight.json').exists():raise FileExistsError('Frozen preflight exists')
    r=rows();selected=np.zeros(len(r),bool);selected[np.load(DEST/'selected_rows.npy')]=True
    v=adapter();rng=np.random.default_rng(SEED);wanted=[]
    for (route,label),cell in r[selected].groupby(['route','label_index']):
        indices=cell.row_position.to_numpy();wanted.extend(rng.choice(indices,min(6,len(indices)),replace=False).tolist())
    wanted=np.sort(np.asarray(wanted));results=[];offset=0
    for b in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['event_id','message_sanitized'],use_threads=False):
        positions=wanted[np.searchsorted(wanted,offset):np.searchsorted(wanted,offset+len(b))]
        for pos in positions:
            raw=b.column(1)[int(pos)-offset].as_py();record=r.iloc[int(pos)]
            text,ledger=view(raw);s='' if raw is None else raw
            reconstructed=''.join(s[a:z] for a,z,_ in ledger['spans'])
            assert reconstructed==s
            variant=s+'\t';before=v.prepare_record({'message_sanitized':s});after=v.prepare_record({'message_sanitized':variant})
            unchanged=before['route']==after['route'] and before['facts']==after['facts']
            original=byte_matrix([stable(text)]);changed=byte_matrix([stable(view(variant)[0])])
            encoded_change=bool((original-changed).nnz)
            # Variant adds only a terminal whitespace character and retains the
            # complete original sequence; no internal value/polarity is changed.
            results.append({'parent_row':int(pos),'component':int(record.component),'fold':int(record.fold),
                'route':record.route,'class':int(record.label_index),'raw_sha256':hashlib.sha256(s.encode()).hexdigest(),
                'ledger_roundtrip':True,'original_sequence_retained':variant[:-1]==s,
                'parser_facts_and_route_unchanged':unchanged,'model_byte_input_changed':encoded_change,
                'usable_equivalent_view':unchanged and encoded_change,'null_parent':raw is None,
                'scope':'Terminal whitespace only; independent syntactic preservation, not a new attack-label annotation.'})
        offset+=len(b)
    assert len(results)==len(wanted)
    pd.DataFrame(results).to_parquet(DEST/'equivalent_view_audit.parquet',index=False)
    usable=[a for a in results if a['usable_equivalent_view'] and not a['null_parent']]
    coverage=Counter((a['route'],a['class']) for a in usable)
    # Actual official supervision cross-component contrasts; do not synthesize
    # missing labels. Only candidate matching, not proof of causal threat truth.
    projections=pd.read_parquet(OUT/'projections.parquet',columns=['facts'])
    facts=[json.loads(s) for s in projections.facts]
    coarse=['action','transport_protocol','src_role','dst_role']
    keys=np.asarray([json.dumps({k:f[k] for k in coarse if k in f},sort_keys=True) for f in facts],object)
    exposed=r[selected].copy();exposed['coarse_key']=keys[exposed.projection_id.to_numpy()]
    contrasts=[]
    for key,group in exposed.groupby('coarse_key',sort=False):
        pairs=group[['row_position','projection_id','component','route','label_index']].drop_duplicates(['projection_id','component','route','label_index'])
        classes=sorted(pairs.label_index.unique().tolist())
        if len(classes)<2:continue
        for a in range(3):
            for b in range(a+1,3):
                aa=pairs[pairs.label_index.eq(a)];bb=pairs[pairs.label_index.eq(b)]
                if aa.empty or bb.empty:continue
                first=aa.iloc[0];eligible=bb[bb.component.ne(first.component)]
                if eligible.empty:continue
                second=eligible.iloc[0];fa=facts[int(first.projection_id)];fb=facts[int(second.projection_id)]
                differences={k:[fa.get(k),fb.get(k)] for k in sorted(set(fa)|set(fb)) if fa.get(k)!=fb.get(k)}
                contrasts.append({'class_pair':[a,b],'rows':[int(first.row_position),int(second.row_position)],
                    'components':[int(first.component),int(second.component)],'routes':[first.route,second.route],
                    'full_fact_differences':differences,'coarse_key':key,'different_observable_facts':bool(differences),
                    'independent_security_criterion_confirmed':False})
    save(DEST/'official_contrast_candidates.json',contrasts)
    supports=r.groupby(['route','label_index']).size()
    vpc_counts=[int(supports.get(('vpc_v2',k),0)) for k in range(3)]
    assert vpc_counts==[30131,0,10620]
    ratios=read(ROOT/'artifacts/v81_diagnosis_20260927/derived_field_label_support.json')
    out={'raw_reconstruction_checks':len(results),'equivalent_view_candidates':len(results),
         'usable_nonnull_equivalent_views':len(usable),'class_coverage':sorted({a['class'] for a in usable}),
         'format_class_coverage':[{'route':route,'class':cl,'views':n} for (route,cl),n in sorted(coverage.items())],
         'official_cross_component_contrast_candidates':len(contrasts),
         'observable_fact_difference_contrasts':sum(a['different_observable_facts'] for a in contrasts),
         'equivalence_views_trained':False,'reason_not_four_arms':'Terminal whitespace supplies valid input variation but no demonstrated link to current ASA opposite-support/VPC zero-M deficits; do capacity first.',
         'VPC_train_class_support':vpc_counts,'VPC_full_content_normal_examples':0,
         'relative_rate_training_support':ratios['support'],'relative_observations_added_to_classifier':False,
         'VPC_missing_label_resolution':'Not resolved: existing common-view candidates cannot certify M/S without missing role/payload/authorization context.',
         'source_sha256':sha(__file__),'original_train_sha256':sha(ROOT/'data/official/train.parquet'),
         'registration_sha256':sha(DEST/'registration.json'),'training_label_edits':0,'pseudo_labels':0}
    save(DEST/'supervision_preflight.json',out)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
