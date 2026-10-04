"""Independent ledger recount and all-format N2 input isolation preflight."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse

from run_v75 import ROOT, OUT, load_sparse, save, sha
from v75_views import BYTE_FEATURES, byte_matrix, matrix_hashes
from v75_corrective import stable
from v99_normalization_feasibility import normalize
from v104_phase_a import FID, FOLDS, X
from v104_phase_b import ASA_IDS, PREP
from v106_frozen_wrapper_audit import DEST, cooked, rewrite


def main():
    target = DEST/'verification_and_N2_preflight.json'
    assert not target.exists()
    reg = json.loads((DEST/'registration.json').read_text())
    diag = json.loads((DEST/'diagnosis.json').read_text())
    checks = {'audit_source_matches_registration': sha(ROOT/'training/v106_frozen_wrapper_audit.py') == reg['source_sha256'],
              'bound_inputs_unchanged': all(sha(ROOT/p)==h for p,h in reg['input_sha256'].items()),
              'bound_outputs_unchanged': all(sha(DEST/p)==h for p,h in diag['outputs_sha256'].items())}
    d = pd.read_parquet(DEST/'paired_OOF_predictions.parquet')
    roots = {}
    for mode in ('collapse','expand'):
        a=d.C_TabM_epoch25.to_numpy(); b=d[mode+'_pred'].to_numpy(); y=d.truth.to_numpy()
        s=diag['variants'][mode]['all_ASA']
        checks[mode+'_counts_recomputed']=bool(int((a!=b).sum())==s['flips'] and
            int((y!=b).sum())==s['after_errors'] and
            int(((a==y)&(b!=y)).sum())==s['regressions'])
        z=d.assign(flip=a!=b, regression=(a==y)&(b!=y), repair=(a!=y)&(b==y))
        roots[mode]={'flipped_roots':int(z.loc[z.flip,'root'].nunique()),
                     'regressed_roots':int(z.loc[z.regression,'root'].nunique()),
                     'repaired_roots':int(z.loc[z.repair,'root'].nunique())}
    r=pd.read_parquet(OUT/'rows.parquet',columns=['row_position','route','new_text_id','source_symbol','label_index'])
    f=pd.read_parquet(FOLDS,columns=['proposed_fold','root'])
    fid=np.load(FID,mmap_mode='r'); ids=np.load(ASA_IDS)
    asa=r.route.eq('asa').to_numpy(); positions=np.flatnonzero(asa)
    assert np.array_equal(positions,d.row_position)
    strings=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    span=pd.read_parquet(DEST/'variant_span_ledger.parquet')
    pairs=span[span['mode']=='collapse'][['before_N1','after_N1']].drop_duplicates()
    assert pairs.groupby('before_N1').after_N1.nunique().max()==1
    replacement=dict(zip(pairs.before_N1,pairs.after_N1))
    before=[normalize(stable(strings.loc[int(i)]),'placeholder_cluster') for i in r.new_text_id.iloc[positions]]
    after=[replacement.get(t,t) for t in before]
    local=np.searchsorted(ids,fid[positions])
    assert np.array_equal(ids[local],fid[positions])
    combos=pd.DataFrame({'local':local,'text':after}).drop_duplicates().reset_index(drop=True)
    map_pair={(int(a),b):i for i,(a,b) in enumerate(combos.itertuples(index=False,name=None))}
    row_pair=np.array([map_pair[(int(a),b)] for a,b in zip(local,after)])
    n1=sparse.load_npz(PREP/'N1_ASA.npz')
    n2=sparse.hstack([byte_matrix(combos.text),n1[combos.local.to_numpy(),BYTE_FEATURES:]],format='csr')
    n2_hash=matrix_hashes(n2)
    # Check physical equality when a hash is shared across N2 representations.
    first={}
    for k,h in enumerate(n2_hash):
        if h in first: assert (n2[k]-n2[first[h]]).nnz==0
        else: first[h]=k
    current=load_sparse(X)
    non_asa_fid=np.unique(fid[~asa])
    shared=[]
    for start in range(0,len(non_asa_fid),4096):
        sub=non_asa_fid[start:start+4096]
        for oldid,h in zip(sub,matrix_hashes(current[sub])):
            if h in first:
                assert (current[int(oldid)]-n2[first[h]]).nnz==0
                shared.append((int(oldid),first[h]))
    expected=pd.read_parquet(DEST/'collapse_input_groups.parquet')
    group_ids,_=pd.factorize(pd.Series([n2_hash[i] for i in row_pair]),sort=False)
    assert np.array_equal(group_ids,expected.new_input.to_numpy())
    non_frame=pd.DataFrame({'fid':fid[~asa],'fold':f.proposed_fold.to_numpy()[~asa]})
    cross=[]
    for oldid,newid in shared:
        a=set(non_frame.loc[non_frame.fid==oldid,'fold'])
        b=set(f.proposed_fold.to_numpy()[positions][group_ids==group_ids[np.flatnonzero(row_pair==newid)[0]]])
        if len(a|b)>1: cross.append((oldid,newid))
    raw=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['message_sanitized','src_ip']).iloc[positions]
    unique=pd.unique(raw.message_sanitized)
    invariant=0
    for t in unique:
        a=cooked(rewrite(t,'collapse')[0]); b=cooked(rewrite(rewrite(t,'expand')[0],'collapse')[0])
        assert a==b
        invariant+=1
    checks['N2_collapse_expand_invariance_all_unique_raw']=True
    body=raw.message_sanitized.str.extract(r'\bsrc\s+[^\s:]+:([^\s/]+)',expand=False).to_numpy()
    mapping=pd.DataFrame({'structured':raw.src_ip.to_numpy(),'body':body})
    structured_to_body=mapping.groupby('structured').body.nunique(dropna=False)
    body_to_structured=mapping.groupby('body',dropna=False).structured.nunique(dropna=False)
    result={'status':'independent_no_fit_verification_and_preflight','source_sha256':sha(__file__),
        'checks':checks,'all_checks_passed':all(checks.values()),'variant_root_counts':roots,
        'all_format_equivalence':{'non_ASA_distinct_inputs_checked':int(len(non_asa_fid)),
            'N2_distinct_row_text_pairs':int(len(combos)),'N2_to_non_ASA_equal_inputs':len(shared),
            'N2_to_non_ASA_crossfold_equal_inputs':len(cross),
            'ASA_internal_crossfold_equal_inputs':diag['input_equivalence_audit']['collapse']['crossfold_equal_inputs'],
            'existing_folds_reusable_for_this_exact_N2':not shared and diag['input_equivalence_audit']['collapse']['crossfold_equal_inputs']==0},
        'N2_invariance_unique_raw_checked':invariant,
        'source_identity_check':{'ASA_rows':len(raw),'body_src_missing_rows':int(pd.isna(body).sum()),
            'structured_src_ip_with_multiple_body_sources':int((structured_to_body>1).sum()),
            'body_sources_with_multiple_structured_src_ip':int((body_to_structured>1).sum()),
            'structured_src_ip_distinct':int(len(structured_to_body)),'body_src_distinct':int(len(body_to_structured)),
            'warning':'Neither identifier is certified as a real actor. Do not merge them for context.'},
        'fits':0,'private_answers_read':False,'models_promoted':0,
        'limitations':['N2 feature invariance does not establish classification benefit.',
                       'Unchanged exact-input conflict floor does not establish stable M/S identifiability across sources.']}
    assert all(checks.values())
    save(target,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
