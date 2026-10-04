"""No-fit diagnosis of frozen v82 errors, objective, basis and source support."""
import json
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v79_execute import rows
from v82_capacity import DEST as PREV, LAST, counts_for

DEST=ROOT/'artifacts/v83_root_review_20260927'


def quantiles(a):return np.quantile(np.asarray(a),[0,.25,.5,.75,1]).tolist() if len(a) else []


def main():
    if (DEST/'review_receipt.json').exists():raise FileExistsError('Published review is frozen')
    DEST.mkdir(exist_ok=True);r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy')
    x=load_sparse(LAST/'X');kernel=np.load(PREV/'primary_kernel.npy',mmap_mode='r')
    models={a:joblib.load(PREV/(read(PREV/('primary_'+a+'_fit.json'))['states'][-1]['name']+'.joblib')) for a in ['linear','nonlinear']}
    def logits(model,ids):
        z=np.asarray(x[ids]@model['coef'])+model['intercept']
        if len(model['kernel_coef']):z+=kernel[ids]@model['kernel_coef']
        return z
    pred={a:np.load(PREV/(read(PREV/('primary_'+a+'_fit.json'))['states'][-1]['name']+'_prediction.npy'))[fid] for a in models}
    selected=np.zeros(len(r),bool);selected[np.load(PREV/'selected_rows.npy')]=True
    counts=counts_for(fid,y,selected,x.shape[0]);used=np.flatnonzero(counts.sum(1))
    fit=~r.fold.isin([0,2]).to_numpy();roles={'fit':fit,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    aa,bb=pred['linear'],pred['nonlinear'];changed=(aa!=bb)
    # Encoded-input support uses only rows that actually contributed to fitting.
    # Labels here are evaluation diagnostics and must never become a routing feature.
    support_records=[]
    support=counts[fid];same=support[np.arange(len(r)),y];other=support.sum(1)-same
    buckets=np.select([(same>0)&(other==0),(same>0)&(other>0),(same==0)&(other>0)],
                      ['same_only','mixed','opposite_only'],default='unseen')
    for role,mask in roles.items():
        for cls in range(3):
            for bucket in ['same_only','mixed','opposite_only','unseen']:
                take=mask&(y==cls)&(buckets==bucket)
                support_records.append({'role':role,'class':cls,'input_support':bucket,'rows':int(take.sum()),
                    'linear_errors':int(((aa!=y)&take).sum()),'nonlinear_errors':int(((bb!=y)&take).sum())})
    save(DEST/'encoded_support.json',support_records)
    entries=[];components=[];cluster_ci={};rng=np.random.default_rng(8301)
    for role,mask in roles.items():
        effect=((aa!=y).astype(int)-(bb!=y).astype(int))[mask]
        frame=pd.DataFrame({'component':r.component.to_numpy()[mask],'effect':effect})
        sums=frame.groupby('component').effect.sum();nz=sums[sums.ne(0)]
        n=len(sums);prob=np.r_[np.full(len(nz),1/n),1-len(nz)/n]
        draws=rng.multinomial(n,prob,size=10000)[:,:len(nz)]@nz.to_numpy()
        cluster_ci[role]={'observed_net_repair':int(effect.sum()),'components':n,'nonzero_effect_components':len(nz),
                          'paired_component_bootstrap_95pct_net_repair':np.quantile(draws,[.025,.975]).tolist(),
                          'scope':'Exploratory component bootstrap, proxy components not verified independent organizations; not a new acceptance threshold.'}
        for comp,value in nz.items():components.append({'role':role,'component':int(comp),'net_repair':int(value)})
        for kind,take in [('repair',mask&(aa!=y)&(bb==y)),('regression',mask&(aa==y)&(bb!=y))]:
            ix=np.flatnonzero(take);ids=fid[ix];za=logits(models['linear'],ids);zb=logits(models['nonlinear'],ids)
            k=kernel[ids]@models['nonlinear']['kernel_coef']
            for j,i in enumerate(ix):
                row=r.iloc[i]
                entries.append({'row_position':int(i),'role':role,'change':kind,'class':int(y[i]),'component':int(row.component),
                    'body_group':int(row.body_group),'route':row.route,'feature_id':int(fid[i]),'projection_id':int(row.projection_id),
                    'old_pred':int(aa[i]),'new_pred':int(bb[i]),'old_M_minus_S':float(za[j,1]-za[j,2]),
                    'new_M_minus_S':float(zb[j,1]-zb[j,2]),'kernel_M_minus_S':float(k[j,1]-k[j,2]),
                    'changed_linear_part_M_minus_S':float((zb[j,1]-zb[j,2])-(za[j,1]-za[j,2])-(k[j,1]-k[j,2]))})
    transitions=pd.DataFrame(entries);transitions.to_parquet(DEST/'changed_rows.parquet',index=False)
    pd.DataFrame(components).to_csv(DEST/'component_changes.csv',index=False)
    # Objective versus classification: use the actual training exposure and exact
    # original per-label frequency, not the much larger unsampled benign pool.
    objective=[]
    for state in read(PREV/'primary_nonlinear_fit.json')['states']:
        model=joblib.load(PREV/(state['name']+'.joblib'));z=logits(model,used);p=z.argmax(1)
        terms=np.logaddexp(0,z).sum(1)[:,None]-z
        class_loss=(counts[used]*terms).sum(0)/counts.sum(0)
        l2=1e-6/2*(np.square(model['coef']).sum()+np.square(model['kernel_coef']).sum())
        objective.append({'name':state['name'],'mean_OVR_loss_by_true_class':class_loss.tolist(),
            'training_class_errors':[(int(counts[used,k].sum()-counts[used,k][p==k].sum())) for k in range(3)],
            'data_loss':float((counts[used]*terms).sum()/counts.sum()),'L2_penalty':float(l2),
            'objective':float((counts[used]*terms).sum()/counts.sum()+l2)})
    # Inspect only the known fit cohort: exact same-label support exists, and the
    # input RBF contrast to its closest M example may be lost by the low-rank basis.
    cohort=np.load(PREV/'fit_457_cohort.npy');remaining=cohort[bb[cohort]!=2]
    uq,repeat=np.unique(fid[remaining],return_counts=True);mids=used[counts[used,1]>0]
    distances=euclidean_distances(x[uq],x[mids],squared=True)
    near=mids[distances.argmin(1)];nearest=distances.min(1)
    basis=joblib.load(PREV/'primary_basis.joblib')
    full_dist=2*(-np.expm1(-basis.gamma*nearest))
    approx=np.square(np.asarray(kernel[uq],float)-np.asarray(kernel[near],float)).sum(1)
    ratio=np.divide(approx,full_dist,out=np.zeros(len(uq)),where=full_dist>1e-10)
    assert (counts[uq,2]>0).all() and (counts[uq,:2].sum(1)==0).all()
    landmarks=np.load(PREV/'primary_landmark_ids.npy')
    pd.DataFrame({'feature_id':uq,'rows':repeat,'nearest_M_feature_id':near,'input_squared_distance':nearest,
                  'exact_RBF_distance_squared':full_dist,'approx_RBF_distance_squared':approx,'retained_RBF_pair_distance_ratio':ratio,
                  'is_landmark':np.isin(uq,landmarks)}).to_csv(DEST/'remaining_S_basis.csv',index=False)
    projection=pd.read_parquet(OUT/'projections.parquet',columns=['facts'])
    facts=[json.loads(s) for s in projection.facts]
    # Bind nearest pairs to actual fit rows of the indicated class, rather than
    # arbitrary global representatives that could belong to another role.
    _,first_s=np.unique(fid[remaining],return_index=True);srows=remaining[first_s]
    mrows_all=np.flatnonzero(selected&(y==1));mid_unique,first_m=np.unique(fid[mrows_all],return_index=True)
    mrows=mrows_all[first_m[np.searchsorted(mid_unique,near)]]
    assert np.array_equal(fid[srows],uq) and np.array_equal(fid[mrows],near)
    pair_records=[];diff_counter=Counter();port_only=0;same_facts=0
    port_keys={'src_port_fixed','dst_port_fixed','src_port_range','dst_port_range'}
    for srow,mrow,n in zip(srows,mrows,repeat):
        fs=facts[int(r.projection_id.iloc[srow])];fm=facts[int(r.projection_id.iloc[mrow])]
        diff={key:[fs.get(key),fm.get(key)] for key in sorted(set(fs)|set(fm)) if fs.get(key)!=fm.get(key)}
        category='same_facts' if not diff else 'only_port_fields' if set(diff).issubset(port_keys) else 'other_fact_differences'
        diff_counter[category]+=int(n)
        pair_records.append({'S_fit_row':int(srow),'M_fit_row':int(mrow),'S_rows':int(n),
                             'S_component':int(r.component.iloc[srow]),'M_component':int(r.component.iloc[mrow]),
                             'S_route':r.route.iloc[srow],'M_route':r.route.iloc[mrow],
                             'fact_difference_category':category,'fact_differences':diff})
    save(DEST/'nearest_fit_class_pairs.json',pair_records)
    # Match only observed common flow facts. Missing fields remain missing.
    core=['action','transport_protocol','src_port_fixed','dst_port_fixed','icmp_type','icmp_code']
    keys=[json.dumps({k:f[k] for k in core if k in f},sort_keys=True) for f in facts]
    code,unique=pd.factorize(np.asarray(keys,object),sort=False);rowcodes=code[r.projection_id.to_numpy()]
    other=selected&~r.route.eq('vpc_v2').to_numpy();c=counts_for(rowcodes,y,other,len(unique))
    vp=selected&r.route.eq('vpc_v2').to_numpy()&(y==2);matched=c[rowcodes[vp]]
    vpc_support={'actual_VPC_S_rows':int(vp.sum()),'shared_fields':core,
         'no_other_format_support':int((matched.sum(1)==0).sum()),
         'only_other_format_M_support':int(((matched[:,1]>0)&(matched[:,[0,2]].sum(1)==0)).sum()),
         'any_other_format_M_support':int((matched[:,1]>0).sum()),'any_other_format_S_support':int((matched[:,2]>0).sum()),
         'scope':'Common-field agreement omits format-only evidence; these are support diagnostics, not valid cross-format relabeling.'}
    # Raw records for every changed decision plus one representative per remaining
    # S group; retain evidence locally, never use these H labels to train a fix.
    wanted=np.unique(np.r_[transitions.row_position.to_numpy(),srows,mrows]);raw=[];offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['event_id','message_sanitized'],use_threads=False):
        ix=wanted[np.searchsorted(wanted,offset):np.searchsorted(wanted,offset+len(batch))]
        for pos in ix:
            record=r.iloc[int(pos)]
            raw.append({'row_position':int(pos),'event_id':batch.column(0)[int(pos)-offset].as_py(),
                        'message':batch.column(1)[int(pos)-offset].as_py(),'facts':facts[int(record.projection_id)],
                        'label':int(y[pos]),'fold':int(record.fold),'component':int(record.component)})
        offset+=len(batch)
    save(DEST/'raw_casebook.json',raw)
    out={'new_classifier_fits':0,'target_answers_read':False,'source_sha256':sha(__file__),
         'paired_cluster_analysis':cluster_ci,'objective_trajectory':objective,
         'remaining_S':{'rows':len(remaining),'unique_encoded_inputs':len(uq),'exact_selected_fit_same_label_support_verified':True,
                        'direct_landmarks':int(np.isin(uq,landmarks).sum()),
                        'nearest_M_input_squared_distance_quantiles':quantiles(np.repeat(nearest,repeat)),
                        'retained_RBF_pair_distance_ratio_quantiles':quantiles(np.repeat(ratio,repeat)),
                        'nearest_M_fact_difference_row_counts':dict(diff_counter),
                        'scope':'Ratio measures new RBF branch approximation, not loss of the retained original linear columns. Nearest-pair differences are not causal threat annotations.'},
         'VPC_shared_observation_support':vpc_support,
         'zero_M_confound':{'primary_class_counts':[17153,67448,19943],'zero_M_class_counts':[17153,19904,19943],
                           'primary_M_fraction':67448/104544,'zero_M_M_fraction':19904/57000,
                           'changed_factors':['M labels/behavior support','class priors','loss denominator','kernel landmark pool'],
                           'pure_source_shortcut_causality_established':False},
         'transition_summary':transitions.groupby(['role','change','class']).agg(rows=('row_position','size'),components=('component','nunique'),encoded_inputs=('feature_id','nunique')).reset_index().to_dict('records')}
    save(DEST/'diagnosis.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
