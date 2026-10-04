"""Trace every predefined error, original span and model input before fitting."""
import collections,difflib,functools,json,re,time
import numpy as np,pandas as pd,pyarrow.parquet as pq,joblib
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from v92_common import *
from v75_corrective import stable
from v75_views import view,byte_matrix,BYTE_FEATURES
from v89_partial_facts import parse
from v89_partial_expression import tokens
from run_v75 import load_sparse

def ordered_tokens(a):
    """Known native field associations, never identity, clock, partial port or label."""
    f={k:v for k,v in a['facts'].items() if a['states'].get(k)=='known'}
    if f.get('action')!='deny':return {}
    z={'atom:'+k+'='+str(v):1. for k,v in f.items()}
    for names in [('transport_protocol','src_role','dst_role'),('transport_protocol','src_role','src_port_fixed'),
                  ('transport_protocol','dst_role','dst_port_fixed'),('src_role','src_port_fixed','dst_role','dst_port_fixed')]:
        if all(k in f for k in names):z['bound:'+'|'.join(k+'='+str(f[k]) for k in names)]=1.
    return z

def main():
    assert not DEST.exists();DEST.mkdir();start=time.monotonic();event('preparation','started')
    r,y,fid,z,old,sel,fit=data();inner=r.fold.eq(1).to_numpy();asa=r.route.eq('asa').to_numpy()
    es=np.load(ROOT/'artifacts/v91_first_principles_20260928/support_S_error_rows.npy')
    obs=np.load(V89/'row_fact_code.npy');sets=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    parsed=[json.loads(s) for s in sets];literal=[json.dumps(sorted(tokens(s))) for s in sets]
    ocode,_=pd.factorize(literal,sort=True);lk=ocode[obs]
    new=np.load(V89/'F00_all_prediction.npy')[np.load(V89/'F00_row_group.npy')]
    em=np.flatnonzero(inner&(y==1)&(old[fid]==1)&(new!=1));assert len(es)==52 and len(em)==668
    target=np.unique(np.r_[es,em]);targetset=set(target.tolist());raws={};stress={};counts=collections.Counter();offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=16384,columns=['message_sanitized','label_binary'],use_threads=False):
        b=batch.to_pandas();rr=r.iloc[offset:offset+len(b)];yy=b.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();assert np.array_equal(yy,y[rr.index])
        for i,text,route,cls in zip(rr.index,b.message_sanitized,rr.route,yy):
            counts[(str(route),int(cls))]+=1
            if i in targetset:raws[int(i)]=text
            key=(str(route),int(cls))
            if key not in stress:stress[key]=(int(i),text)
        offset+=len(b)
        if offset%524288==0:emit(stage='source_trace',rows=offset,seconds=round(time.monotonic()-start,1))
    assert offset==len(r) and len(raws)==len(target)
    # One representative competing original TRAINING row per exact R0 error input.
    x=load_sparse(LAST/'X');texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    oldfacts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet').facts]
    groups=[];trace=[];competitors=set();candidateM=np.flatnonzero(fit&asa&(y==1));candidateS=np.flatnonzero(fit&asa&(y==2))
    for error_kind,errors,competition in [('support_S',es,candidateM),('F00_M_regression',em,candidateS)]:
        for f in np.unique(fid[errors]):
            ii=errors[fid[errors]==f];i=int(ii[0]);same=competition[lk[competition]==lk[i]];level='same_known_facts'
            if not len(same):
                fi=parsed[obs[i]]['facts'];fields=['action','transport_protocol','src_role','dst_role','dst_port_fixed']
                same=np.array([j for j in competition if all(parsed[obs[j]]['facts'].get(k)==fi.get(k) for k in fields)],dtype=int);level='same_coarse_behavior'
            j=None
            if len(same):
                ids,first=np.unique(fid[same],return_index=True);sim=np.asarray(x[ids]@x[f].T.toarray()).ravel();j=int(same[first[sim.argmax()]]);competitors.add(j)
            t=stable(texts.loc[r.new_text_id.iloc[i]]);diff=x[f,:BYTE_FEATURES]-byte_matrix([t]);assert diff.nnz==0 or np.abs(diff.data).max()<1e-7
            p=parse(raws[i],'asa');assert p['facts']==parsed[obs[i]]['facts'] and p['states']==parsed[obs[i]]['states']
            known={k:v for k,v in p['facts'].items() if p['states'].get(k)=='known'}
            missing={k:v for k,v in known.items() if oldfacts[r.projection_id.iloc[i]].get(k)!=v}
            rec={'kind':error_kind,'R0_id':int(f),'representative_row':i,'error_rows':ii.tolist(),'error_row_count':len(ii),
                'current_text':t,'known_facts':known,'observed_spans':p['observations'],'known_facts_not_preserved_in_old_fact_dict':missing,
                'training_same_literal_class_rows':np.bincount(y[fit&(lk==lk[i])],minlength=3).tolist(),
                'training_same_literal_class_components':[int(r.component[fit&(lk==lk[i])&(y==c)].nunique()) for c in range(3)],
                'competitor_row':j,'competitor_match_level':level if j is not None else 'no_training_competitor_at_this_behavior',
                'evidence_grade':'known_field_binding_expression_gap' if missing else 'full_input_difference_semantics_not_proven',
                'input_coordinate_gap_scope':'Old fact coordinates absent/conflicting; byte grams may retain literal characters, not evidence of causal loss.'}
            if j is not None:
                other=stable(texts.loc[r.new_text_id.iloc[j]]);rec.update(competitor_current_text=other,
                    current_text_differences=[{'error_span':[a,b],'competitor_span':[c,d],'error_literal':t[a:b],'competitor_literal':other[c:d]} for tag,a,b,c,d in difflib.SequenceMatcher(None,t,other,autojunk=False).get_opcodes() if tag!='equal'])
                rec['raw_difference_semantics_confirmed']=False
            groups.append(rec)
            for row in ii:trace.append({'row_position':int(row),'kind':error_kind,'R0_id':int(f),'label':int(y[row]),'component':int(r.component.iloc[row]),'source_raw_sha256':__import__('hashlib').sha256(raws[int(row)].encode()).hexdigest(),'competitor_row':j,'evidence_grade':rec['evidence_grade']})
    pd.DataFrame(trace).to_parquet(DEST/'error_row_trace.parquet',index=False);save(DEST/'input_group_trace.json',groups)
    # Increment only supplies observed named associations beside complete R0.
    # Vocabulary is fitted on actual parameter-training rows, without evaluation labels.
    inc=DictVectorizer(dtype=np.float32,sparse=True);fitobs=np.unique(obs[fit&asa]);inc.fit([ordered_tokens(parsed[k]) for k in fitobs])
    dz=normalize(inc.transform([ordered_tokens(p) for p in parsed]),copy=False)
    # This feature contract must not collapse multiple row observations to consensus.
    asaids=np.unique(fid[asa]);assert not np.isin(fid[~asa],asaids).any(),'ASA and other routes share exact R0; route gate required';representatives=[]
    for f,rr in r[asa].assign(obs=obs[asa],fid=fid[asa]).groupby('fid'):
        ix=rr.index.to_numpy();literalset={json.dumps(ordered_tokens(parsed[obs[i]]),sort_keys=True) for i in ix}
        assert len(literalset)==1,('row_level_increment_conflicts',int(f))
        representatives.append(int(ix[0]))
    representatives=np.asarray(representatives);assert np.array_equal(fid[representatives],asaids)
    xx=x[asaids];xi=dz[obs[representatives]].tocsr();sparse.save_npz(DEST/'ASA_R0.npz',xx);sparse.save_npz(DEST/'ASA_increment.npz',xi)
    np.save(DEST/'ASA_input_ids.npy',asaids);joblib.dump(inc,DEST/'increment_encoder.joblib')
    fitcount=raw_counts(fid,y,fit,len(z));np.save(DEST/'fit_counts.npy',fitcount)
    fullweight=fitcount[asaids];np.save(DEST/'ASA_fit_counts.npy',fullweight)
    rolecounts=[]
    for (route,cls),n in sorted(counts.items()):
        m=r.route.eq(route).to_numpy()&(y==cls);rolecounts.append({'route':route,'class':cls,'all_rows':n,'fit_rows':int((m&fit).sum()),'fit_original_mass_used':int((m&fit).sum()),'correction_enabled':route=='asa','other_formats_loss_constant_but_supervised':route!='asa'})
    pd.DataFrame(rolecounts).to_csv(DEST/'format_supervision.csv',index=False)
    stresses=[]
    for (route,cls),(row,text) in stress.items():
        if not isinstance(text,str):continue
        renamed=re.sub(r'((?:USER|HOST|CRED|ORG)-)(\d+)',lambda m:m[1]+str(int(m[2])+9999999),text)
        a=stable(view(text)[0]);b=stable(view(renamed)[0]);pa=parse(text,route);pb=parse(renamed,route)
        stresses.append({'route':route,'class':cls,'row':row,'corrective_identity_invariant':a==b,'increment_identity_invariant':ordered_tokens(pa)==ordered_tokens(pb),'product_empty_invariant':'No product/vendor column used','span_literals_checked':all(text[o['span'][0]:o['span'][1]]==o['literal'] for o in pa['observations'])})
    save(DEST/'input_stress.json',stresses)
    oldnames=joblib.load(OUT/'facts_encoder.joblib').names().tolist()
    save(DEST/'representation_contract.json',{'increment':'Fit-only collision-free known ASA atoms and named field associations; always alongside complete R0, not independent new observations.',
        'new_information_claim':False,'new_expression_claim':True,'no_partial_port_reconstruction':True,'no_exact_IP_host_rule_clock_product':True,
        'fit_vocabulary_size':len(inc.vocabulary_),'known_missing_rows_in_target':sum(a['error_row_count'] for a in groups if a['known_facts_not_preserved_in_old_fact_dict']),
        'old_fact_coordinate_names':oldnames,'sequence_scope':'Known source/destination field bindings, not lossless arbitrary raw order','unseen_token_rule':'Zero incremental coordinate; full R0 preserved.'})
    result={'status':'complete','original_labels_checked':len(r),'S_error_rows_traced':len(es),'S_input_groups_traced':len(np.unique(fid[es])),
        'M_regression_rows_traced':len(em),'M_input_groups_traced':len(np.unique(fid[em])),'all_actual_route_class_cells':len(rolecounts),
        'actual_text_matrix_groups_reconstructed':len(groups),'ASA_all_inputs':len(asaids),'ASA_fit_inputs':int((fullweight.sum(1)>0).sum()),
        'increment_width':xi.shape[1],'identity_stress_failures':sum(not a['corrective_identity_invariant'] or not a['increment_identity_invariant'] for a in stresses),
        'facts_only_conflicting_target_rows':int((pd.Series([sum(a['training_same_literal_class_rows'][c] for c in range(3) if c!=int(y[a['representative_row']]))>0 for a in groups])*pd.Series([a['error_row_count'] for a in groups])).sum()),
        'source_sha256':sha(__file__),'seconds':time.monotonic()-start,'new_fits':0,'scope':'Every predefined720 error row mapped; each distinct input has one nearest training competitor where available. Literal gap proven, threat semantics and future transfer not proven.'}
    save(DEST/'preparation.json',result);event('preparation','completed',**result);emit(**result)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
