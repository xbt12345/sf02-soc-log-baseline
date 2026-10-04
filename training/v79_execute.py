"""Pre-registered official-only objective, boundary and transfer experiments.

Exact feature equality aggregation retains original row frequency and conflicting
labels. No evaluation or private-answer labels participate in objective selection.
"""
import argparse
import gc
import hashlib
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, load_sparse, SparseWriter, read, save, sha, metrics

DEST = ROOT / 'artifacts/v79_execution_20260927'
WIDTH = 65792 + 477 + 18
SEED = 7901
ALPHA = 1e-6
FOLDS = [(0, 2), (1, 3), (4, 1)]
FORMATS = ['native_flow', 'native_firewall', 'asa_protocol']


def emit(**x):
    print(__import__('json').dumps(x, ensure_ascii=False), flush=True)


def rows():
    return pd.read_parquet(OUT / 'rows.parquet', columns=[
        'row_position', 'projection_id', 'new_text_id', 'component', 'body_group',
        'source_symbol', 'fold', 'route', 'label_index'])


def register():
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir()
    c = {
        'version': 'v79-execution-1', 'seed': SEED, 'second_seed': 7902,
        'alpha': ALPHA, 'source_folds_H_C': FOLDS, 'format_protocols': FORMATS,
        'input': 'Frozen R0 corrected byte1/2 + facts + independent src_port; V78 v4 adapter for unsupported raw replay',
        'P1': ['A all fit rows, original frequency', 'B all M/S, <=2048 benign per format, sample row mean',
               'C same random sample as B, fixed full-fit N denominator, inverse inclusion probability'],
        'sampling': 'Fixed uniform-without-replacement draw, positive pi for every fit benign; seed7901 then independently7902. Fixed-draw HT estimate, not exact full risk or variance-free optimum.',
        'P2': ['OVR', 'joint softmax', 'joint softmax + lambda1 conditional M/S CE normalized on true nonbenign weights'],
        'solver': {'method': 'L-BFGS-B', 'maxiter': 1000, 'gtol': 1e-6, 'ftol': 1e-12, 'maxcor': 10,
                   'acceptance': 'success and gradient_inf<=1e-5; otherwise no convergence claim'},
        'selection': 'Primary calibration only. Baseline B-OVR; choose fewer errors + every class recall/precision/F1 nondecreasing + normal FP nonincreasing, then macroF1; failing guards retains baseline. Evaluation never chooses coefficients/targets.',
        'P4': 'Selected objective repeated on three prebound source folds, each carrier zero-M paired with same-fold supervised, whole-format holdout known schema, and generic residual-only for heldout format. All component isolation retained; absent classes NA.',
        'P3': 'Conditional only: inspect fit-side raw/encoded conflicts and evidence bindings before ordered encoder; no rerun of failed 13-field MLP',
        'shadow': 'Expand selected primary fit with calibration rows and repeat fixed sampling/objective. Require heldout each-class recall/precision/F1 nondecrease, FP no increase, total errors no increase before final all-record fit.',
        'development_gate': {'max_errors': 4340, 'min_malicious_tp': 11729, 'min_malicious_precision': .866687628,
                             'min_suspicious_recall': .956811042, 'min_suspicious_precision': .932814045, 'max_normal_fp': 34},
        'scope': 'Previously inspected official train folds and private target are development; no external training data, pseudo labels, target optimization or new blind-test claim.',
        'source_sha256': sha(__file__), 'plan_sha256': sha(ROOT / 'docs/V79_MS_BOUNDARY_AND_UNKNOWN_FORMAT_PLAN.md'),
        'input_sha256': {'rows': sha(OUT / 'rows.parquet'), 'train': sha(ROOT / 'data/official/train.parquet'),
                         'v78_v4': sha(ROOT / 'training/v78_denial_adapter_v4.py'),
                         'text_data': sha(OUT / 'corrective/text.data'), 'facts_data': sha(OUT / 'facts.data'),
                         'metadata_code': sha(OUT / 'metadata_code.npy')},
    }
    r = rows()
    for h, cal in FOLDS:
        role = np.where(r.fold.eq(h), 'H', np.where(r.fold.eq(cal), 'C', 'fit'))
        for field in ['component', 'body_group', 'source_symbol']:
            take = r[field] >= 0
            assert pd.DataFrame({'group': r.loc[take, field], 'role': role[take]}).groupby('group').role.nunique().max() == 1
    from v78_denial_adapter_v4 import self_check
    c['parser_negative_checks'] = self_check()
    c['support'] = r.groupby(['fold', 'route', 'label_index']).size().rename('rows').reset_index().to_dict('records')
    save(DEST / 'registration.json', c)
    emit(stage='registered', source_folds=FOLDS, parser_checks=c['parser_negative_checks'])


def check():
    c = read(DEST / 'registration.json')
    assert c['source_sha256'] == sha(__file__), 'source changed after registration'
    return c


def canonical(matrix, name):
    """Hash candidate then byte-compare every equal candidate (no hash-only merge)."""
    start = time.monotonic()
    mapping = np.empty(matrix.shape[0], np.int32)
    buckets = {}
    representatives = []
    for i in range(matrix.shape[0]):
        a, b = matrix.indptr[i:i+2]
        ind, dat = matrix.indices[a:b], matrix.data[a:b]
        digest = hashlib.blake2b(ind.tobytes() + dat.tobytes(), digest_size=16).digest()
        found = None
        for code in buckets.get(digest, []):
            j = representatives[code]; u, v = matrix.indptr[j:j+2]
            if np.array_equal(ind, matrix.indices[u:v]) and np.array_equal(dat, matrix.data[u:v]):
                found = code; break
        if found is None:
            found = len(representatives); representatives.append(i)
            buckets.setdefault(digest, []).append(found)
        mapping[i] = found
        if i and i % 100000 == 0:
            emit(stage='feature_equality', block=name, rows=i, unique=len(representatives), seconds=round(time.monotonic()-start, 1))
    return mapping, np.array(representatives, np.int32)


def pack():
    check()
    if (DEST / 'packed.json').exists():
        raise FileExistsError('packed cache exists')
    r = rows()
    text = load_sparse(OUT / 'corrective/text'); fact = load_sparse(OUT / 'facts')
    meta = load_sparse(OUT / 'metadata'); key = np.load(OUT / 'metadata_code.npy', mmap_mode='r')
    tc, tr = canonical(text, 'text'); fc, fr = canonical(fact, 'facts')
    keys = pd.MultiIndex.from_arrays([tc[r.new_text_id], fc[r.projection_id], key])
    code, unique = pd.factorize(keys, sort=False)
    del keys; gc.collect()
    _, first = np.unique(code, return_index=True)
    writer = SparseWriter(DEST / 'X', WIDTH)
    for beg in range(0, len(first), 2048):
        ix = first[beg:beg+2048]
        writer.append(sparse.hstack([text[r.new_text_id.to_numpy()[ix]], fact[r.projection_id.to_numpy()[ix]], meta[ix]], format='csr'))
    writer.close()
    np.save(DEST / 'row_feature_id.npy', code.astype(np.int32))
    np.save(DEST / 'feature_representative.npy', first.astype(np.int32))
    # Equality implies encoded input equality, not raw semantic equivalence or incident independence.
    out = {'original_rows': len(r), 'encoded_unique': len(first), 'text_unique': len(tr), 'fact_unique': len(fr),
           'retained_original_label_counts': np.bincount(r.label_index, minlength=3).tolist(),
           'exact_hash_and_byte_equality': True, 'original_frequency_retained': True,
           'x_bytes': sum((DEST / ('X'+s)).stat().st_size for s in ['.data', '.indices', '.indptr']),
           'row_feature_id_sha256': sha(DEST / 'row_feature_id.npy')}
    save(DEST / 'packed.json', out); emit(stage='packed', **out)


def load():
    check()
    return rows(), load_sparse(DEST / 'X'), np.load(DEST / 'row_feature_id.npy', mmap_mode='r')


def weights(r, fid, n, fit, arm, seed=SEED):
    rng = np.random.default_rng(seed)
    sample = fit.copy(); pi = np.ones(len(r), np.float64)
    if arm != 'A':
        for route in sorted(r.route.unique()):
            ix = np.flatnonzero(fit & r.route.eq(route).to_numpy() & r.label_index.eq(0).to_numpy())
            if len(ix) > 2048:
                sample[ix] = False; sample[rng.choice(ix, 2048, replace=False)] = True
                pi[ix] = 2048 / len(ix)
    ix = np.flatnonzero(sample)
    rowweights = 1 / pi[ix] if arm == 'C' else np.ones(len(ix))
    counts = np.bincount(fid[ix]*3 + r.label_index.to_numpy()[ix], weights=rowweights, minlength=n*3).reshape(n, 3)
    denom = int(fit.sum()) if arm == 'C' else len(ix)
    assert np.array_equal(sample & r.label_index.ne(0).to_numpy(), fit & r.label_index.ne(0).to_numpy())
    return counts, denom, {'fit_population_rows': int(fit.sum()), 'exposed_original_rows': len(ix),
                          'exposed_per_class': np.bincount(r.label_index.to_numpy()[ix], minlength=3).tolist(),
                          'estimated_mass_per_class': counts.sum(0).tolist(), 'denominator': denom,
                          'min_fit_pi': float(pi[fit].min()), 'selected_rows_sha256': hashlib.sha256(ix.astype('<i4').tobytes()).hexdigest()}


def objective(flat, x, counts, denom, kind):
    w = flat.reshape(WIDTH+1, 3)
    total = counts.sum(1)
    z = x @ w[:-1] + w[-1]
    if kind == 'ovr':
        loss = (total @ np.logaddexp(0, z).sum(1) - (counts*z).sum()) / denom
        res = (expit(z)*total[:, None] - counts) / denom
    else:
        lse = logsumexp(z, axis=1)
        loss = (total @ lse - (counts*z).sum()) / denom
        res = (np.exp(z-lse[:, None])*total[:, None] - counts) / denom
        if kind == 'ms':
            non = counts[:, 1:].sum(1); nd = non.sum()
            if nd:
                logms = logsumexp(z[:, 1:], axis=1)
                loss += (non @ logms - (counts[:, 1:]*z[:, 1:]).sum()) / nd
                res[:, 1:] += (np.exp(z[:, 1:]-logms[:, None])*non[:, None]-counts[:, 1:])/nd
    loss += ALPHA/2 * np.square(w[:-1]).sum()
    g = np.empty_like(w); g[:-1] = np.asarray(x.T@res) + ALPHA*w[:-1]; g[-1] = res.sum(0)
    return float(loss), g.ravel()


def solve(x, counts, denom, kind, name, exposure):
    path = DEST / (name+'.joblib')
    if path.exists():
        raise FileExistsError(path)
    used = np.flatnonzero(counts.sum(1) > 0)
    xx = x[used]; cc = counts[used]
    # Small feature-bounded directional derivative check, independent central difference.
    w = np.zeros((WIDTH+1)*3, np.float64)
    w[-3:] = np.log(np.maximum(cc.sum(0)/cc.sum(), 1e-9))
    rng = np.random.default_rng(SEED)
    d = rng.normal(size=len(w)); d /= np.linalg.norm(d)
    f, g = objective(w, xx[:20], cc[:20], denom, kind)
    eps = 1e-5
    numeric = (objective(w+eps*d, xx[:20], cc[:20], denom, kind)[0]-objective(w-eps*d, xx[:20], cc[:20], denom, kind)[0])/(2*eps)
    assert abs(numeric-g@d) < 1e-6
    start = time.monotonic(); it = [0]; trace = []
    def callback(v):
        it[0] += 1
        if it[0] % 25 == 0:
            f, g = objective(v, xx, cc, denom, kind)
            item = {'model': name, 'iteration': it[0], 'objective': f, 'gradient_inf': float(abs(g).max()), 'seconds': round(time.monotonic()-start, 1)}
            trace.append(item); save(DEST/(name+'_progress.json'), trace); emit(**item)
    result = minimize(objective, w, args=(xx, cc, denom, kind), method='L-BFGS-B', jac=True, callback=callback,
                      options={'maxiter': 1000, 'maxls': 30, 'gtol': 1e-6, 'ftol': 1e-12, 'maxcor': 10})
    ww = result.x.reshape(WIDTH+1, 3)
    model = {'coef': ww[:-1], 'intercept': ww[-1], 'kind': kind, 'arm': exposure.get('arm')}
    joblib.dump(model, path)
    report = {'name': name, 'kind': kind, 'iterations': int(result.nit), 'objective': float(result.fun),
              'gradient_inf': float(abs(result.jac).max()), 'converged': bool(result.success and abs(result.jac).max()<=1e-5),
              'scipy_success': bool(result.success), 'solver_message': str(result.message), 'seconds': time.monotonic()-start,
              'directional_gradient_error': float(abs(numeric-g@d)), 'exposure': exposure,
              'unique_inputs_used': len(used), 'model_sha256': sha(path)}
    # g in callback is scoped locally; outer g remains the gradient checked above.
    save(DEST/(name+'_fit.json'), report); emit(stage='fit_finished', **report)
    return model


def score(r, fid, p, mask):
    y = r.label_index.to_numpy()[mask]; pred = p[fid[mask]]
    return metrics(np.bincount(y*3+pred, minlength=9).reshape(3, 3))


def evaluate(r, x, fid, model, h, c, name):
    p = (x@model['coef']+model['intercept']).argmax(1).astype(np.int8)
    np.save(DEST/(name+'_feature_prediction.npy'), p)
    out = {}; table = []
    for role, mask in [('fit', ~r.fold.isin([h,c]).to_numpy()), ('calibration', r.fold.eq(c).to_numpy()), ('evaluation', r.fold.eq(h).to_numpy())]:
        out[role] = score(r, fid, p, mask)
        for route in ['ALL'] + sorted(r.route.unique()):
            take = mask if route == 'ALL' else mask & r.route.eq(route).to_numpy()
            result = score(r, fid, p, take); cm = np.array(result['cm'])
            for k in range(3):
                table.append({'model': name, 'role': role, 'route': route, 'class': k, 'support': int(cm[k].sum()), 'correct': int(cm[k,k]),
                              'recall': result['recall'][k], 'precision': result['precision'][k], 'f1': result['f1'][k],
                              'pred_B': int(cm[k,0]), 'pred_M': int(cm[k,1]), 'pred_S': int(cm[k,2])})
    pd.DataFrame(table).to_csv(DEST/(name+'_classwise.csv'), index=False)
    save(DEST/(name+'_scores.json'), out)
    return out


def guards(a, b, strict=True):
    out = {'errors': b['errors'] < a['errors'] if strict else b['errors'] <= a['errors'],
           'normal_fp': b['normal_false_alerts'] <= a['normal_false_alerts']}
    for key in ['recall', 'precision', 'f1']:
        out[key] = all(y is not None and y >= v-1e-12 for v,y in zip(a[key], b[key]) if v is not None)
    return out


def choose(names, reference, file):
    baseline = read(DEST/(reference+'_scores.json'))['calibration']
    candidates = []
    for name in names:
        met = read(DEST/(name+'_scores.json'))['calibration']
        gate = guards(baseline, met)
        ok = all(gate.values()) and read(DEST/(name+'_fit.json'))['converged']
        candidates.append({'name': name, 'metrics': met, 'guards': gate, 'eligible': ok})
    eligible = [v for v in candidates if v['eligible']]
    selected = min(eligible, key=lambda v:(v['metrics']['errors'], -v['metrics']['macro_f1']))['name'] if eligible else reference
    save(DEST/file, {'reference': reference, 'selected': selected, 'eligible_improvement': bool(eligible), 'candidates': candidates, 'scope': 'Calibration labels only'})
    emit(stage='selected', selection=file, selected=selected, eligible_improvement=bool(eligible))
    return selected


def primary():
    r, x, fid = load(); h,c = FOLDS[0]; fit = ~r.fold.isin([h,c]).to_numpy()
    with threadpool_limits(limits=4):
        for arm in ['A','B','C']:
            name = 'P1_'+arm+'_ovr'; counts, denom, e = weights(r, fid, x.shape[0], fit, arm)
            e['arm'] = arm
            m = solve(x, counts, denom, 'ovr', name, e); evaluate(r,x,fid,m,h,c,name)
        sel = choose(['P1_A_ovr','P1_C_ovr'], 'P1_B_ovr', 'P1_selection.json')
        arm = sel.split('_')[1]
        counts, denom, e = weights(r,fid,x.shape[0],fit,arm); e['arm']=arm
        for kind in ['softmax','ms']:
            name = 'P2_'+arm+'_'+kind
            m=solve(x,counts,denom,kind,name,e); evaluate(r,x,fid,m,h,c,name)
        choose(['P2_'+arm+'_softmax','P2_'+arm+'_ms'],sel,'P2_selection.json')


def selected_spec():
    name = read(DEST/'P2_selection.json')['selected']
    model = joblib.load(DEST/(name+'.joblib'))
    return name, model['arm'], model['kind']


def transfer():
    r,x,fid=load(); selected,arm,kind=selected_spec()
    result={'selected':selected,'source_folds':[],'formats':{},'seed_confirmation':None}
    with threadpool_limits(limits=4):
        for index,(h,c) in enumerate(FOLDS):
            name=selected if index==0 else 'P4_source_'+str(h)
            if index:
                fit=~r.fold.isin([h,c]).to_numpy(); cc,den,e=weights(r,fid,x.shape[0],fit,arm);e['arm']=arm
                model=solve(x,cc,den,kind,name,e);scores=evaluate(r,x,fid,model,h,c,name)
            else:
                scores=read(DEST/(name+'_scores.json'))
            result['source_folds'].append({'H':h,'C':c,'name':name,'metrics':scores['evaluation']})
        h,c=FOLDS[0]; fit=~r.fold.isin([h,c]).to_numpy()
        for carrier in FORMATS:
            result['formats'][carrier]={}
            for protocol in ['zero_M','whole_format']:
                remove=r.route.eq(carrier).to_numpy()
                if protocol=='zero_M':remove &= r.label_index.eq(1).to_numpy()
                restricted=fit&~remove
                cc,den,e=weights(r,fid,x.shape[0],restricted,arm);e.update({'arm':arm,'removed_supervision_rows':int((fit&remove).sum())})
                name='P4_'+carrier+'_'+protocol
                model=solve(x,cc,den,kind,name,e); scores=evaluate(r,x,fid,model,h,c,name)
                pp=np.load(DEST/(name+'_feature_prediction.npy'))
                hm=r.fold.eq(h).to_numpy() & r.route.eq(carrier).to_numpy()
                cm=r.fold.eq(c).to_numpy() & r.route.eq(carrier).to_numpy()
                entry={'removed_fit_rows':e['removed_supervision_rows'],'H_metrics':score(r,fid,pp,hm), 'C_metrics':score(r,fid,pp,cm),
                       'H_components':int(r.loc[hm,'component'].nunique()),'known_schema':True}
                if protocol=='whole_format':
                    # Disable carrier-specific fact extraction; keep complete universal residual, preserve record port and clear its message-conflict bit.
                    from sklearn.preprocessing import normalize
                    enc=joblib.load(OUT/'facts_encoder.joblib')
                    default=normalize(enc.transform([{}]).astype(np.float32),copy=False)
                    ids=np.unique(fid[hm|cm]); blocks=[]
                    for beg in range(0,len(ids),2048):
                        ix=ids[beg:beg+2048]; xx=x[ix]; meta=xx[:,-18:].tolil();meta[:,17]=0
                        generic=sparse.hstack([xx[:,:65792],sparse.vstack([default]*len(ix),format='csr'),meta.tocsr()],format='csr')
                        blocks.append((generic@model['coef']+model['intercept']).argmax(1))
                    gp=pp.copy();gp[ids]=np.concatenate(blocks)
                    entry['unknown_syntax_no_adapter']={'H_metrics':score(r,fid,gp,hm),'C_metrics':score(r,fid,gp,cm),
                        'scope':'Carrier facts disabled; only universal residual text and record port. No claim of novel arbitrary grammar support.'}
                result['formats'][carrier][protocol]=entry
                emit(stage='carrier_finished',carrier=carrier,protocol=protocol,**entry)
                save(DEST/'P4_transfer_progress.json',result)
        cc,den,e=weights(r,fid,x.shape[0],fit,arm,seed=7902);e.update({'arm':arm,'seed':7902})
        name='P4_second_seed';model=solve(x,cc,den,kind,name,e);scores=evaluate(r,x,fid,model,h,c,name)
        result['seed_confirmation']=scores['evaluation']
    save(DEST/'P4_transfer.json',result)


def shadow():
    r,x,fid=load();selected,arm,kind=selected_spec();h,c=FOLDS[0]
    fit=~r.fold.eq(h).to_numpy();cc,den,e=weights(r,fid,x.shape[0],fit,arm);e['arm']=arm
    with threadpool_limits(limits=4):
        model=solve(x,cc,den,kind,'P5_shadow',e);scores=evaluate(r,x,fid,model,h,c,'P5_shadow')
    previous=read(DEST/(selected+'_scores.json'))['evaluation']
    gate=guards(previous,scores['evaluation'],strict=False)
    p=np.load(DEST/(selected+'_feature_prediction.npy'));q=np.load(DEST/'P5_shadow_feature_prediction.npy')
    take=r.fold.eq(h).to_numpy();y=r.label_index.to_numpy()[take];bp=p[fid[take]];ap=q[fid[take]]
    report={'selected':selected,'before':previous,'after':scores['evaluation'],'gates':gate,
            'fixed':int(((bp!=y)&(ap==y)).sum()),'broken':int(((bp==y)&(ap!=y)).sum()),
            'passed':all(gate.values()) and read(DEST/'P5_shadow_fit.json')['converged'],
            'calibration_in_shadow_fit':True,'H_unchanged':True}
    save(DEST/'P5_shadow.json',report);emit(stage='shadow_finished',**report)


def replay():
    """Prediction first without answers; then one fixed inspected-development regression."""
    import pyarrow.parquet as pq
    from sklearn.preprocessing import normalize
    from run_v75 import adapter, NAMES
    from v75_views import view,byte_matrix
    from v75_corrective import stable
    from v75_metadata import encode
    from v78_denial_adapter_v4 import parse
    check();selected,arm,kind=selected_spec();model=joblib.load(DEST/(selected+'.joblib'))
    v=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');pred=[];routes=[];event=[];activated=0
    start=time.monotonic()
    with threadpool_limits(limits=4):
        for batch in pq.ParquetFile(ROOT/'data/official/valid_input.parquet').iter_batches(batch_size=2048,columns=['event_id','message_sanitized','src_port'],use_threads=False):
            df=batch.to_pandas();messages=[s if isinstance(s,str) else '' for s in df.message_sanitized]
            rr=[v.prepare_record({'message_sanitized':s}) for s in messages];facts=[]
            for s,p in zip(messages,rr):
                patch=parse(s) if p['route']=='unsupported' else None
                facts.append(patch['facts'] if patch else p['facts']);activated+=int(bool(patch))
            fx=normalize(enc.transform(facts).astype(np.float32),copy=False)
            meta,_,_=encode(df.src_port.tolist(),[f.get('src_port_fixed',65536) for f in facts])
            xx=sparse.hstack([byte_matrix([stable(view(s)[0]) for s in messages]),fx,meta],format='csr')
            pred.extend((xx@model['coef']+model['intercept']).argmax(1).tolist());routes.extend([p['route'] for p in rr]);event.extend(df.event_id.tolist())
            if len(pred)%102400==0:emit(stage='frozen_development_inference',rows=len(pred),seconds=round(time.monotonic()-start,1))
    predictions=pd.DataFrame({'event_id':event,'route':routes,'prediction':np.array(pred,np.int8)})
    predictions.to_parquet(DEST/'development_predictions.parquet',index=False)
    save(DEST/'development_inference_receipt.json',{'selected':selected,'answers_read':False,'rows':len(pred),'adapter_activated':activated,'prediction_sha256':sha(DEST/'development_predictions.parquet'),'model_sha256':sha(DEST/(selected+'.joblib'))})
    answers=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet')
    d=predictions.merge(answers,on='event_id',validate='one_to_one',sort=False);y=d.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy();p=d.prediction.to_numpy()
    met=metrics(np.bincount(y*3+p,minlength=9).reshape(3,3));table=[]
    old=ROOT/'artifacts/v78_boundary_20260922'
    prior=pd.read_parquet(old/'frozen_development/predictions.parquet',columns=['event_id','O_sgd_pred'])
    patch=pd.read_parquet(old/'denial_adapter_v4/changed_predictions.parquet');patch=patch[patch.dataset.eq('development')]
    prior.iloc[patch.row_position.to_numpy(),prior.columns.get_loc('O_sgd_pred')]=patch.O_sgd_pred.to_numpy()
    assert np.array_equal(d.event_id,prior.event_id)
    b=prior.O_sgd_pred.to_numpy()
    for route in ['ALL']+sorted(d.route.unique()):
        take=np.ones(len(d),bool) if route=='ALL' else d.route.eq(route).to_numpy();cm=np.bincount(y[take]*3+p[take],minlength=9).reshape(3,3);mm=metrics(cm)
        for k in range(3):
            cell=take&(y==k)
            table.append({'route':route,'label':NAMES[k],'support':int(cm[k].sum()),'correct':int(cm[k,k]),'recall':mm['recall'][k],'precision':mm['precision'][k],
                          'pred_B':int(cm[k,0]),'pred_M':int(cm[k,1]),'pred_S':int(cm[k,2]),'fixed':int((cell&(b!=y)&(p==y)).sum()),'broken':int((cell&(b==y)&(p!=y)).sum())})
    pd.DataFrame(table).to_csv(DEST/'development_classwise.csv',index=False)
    gates={'errors_le4340':met['errors']<=4340,'M_tp_ge11729':met['cm'][1][1]>=11729,'M_precision':met['precision'][1]>=.8666876277314888,
           'S_recall':met['recall'][2]>=.9568110421253123,'S_precision':met['precision'][2]>=.932814044903176,'normal_fp_le34':met['normal_false_alerts']<=34}
    save(DEST/'development_regression.json',{'selected':selected,'metrics':met,'gates':gates,'passed':all(gates.values()),'scope':'Frozen diagnostic classifier; previously inspected target, no selection or tuning on these labels','fixed':int(((b!=y)&(p==y)).sum()),'broken':int(((b==y)&(p!=y)).sum())})
    emit(stage='development_scored',metrics=met,gates=gates)


def final_fit():
    check()
    assert read(DEST/'P5_shadow.json')['passed'], 'Shadow failed; no final fit allowed'
    assert read(DEST/'development_regression.json')['passed'], 'Diagnostic regression failed; no final fit allowed'
    r,x,fid=load();_,arm,kind=selected_spec();fit=np.ones(len(r),bool);cc,den,e=weights(r,fid,x.shape[0],fit,arm);e['arm']=arm
    # User requires all original supervision, including normal. Final objective uses A; if selection picked sampling with changed objective, do not silently switch.
    if arm!='A':
        save(DEST/'final_fit_blocked.json',{'reason':'Selected objective uses sampled benign; all-record final objective equivalence not validated. No silent change to selected target.'})
        return
    with threadpool_limits(limits=4):solve(x,cc,den,kind,'final_all_records',e)



if __name__ == '__main__':
    stages={'register':register,'pack':pack,'primary':primary,'transfer':transfer,'shadow':shadow,'replay':replay,'final':final_fit}
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=list(stages)); args=ap.parse_args()
    stages[args.stage]()
