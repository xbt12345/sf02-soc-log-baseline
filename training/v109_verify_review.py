"""Independently recount frozen member results using NumPy; no fitting."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, save, sha
from v109_frozen_member_audit import DEST, OLD


def main():
    j=json.loads((DEST/'frozen_member_audit.json').read_text(encoding='utf-8'))
    d=pd.read_parquet(OLD/'support_and_error_ledger.parquet')
    checks={'rows_112807':len(d)==112807,'zero_new_fits':j['classifier_fits']==j['calibration_fits']==0,
      'ledger_hash':sha(OLD/'support_and_error_ledger.parquet')==j['ledger_sha256'],
      'source_hash':sha(ROOT/'training/v109_frozen_member_audit.py')==j['source_sha256'],
      'input_hashes':all(sha(ROOT/p)==h for p,h in j['input_sha256'].items())}
    totals={v:{c:{} for c in ('M','S')} for v in ('N1','N2')}
    for r in j['results']:
        v,k=r['view'],r['fold'];z=np.load(DEST/f'{v}_fold{k}_member_logits.npy').astype('f8')
        pp=np.exp(z-z.max(2,keepdims=True));pp/=pp.sum(2,keepdims=True)
        mask=d.fold.to_numpy()==k;loc=d.local.to_numpy()[mask];y=d.truth.to_numpy()[mask]
        pred=pp.mean(1).argmax(1)[loc];members=z.argmax(2)[loc];ml=z.mean(1).argmax(1)[loc]
        checks[f'{v}_fold{k}_original_decisions']=np.array_equal(pred,d.loc[mask,f'{v}_TabM25'].to_numpy())
        checks[f'{v}_fold{k}_gradient_autograd']=r['analytic_gradient_autograd_check']
        for c,name in [(1,'M'),(2,'S')]:
            take=y==c;bad=take&(pred!=c);allbad=(members!=c).all(1)
            counts={'rows':int(take.sum()),'mean_probability_errors':int(bad.sum()),
              'errors_all_16_members_wrong':int((bad&allbad).sum()),'errors_some_member_correct':int((bad&~allbad).sum()),
              'mean_logit_errors_diagnostic_only':int((take&(ml!=c)).sum()),
              'mean_logit_repairs':int((bad&(ml==c)).sum()),'mean_logit_regressions':int((take&(pred==c)&(ml!=c)).sum())}
            checks[f'{v}_fold{k}_{name}_independent_counts']=all(a==r['counts']['heldout'][name][key] for key,a in counts.items())
            for key,a in counts.items():totals[v][name][key]=totals[v][name].get(key,0)+a
    checks['historical_oof_error_totals']=(totals['N1']['M']['mean_probability_errors']==318 and totals['N1']['S']['mean_probability_errors']==2094
       and totals['N2']['M']['mean_probability_errors']==296 and totals['N2']['S']['mean_probability_errors']==5194)
    result={'all_checks_passed':all(checks.values()),'checks':checks,'heldout_totals':totals,
      'scope':'Saved-logit NumPy recount, original OOF decisions and derivative checks; no fit, no new blind test, no model quality acceptance.',
      'artifact_sha256':{p.name:sha(p) for p in [DEST/'frozen_member_audit.json',*sorted(DEST.glob('*_member_logits.npy'))]},
      'source_sha256':sha(__file__)}
    save(DEST/'verification.json',result)
    assert result['all_checks_passed'],checks
    print(json.dumps({'all_checks_passed':True,'checks':len(checks),'heldout_totals':totals},ensure_ascii=False))


if __name__=='__main__':main()
