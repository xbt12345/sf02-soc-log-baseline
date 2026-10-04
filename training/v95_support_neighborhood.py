"""Frozen R0 neighborhood of the dominant V suspicious failure component."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v89_common import LAST, DEST as V89, data
from v75_corrective import stable
from v95_prepare import DEST


def main():
    assert not (DEST / 'dominant_component_neighborhood.json').exists()
    r, y, fid, _, _, _, _ = data()
    roles = pd.read_parquet(DEST / 'full_format_roles.parquet', columns=['role']).role.to_numpy()
    target = r[(r.component == 27221) & (r.label_index == 2) & (roles == 'V')]
    assert len(target) == 685
    qids, qcount = np.unique(fid[target.index], return_counts=True)
    assert sorted(qcount.tolist()) == [4, 681]
    x = load_sparse(LAST / 'X')
    texts = pd.read_parquet(OUT / 'text_dictionary.parquet').set_index('text_id').text
    obs = np.load(V89 / 'row_fact_code.npy')
    facts = pd.read_parquet(V89 / 'row_fact_dictionary.parquet').observation_json
    rows = []
    for q, n in zip(qids, qcount):
        rep = int(target.index[fid[target.index] == q][0])
        head = {'R0_id': int(q), 'V_S_rows': int(n), 'normalized_text': stable(texts.loc[r.new_text_id.iloc[rep]]),
                'parsed_known_facts': json.loads(facts.iloc[obs[rep]])['facts'], 'neighbors': {}}
        for cl in [1, 2]:
            pool = np.flatnonzero(np.isin(roles, ['A', 'B']) & r.route.eq('asa').to_numpy() & (y == cl))
            unique, index = np.unique(fid[pool], return_index=True)
            sim = np.asarray(x[unique] @ x[int(q)].T.toarray()).ravel()
            nearest = np.argsort(sim)[-3:][::-1]
            head['neighbors'][str(cl)] = []
            for j in nearest:
                row = int(pool[index[j]])
                head['neighbors'][str(cl)].append({'R0_id': int(unique[j]), 'cosine_like_block_dot': float(sim[j]),
                    'component': int(r.component.iloc[row]), 'normalized_text': stable(texts.loc[r.new_text_id.iloc[row]])})
        rows.append(head)
    result = {'status': 'frozen_input_support_diagnosis', 'target_component': 27221,
              'V_suspicious_rows': 685, 'distinct_R0': 2, 'records': rows,
              'interpretation_limit': 'Sparse block dot is a similarity diagnostic, not a semantic attack label or calibrated probability. R0 byte coordinates retain ICMP type/code characters; parsed known facts are empty.',
              'new_classifier_fits': 0, 'source_sha256': sha(__file__)}
    save(DEST / 'dominant_component_neighborhood.json', result)
    print(json.dumps({'V_S_rows': 685, 'distinct_R0': 2,
                      'parsed_known_facts_empty': all(not x['parsed_known_facts'] for x in rows),
                      'top_M_similarity': [round(z['neighbors']['1'][0]['cosine_like_block_dot'], 3) for z in rows],
                      'top_S_similarity': [round(z['neighbors']['2'][0]['cosine_like_block_dot'], 3) for z in rows]}), flush=True)


if __name__ == '__main__':
    main()
