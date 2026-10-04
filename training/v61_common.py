"""Common data contract for v6.1; no raw identity or label in model inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = ['action','outcome','transport_protocol','src_role','dst_role','src_port_fixed',
          'dst_port_fixed','src_port_range','dst_port_range','icmp_type','icmp_code',
          'icmp_message','icmp_unreachable']
MISSING = '__MISSING__'
MODEL_ID = 'cisco-ai/SecureBERT2.0-base'
REVISION = '7f7c16d1b2316c5046759667ed97f527aa1b7709'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def metrics(y, probabilities):
    from sklearn.metrics import average_precision_score, log_loss
    y = np.asarray(y, dtype=int); p = np.asarray(probabilities, dtype=float)
    pred = p.argmax(1); cm = np.zeros((3,3), dtype=int); np.add.at(cm, (y,pred), 1)
    support = cm.sum(1); denominator = support + cm.sum(0)
    f1 = np.divide(2*cm.diagonal(), denominator, out=np.zeros(3), where=denominator>0)
    recall = [float(cm[i,i]/support[i]) if support[i] else None for i in range(3)]
    return {'rows': len(y), 'confusion_B_M_S':cm.tolist(), 'errors':int((pred!=y).sum()),
            'recall_B_M_S':recall, 'macro_f1_M_S':float(f1[1:].mean()),
            'macro_f1_B_M_S':float(f1.mean()), 'threat_to_benign':int(cm[1:,0].sum()),
            'normal_false_positive_rate':float(1-recall[0]) if support[0] else None,
            'S_average_precision':float(average_precision_score(y==2,p[:,2])) if 0<(y==2).sum()<len(y) else None,
            'cross_entropy':float(log_loss(y,p,labels=[0,1,2]))}


def effects(base, candidate):
    dr = [None if a is None or b is None else b-a for a,b in zip(base['recall_B_M_S'],candidate['recall_B_M_S'])]
    df = candidate['macro_f1_M_S'] - base['macro_f1_M_S']
    return {'delta_macro_f1_M_S':df, 'delta_recall_B_M_S':dr,
            'inner_continuation_effect_size_met':bool(df>=.02 and dr[2] is not None and dr[2]>=.05 and dr[1]>=-.01),
            'not_final_quality_acceptance':True}


def fit_vocab(facts):
    return [[MISSING, '__UNKNOWN__'] + sorted(set(facts[:,j])-{MISSING,'__UNKNOWN__'}) for j in range(facts.shape[1])]


def encode_facts(facts, vocab):
    out = np.empty(facts.shape, dtype=np.int64)
    for j, values in enumerate(vocab):
        mapping = {k:i for i,k in enumerate(values)}
        out[:,j] = [mapping.get(str(v),1) for v in facts[:,j]]
    return out
