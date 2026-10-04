"""Single message-only three-class inference entry for v3.8 candidates."""
import numpy as np
import v38_representation as rep
import v38_learning as learn

LABELS=np.asarray(['benign','malicious','suspicious'])

def classify_records(bundle,records):
    if bundle.get('version')!=rep.VERSION or bundle.get('view') not in rep.VIEWS:
        raise ValueError('Unsupported classifier representation or view')
    if bundle.get('decision')!='three_class_argmax':raise ValueError('Unexpected decision contract')
    if not records:return {'pred_label':np.asarray([],dtype=str),'probabilities':np.zeros((0,3)), 'quality_accepted':False}
    projected=[rep.view_record(rep.prepare_record(row),bundle['view']) for row in records]
    probabilities,pred=learn.classify(bundle,[p['text'] for p in projected],[p['facts'] for p in projected])
    if not np.isfinite(probabilities).all() or probabilities.shape!=(len(records),3):
        raise ValueError('Invalid probabilities')
    return {'pred_label':LABELS[pred],'probabilities':probabilities,'quality_accepted':False}
