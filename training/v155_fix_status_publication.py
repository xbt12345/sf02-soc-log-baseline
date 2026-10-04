"""Adapt verified receipt to existing MCP schema; never change training evidence."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v138_runtime import save
OUT=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'

def main():
    assert not (OUT/'publication_status_repair.json').exists()
    pub=read(OUT/'publication.json');check_bindings(pub['source_sha256'])
    source=OUT/'final_delivery.json';verification=OUT/'verification.json'
    d=read(source);v=read(verification);assert d['verification_sha256']==sha(verification)
    snapshot=OUT/'first_results_publication_snapshot';snapshot.mkdir()
    for rel in pub['source_sha256']:
        target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    save(snapshot/'manifest.json',pub)
    adapter=OUT/'mcp_delivery.json'
    save(adapter,dict(d,validation_scope=v['validation_scope']+' '+v['full_population_scope'],
        adapter_scope='Read-only schema join of actual sealed evaluation receipts; no classifier changes.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [source,verification]}))
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp)
    entry=next(x for x in cat['documents'] if x['id']=='v155-delivery');entry['path']=adapter.relative_to(ROOT).as_posix();entry['sha256']=sha(adapter)
    cat['documents'].insert(1,dict(id='v155-training-receipt',title='V155未经适配的真实训练交付凭据',path=source.relative_to(ROOT).as_posix(),
        category='delivery_evidence',summary='六拟合与完整验收原始凭据保持不变',sha256=sha(source),keywords=['V155','实际','交付','原始凭据']))
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;cp.write_bytes(raw)
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8')
    s=s.replace("self.assertIn('v146-delivery', status.source_ids)","self.assertIn('v155-delivery', status.source_ids)")
    s=s.replace("self.assertIn('无模型晋升', status.current_summary)","self.assertIn('未晋升', status.current_summary)")
    s=s.replace("self.assertIn('6次既有第二层拟合', status.current_summary)","self.assertIn('V155六拟合完成', status.current_summary)")
    s=s.replace("self.assertIn('1200次完整梯度', status.current_summary)","self.assertIn('1800梯度', status.current_summary)")
    s=s.replace("self.assertIn('2408次有限试探', status.current_summary)","self.assertIn('1806proposal', status.current_summary)")
    s=s.replace("self.assertIn('1200次接受更新', status.current_summary)","self.assertIn('1200更新', status.current_summary)")
    s=s.replace("self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v146-delivery')","self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v155-delivery')")
    needle="        self.assertEqual(design['latest_actual_training'], 'V138')"
    s=s.replace(needle,needle+"\n        actual = json.loads(readonly._document_text(readonly._documents_by_id(catalog)['v155-training-receipt']))\n        self.assertEqual(actual['classifier_fits'], 6)\n        self.assertEqual(actual['latest_actual'], 'V155')\n        self.assertFalse(actual['quality_acceptance'])")
    tp.write_text(s,encoding='utf-8')
    check_bindings(read(adapter)['source_sha256'])
    save(OUT/'publication_status_repair.json',dict(status='actual_receipt_scope_schema_join_and_current_authority_test_updated',
        original_training_receipt_sha256_unchanged=sha(source),first_publication_failure='mcp_publication_attempt1.txt',
        training_sources_or_seal_changed=False,new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),adapter,source,verification,cp,tp,OUT/'mcp_publication_attempt1.txt']}))
    print(json.dumps(dict(status='publication_schema_repaired_no_training_evidence_changes',catalog_bytes=len(raw)),ensure_ascii=False))

if __name__=='__main__':main()
