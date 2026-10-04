"""Promote actual completed delivery metadata, never promote failed model."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v159_complete_actual_result_publication_20261002'
TRIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for seal in [TRIAL/'run_seal.json',ROOT/'artifacts/v159_evaluation_source_sum_repair_20261002/run_seal.json',ROOT/'artifacts/v159_evaluation_cached_continuation_20261002/run_seal.json']:
        binding=read(seal)['source_sha256'];assert all(p.relative_to(ROOT).as_posix() not in binding for p in mutable)
    for p in mutable:
        dest=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    specs=[('v159-delivery','V159六fit完整失败交付','artifacts/v159_class_boundary_numeric_trial_20261002/final_delivery.json'),('v159-results-review','V159完整实际结果与下一问题','docs/V159_COMPLETE_ACTUAL_RESULTS_AND_NEXT_ISSUE.md'),('v159-results-data','V159完整错误与费用重算','artifacts/v159_complete_result_records_20261002/actual_result_summary.json'),('v159-quality','V159完整三分类原行质量','artifacts/v159_class_boundary_numeric_trial_20261002/quality.json'),('v159-independent-results','V159独立全gold及费用核验','artifacts/v159_independent_boundary_numeric_result_audit_20261002/audit.json'),('v159-next-mechanism','V159独立下一保护方向机制','docs/V159_INDEPENDENT_FINAL_REVIEW_AND_NEXT_MECHANISM_20261002.md')]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+[ROOT/s[2] for s in specs]};(OUT/'pre_publication_bindings.json').write_text(json.dumps(dict(source_sha256=bound),indent=2)+'\n',encoding='utf-8')
    d=read(TRIAL/'final_delivery.json');assert d['classifier_fits']==6 and not d['quality_acceptance'] and d['cumulative_actual_head_calls']==14586 and d['cumulative_actual_full_class_gradients']==362 and not d['model_promoted']
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery'
    for entry in cat['documents']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['id']
    published=[]
    for ident,title,name in specs:
        src=ROOT/name;dest=OUT/'physical_source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest) and dest.stat().st_size<=256*1024 and all(e['id']!=ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),category='actual_delivery' if ident=='v159-delivery' else 'review_evidence',keywords=['V159']))
        published.append(dict(id=ident,path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest)))
    p=cat['project'];p['authoritative_delivery_id']='v159-delivery';p['authoritative_direction_id']='v159-results-review';p['as_of']='2026-10-02'
    p['current_summary']='V159完成6次真实拟合及2056871完整原行三分类验收，训练342完整类梯度/574proposal/165更新；ASA A276M/4860S，B1902M/1629S，质量失败、未晋升。B OOF7030错/纯输入6824错，仅修复38训练行和2外折M，联合225558旧能力通过。全部失败/诊断/预检/750评价累计14586头意见及362完整类梯度；前3真实重放FIT缓存边界单列，后3完整q/logq保存。下一唯一主动保护约束方向资格，当前不新增官方调用，三个问题及全目标active。'
    p['current_direction']=['最新完整实际交付V159失败；本配置不补训练、不换端点、不用未花完预算追加fit。','下一项先资格主动保护约束共同下降方向，保留同一模型/人口/原频/精确argmax保护；初始化候选另因素，不同轮混改。','成本全部累计14586/362；第一训练问题未过，支持、迁移及完整三分类泛化目标active，已看开发折不冒充盲测。']
    p['known_limits'].insert(0,'V159前3实际评价未保存全q，仅FIT重放行/source及全q精确argmax断言控制流证据；冻结完整q来源单列。已看开发折、N无新监督，不能申报盲验或N新学习。')
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    lead='最新完整实际结果：[V159六拟合与全原行质量失败](docs/V159_COMPLETE_ACTUAL_RESULTS_AND_NEXT_ISSUE.md)。ASA A276M/4860S、B1902M/1629S；B OOF7030错，第一训练问题未过。累计14586头/意见、362完整类梯度、6fit/165更新，全部失败成本保留、模型未晋升。下一唯一保护约束方向先做有界资格，三个问题及完整目标active。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159最新完整实际交付与未关闭问题\n\n'+lead+'父独立全gold/费用审查已实测exit0，见docs/V159_INDEPENDENT_FINAL_REVIEW_AND_NEXT_MECHANISM_20261002.md。前3完整actual q未保存边界及缓存控制流证据如实单列。MCP delivery=v159-delivery、direction=v159-results-review；历史V158及V159预检0fit快照保留。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v159-numeric-preflight-complete'","'v159-results-review'").replace('"v159-numeric-preflight-complete"','"v159-results-review"')
    tests=tests.replace("self.assertIn('v158-delivery', status.source_ids)","self.assertIn('v159-delivery', status.source_ids)").replace("self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v158-delivery')","self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v159-delivery')")
    tests=tests.replace("self.assertIn('V158完成60次真实拟合', status.current_summary)","self.assertIn('V159完成6次真实拟合', status.current_summary)").replace("self.assertIn('1200完整梯度', status.current_summary)","self.assertIn('342完整类梯度', status.current_summary)").replace("self.assertIn('1200proposal', status.current_summary)","self.assertIn('574proposal', status.current_summary)").replace("self.assertIn('1200更新', status.current_summary)","self.assertIn('165更新', status.current_summary)")
    extra='''    def test_v159_complete_delivery_failed_training_and_full_quality(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        delivery=json.loads(readonly._document_text(entries['v159-delivery']))
        data=json.loads(readonly._document_text(entries['v159-results-data']))
        self.assertEqual(delivery['classifier_fits'],6)
        self.assertEqual(delivery['ASA_errors']['B'],dict(M=1902,S=1629))
        self.assertEqual(delivery['cumulative_actual_head_calls'],14586)
        self.assertEqual(delivery['cumulative_actual_full_class_gradients'],362)
        self.assertEqual(data['arms']['B']['OOF_errors'],7030)
        self.assertEqual(data['arms']['B']['OOF_pure_current_input_errors'],6824)
        self.assertFalse(delivery['first_issue_training_qualification'])
        self.assertFalse(delivery['quality_acceptance'])
        self.assertFalse(delivery['model_promoted'])
        self.assertEqual(delivery['cached_v5_actual_evaluation_heads'],360)
        self.assertEqual(delivery['new_v7_actual_evaluation_heads'],280)

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,extra+needle),encoding='utf-8');check_bindings(bound)
    (OUT/'publication.json').write_text(json.dumps(dict(status='completed_actual_V159_failed_delivery_published_no_model_promoted',catalog_bytes=len(raw),documents=published,authoritative_delivery='v159-delivery',authoritative_direction='v159-results-review',actual_cumulative_heads=14586,actual_cumulative_gradients=362,new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,goal_status='active',quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='V159_complete_actual_failed_delivery_published',catalog_bytes=len(raw),latest_actual='V159')))
if __name__=='__main__':main()
