"""Publish completed diagnostic evidence; latest trained delivery stays V159."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v160_complete_fixed_endpoint_result_publication_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    bound=read(ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002/run_seal.json')['source_sha256'];assert all(p.relative_to(ROOT).as_posix() not in bound for p in mutable)
    for p in mutable:
        t=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    specs=[('v160-results-review','V160三终点实际诊断未通过','docs/V160_COMPLETE_FIXED_ENDPOINT_ACTUAL_RESULTS_20261002.md'),('v160-results-data','V160完整诊断成本与恢复','artifacts/v160_complete_fixed_endpoint_result_records_20261002/actual_result_summary.json')]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*(ROOT/s[2] for s in specs)]}
    summary=read(ROOT/specs[1][2]);assert summary['actual_new_head_feature_calls']==732 and not summary['quality_acceptance'] and summary['new_fits']==summary['permanent_updates']==0
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v159-delivery';historical={e['id']:dict(e) for e in cat['documents']}
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256']
    for ident,title,name in specs:
        src=ROOT/name;t=OUT/'physical_source_snapshots'/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,t)
        assert t.stat().st_size<=256*1024 and ident not in historical
        cat['documents'].insert(0,dict(id=ident,title=title,path=t.relative_to(ROOT).as_posix(),sha256=sha(t),category='review_evidence',keywords=['V160']))
    p=cat['project'];p['authoritative_direction_id']='v160-results-review';p['as_of']='2026-10-02'
    p['current_summary']='V159完成6次真实拟合，342完整类梯度/574proposal/165更新；ASA A276M/4860S、B1902M/1629S，质量失败。V160三终点诊断完成：732头/意见、6类梯度、10间隔梯度、0fit/0永久更新；角色0安全风险下降但分类不变，1/2证书失败，全部恢复保护通过。累计15318头、368类梯度、10间隔梯度。三目标active，下一仅保存向量数值残差修正。'
    p['current_direction']=['V160三角色机制未过，不新增fit；最新完整训练交付仍V159失败。','下一只审查保存向量的重构残差；新调用须另行资格与封存。','全部参数恢复、联合TRAIN保护通过；训练掌握及全目标active。']
    for e in cat['documents']:
        if e['id'] in historical:assert e==historical[e['id']]
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    lead='最新诊断：[V160三固定终点机制未通过](docs/V160_COMPLETE_FIXED_ENDPOINT_ACTUAL_RESULTS_20261002.md)。732头/意见、6类梯度、10间隔梯度，0fit/0永久更新；角色0风险下降但分类不变，1/2证书失败；全部恢复并通过联合保护。最新完整训练模型仍V159质量失败；三目标active，下一只审查保存向量残差。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V160实际有限诊断完成，机制未通过\n\n'+lead+'MCP delivery=v159-delivery、direction=v160-results-review；累计15318头/368类梯度/10间隔梯度。本轮资格和封存时0调用是历史阶段快照；实际三角色诊断已完成，不复用剩余预算追加fit。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v159-results-review'","'v160-results-review'").replace('"v159-results-review"','"v160-results-review"')
    extra='''    def test_v160_completed_diagnostic_does_not_promote_failed_mechanism(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        data=json.loads(readonly._document_text(entries['v160-results-data']))
        self.assertEqual(data['actual_new_head_feature_calls'],732)
        self.assertEqual(data['actual_new_class_gradients'],6)
        self.assertEqual(data['actual_new_margin_gradients'],10)
        self.assertEqual(data['new_fits'],0)
        self.assertEqual(data['permanent_updates'],0)
        self.assertTrue(data['all_parameters_restored'])
        self.assertEqual(data['latest_completed_trained_evaluated_model'],'V159')
        self.assertFalse(data['quality_acceptance'])
        self.assertEqual(data['root_goal_status'],'active')

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,extra+needle),encoding='utf-8');check_bindings(bindings)
    (OUT/'publication.json').write_text(json.dumps(dict(status='V160_completed_failed_mechanism_diagnostic_published_latest_trained_V159',catalog_bytes=len(raw),historical_documents_preserved=len(historical),authoritative_delivery='v159-delivery',authoritative_direction='v160-results-review',new_publication_model_calls=0,new_publication_gradients=0,new_publication_fits=0,new_publication_updates=0,quality_acceptance=False,goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='V160_actual_diagnostic_published_no_model_promoted',catalog_bytes=len(raw))))

if __name__=='__main__':main()
