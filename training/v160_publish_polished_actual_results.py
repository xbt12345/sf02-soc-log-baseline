"""Publish numerical finite success plus observed classification misalignment."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v160_polished_publication_20261002'
REPORT=ROOT/'docs/V160_POLISHED_FINITE_ACTUAL_RESULTS_AND_CLASSIFICATION_OBJECTIVE_20261002.md'

def main():
    assert not OUT.exists();OUT.mkdir()
    own=ROOT/'artifacts/v160_cached_polished_actual_result_audit_20261002/audit.json';a=read(own);check_bindings(a['source_sha256']);assert a['actual_new_counts']['head_attempts']==410 and a['actual_new_counts']['gradient_attempts']==0
    independent=ROOT/'artifacts/v160_independent_cached_polished_probe_review_20261002/audit.json';assert independent.exists()
    summary=dict(status='three_fixed_endpoints_numeric_finite_passed_classification_learning_not_passed',actual_cached_probe_heads=410,actual_cached_probe_features=410,actual_cached_proposals=19,new_cached_probe_gradients=0,new_fits=0,permanent_updates=0,cumulative_V159_V160_heads=15728,cumulative_full_class_gradients=368,cumulative_margin_gradients=10,all_parameters_restored=True,all_joint_TRAIN_retention_passed=True,all_safe_probe_classification_changes=0,old_role2_error_CE_masked_by_old_correct_CE=True,latest_completed_trained_evaluated_model='V159',latest_trained_quality='failed',root_goal_status='active',next_single_factor='fixed pure old M/S error original-frequency CE target and exact correct-row guards; full CE reported, not a necessary classification-improvement hard constraint',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,own,independent,ROOT/'artifacts/v160_independent_safe_direction_objective_alignment_20261002/review.json',ROOT/'artifacts/v161_synthetic_accuracy_vs_mean_CE_review_20261002/review.json']})
    (OUT/'actual_result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for seal_path in [ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002/run_seal.json',ROOT/'artifacts/v160_cached_polished_direction_finite_probe_20261002/run_seal.json']:
        bound=read(seal_path)['source_sha256'];assert all(p.relative_to(ROOT).as_posix() not in bound for p in mutable)
    for p in mutable:
        t=OUT/'previous'/p.relative_to(ROOT);t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    cat=read(mutable[2]);historic={e['id']:dict(e) for e in cat['documents']};assert cat['project']['authoritative_delivery_id']=='v159-delivery'
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256']
    doc=OUT/'sources/results.md';doc.parent.mkdir();shutil.copyfile(REPORT,doc)
    assert doc.stat().st_size<=256*1024
    cat['documents'].insert(0,dict(id='v160-polished-review',title='V160实际数值通过与旧错退化',path=doc.relative_to(ROOT).as_posix(),sha256=sha(doc),category='review_evidence',keywords=['V160']))
    p=cat['project'];p['authoritative_direction_id']='v160-polished-review'
    p['current_summary']='V159完成6次真实拟合，342完整类梯度/574proposal/165更新，质量失败、未晋升。V160数值有限通过但分类目标错位；0新fit/更新；累计15728头/368类/10间隔梯度。三目标active，下一pure旧错M/S目标诊断。'
    p['current_direction']=['下一固定pure旧错M/S原频目标与逐行正确保护，先封存诊断，不直接fit。']
    # Consolidate duplicate V150 limitations, preserving each unique fact.
    old1='解码参数拟合属于实际拟合；冻结基座已见内层TRAIN标签，不能作为独立泛化证据。'
    old2='V150基座已见内层留出标签；受限线性解码不等于独立泛化、威胁判据或信息不可恢复。'
    old3='V150六事实解码拟合保存4053504系数，但分类器参数均未更新。'
    assert all(v in p['known_limits'] for v in [old1,old2,old3]);index=p['known_limits'].index(old1)
    p['known_limits']=[v for v in p['known_limits'] if v not in [old1,old2,old3]]
    p['known_limits'].insert(index,'V150解码是实际拟合，4053504系数但未更新分类器；基座见内层TRAIN/留出标签，受限线性不证明独立泛化、威胁判据或信息不可恢复。')
    p['known_limits'][-1]='V159质量失败；V160数值有限通过不证明分类掌握，旧错目标尚未训练。'
    for e in cat['documents']:
        if e['id'] in historic:assert e==historic[e['id']]
    data=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(data)<=256*1024;mutable[2].write_bytes(data)
    lead='最新实际诊断：[V160数值有限通过与分类目标错位](docs/V160_POLISHED_FINITE_ACTUAL_RESULTS_AND_CLASSIFICATION_OBJECTIVE_20261002.md)。三个固定终点有限通过，0新fit/永久更新；角色2旧错误退化被整体CE下降掩盖。累计15728头/368类/10间隔梯度；最新完整训练仍V159质量失败。下一固定pure旧错目标及逐行正确保护，先新预算诊断。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V160数值修复完成，V161分类目标对齐待诊断\n\n'+lead+'当前MCP delivery=v159-delivery，direction=v160-polished-review。旧诊断各阶段失败、费用和公开快照保持；完整CE仍分解记录，不成为正确分类修复的必要硬门槛。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v160-results-review'","'v160-polished-review'").replace('"v160-results-review"','"v160-polished-review"');mutable[3].write_text(tests,encoding='utf-8')
    (OUT/'publication.json').write_text(json.dumps(dict(status=summary['status'],catalog_bytes=len(data),historical_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v160-polished-review',new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='polished_finite_and_objective_misalignment_actual_results_published',catalog_bytes=len(data))))

if __name__=='__main__':main()
