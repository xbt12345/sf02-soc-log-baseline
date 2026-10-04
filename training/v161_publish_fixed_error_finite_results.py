"""Publish actual finite target qualification, preserve all historic metadata."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v161_results_20261002'
REPORT=ROOT/'docs/V161_COMPLETE_FIXED_ERROR_FINITE_RESULTS_20261002.md'

def main():
    assert not OUT.exists();OUT.mkdir()
    audits=[ROOT/'artifacts/v161_saved_fixed_error_result_audit_20261002/audit.json',ROOT/'artifacts/v161_cached_tail_actual_result_audit_20261002/audit.json']
    a,b=[read(p) for p in audits]
    for audit in [a,b]:check_bindings(audit['source_sha256'])
    assert a['actual_new_counts']['head_attempts']==1114 and a['actual_new_counts']['gradient_attempts']==12 and b['finite_error_target_pass'] and b['counts']['head_attempts']==66
    assert all(r['finite_error_target_pass'] for r in a['reports'][:2])
    summary=dict(status='three_fixed_error_target_endpoints_actual_finite_passed_training_mastery_not_passed',actual_new_heads=1180,actual_new_features=1180,actual_fixed_error_target_gradients=12,actual_new_full_class_gradients=0,actual_new_margin_gradients=0,actual_finite_proposals=47,actual_QP_solves=3,new_fits=0,permanent_updates=0,cumulative_heads=16908,cumulative_features=16908,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=12,cumulative_margin_gradients=10,all_parameters_restored=True,all_joint_TRAIN_retention_passed=True,all_safe_candidate_classification_changes=0,latest_complete_trained_model='V159',latest_complete_trained_quality='failed',root_goal_status='active',next='separately bounded actual fixed-error-target trajectory with cumulative pure-correct repair protection',quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,*audits]})
    def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(OUT/'actual_result_summary.json',summary)
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for sealpath in [ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/run_seal.json',ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002/run_seal.json']:
        bound=read(sealpath)['source_sha256'];assert not any(p.relative_to(ROOT).as_posix() in bound for p in mutable)
    for p in mutable:
        target=OUT/'previous'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    cat=read(mutable[2]);historic={r['id']:dict(r) for r in cat['documents']};assert cat['project']['authoritative_delivery_id']=='v159-delivery'
    for r in cat['documents']:assert sha(ROOT/r['path'])==r['sha256']
    snapshot=OUT/'review.md';shutil.copyfile(REPORT,snapshot)
    cat['documents'].insert(0,dict(id='v161-review',title='V161实际有限资格',path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(snapshot),category='review_evidence',keywords=['V161']))
    project=cat['project'];project['authoritative_direction_id']='v161-review'
    project['current_summary']='V159六fit完整验收失败。V161三端点旧错目标有限通过，0新fit/更新；累计16908头/368原类/12旧错/10间隔梯度。下一有界轨迹。'
    project['current_direction']=['固定旧错目标有界轨迹，累计保护新修复。']
    # Remove only redundant development/independent-evaluation wording;
    # version-specific facts remain in the consolidated first statement.
    duplicate='已查看开发折，不是独立盲测或外部部署。';v158='V158仍是此前已查看的开发来源，未取得独立外部环境验收。'
    assert duplicate in project['known_limits'] and v158 in project['known_limits']
    project['known_limits']=[r for r in project['known_limits'] if r not in [duplicate,v158]]
    project['known_limits'][0]=project['known_limits'][0].replace('已看开发折、N无新监督，不能申报盲验或N新学习。','V158及当前均为已看开发来源，非盲验或外部部署；N无新监督。')
    project['known_limits'][-1]='V159质量失败；V161有限资格不证明分类掌握，尚无新拟合。'
    for r in cat['documents']:
        if r['id'] in historic:assert r==historic[r['id']]
    data=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(data)<=256*1024
    mutable[2].write_bytes(data)
    lead='最新实际机制：[V161三终点固定旧错目标有限资格](docs/V161_COMPLETE_FIXED_ERROR_FINITE_RESULTS_20261002.md)。0新fit/永久更新、0安全候选翻类；累计16908头/368原类/12错误目标/10间隔梯度。下一单独封存有界轨迹；最新完整训练仍V159质量失败。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V161固定错误目标有限资格完成\n\n'+lead+'MCP delivery=v159-delivery，direction=v161-review。原20级域失败与新40级尾段的来源、记录及成本保持分开。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v160-polished-review'","'v161-review'").replace('"v160-polished-review"','"v161-review"');mutable[3].write_text(tests,encoding='utf-8')
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(data),historic_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v161-review',new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}))
    print(json.dumps(dict(status='V161_actual_fixed_error_finite_results_published',catalog_bytes=len(data))))

if __name__=='__main__':main()
