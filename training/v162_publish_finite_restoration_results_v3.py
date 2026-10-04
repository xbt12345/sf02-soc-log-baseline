"""Publish mixed actual finite results without changing historic records."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v162_results_20261002'
REPORT=ROOT/'docs/V162_COMPLETE_FINITE_RESTORATION_RESULTS_20261002.md'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists()
    audits=[ROOT/'artifacts/v162_saved_finite_restoration_actual_audit_20261002/audit.json',ROOT/'artifacts/v162_independent_actual_restoration_review_20261002/review.json',ROOT/'artifacts/v162_independent_role0_local_constraint_comparison_20261002/comparison.json']
    a,b,c=[read(p) for p in audits]
    check_bindings(a['source_sha256']);independent_bindings=audits[1].parent/'pre_review_bindings.json';check_bindings(read(independent_bindings)['source_sha256'])
    assert a['actual_new_counts']['head_attempts']==338 and a['actual_new_counts']['margin_attempts']==32 and not a['all_three_actual_finite_restoration_pass'] and not b['all_three_finite_qualified']
    assert [r['actual_finite_restoration_pass'] for r in a['reports']]==[False,True,False]
    assert all(r['parameter_hash_restored'] and r['original_joint_retention_passed'] for r in a['reports'])
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    original_cat=read(mutable[2]);cat=read(mutable[2]);historic={r['id']:dict(r) for r in cat['documents']};assert cat['project']['authoritative_delivery_id']=='v159-delivery'
    for r in cat['documents']:assert sha(ROOT/r['path'])==r['sha256']
    snapshot=OUT/'review.md'
    cat['documents'].insert(0,dict(id='v162-review',title='V162有限恢复结果',path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(REPORT),category='review_evidence',keywords=['V162']))
    project=cat['project'];project['authoritative_direction_id']='v162-review'
    project['current_summary']='V159六fit完整验收失败。V162角色1有限修复8条，0/2失败；0新fit/更新。累计17246头/368原类/12旧错/42间隔梯度。'
    project['current_direction']=['诊断等式恢复和四次预算末端未恢复，暂不训练。']
    replacements={
      'V159前3实际评价未保存全q，仅FIT重放行/source及全q精确argmax断言控制流证据；冻结完整q来源单列。V158及当前均为已看开发来源，非盲验或外部部署；N无新监督。':'V159前3评价未存全q：仅FIT重放行/source及全q精确argmax断言控制流；冻结完整q来源另列。V158/当前已看开发来源，非盲验/外部部署；N无新监督。',
      '原 578 困难 S 对登记辅助配对直接覆盖 0；本轮不产生新的同类标签。':'原578困难S登记辅助配对直接覆盖0；本轮无新同类标签。',
      '原末层 12 拟合、V142 三拟合和 V146 六拟合均不自动追加。':'原末层12、V142三、V146六拟合均不自动追加。',
      'V150解码是实际拟合，4053504系数但未更新分类器；基座见内层TRAIN/留出标签，受限线性不证明独立泛化、威胁判据或信息不可恢复。':'V150实际解码拟合4053504系数，未更新分类器；基座见内层TRAIN/留出标签，受限线性不证明独立泛化、威胁判据或信息不可恢复。',
      'V155已实际完成6拟合/1800梯度；9零步诊断梯度另计。完整质量失败，没有新独立同类支持或新环境泛化确认。':'V155实际6拟合/1800梯度，9零步梯度另计；完整质量失败，无新独立同类支持/新环境泛化确认。',
      'V159质量失败；V161有限资格不证明分类掌握，尚无新拟合。':'V159质量失败；V161/162有限资格不证明分类掌握，无新拟合。'
      ,'正常/其他格式冻结 107 错，无模型晋升；项目预登记 A0 质量门槛保持。':'正常/其他格式冻结107错，无晋升；预登记A0质量门槛保持。'
      ,'V152两次合法TRAIN角色不是新增独立样本，historical outer正确标记不能认证对应TRAIN角色。':'V152两次合法TRAIN非新增独立样本；historical outer正确标记不认证对应TRAIN角色。'
      ,'V151首版key名称过宽，另版复核当前路径；50->116含包装规范变化，不能全部归因日期。':'V151首版key过宽，另版复核当前路径；50->116含包装规范变化，非全因日期。'
      ,'V157只复现固定V146 A完整函数；不证明跨来源因果、不增加标签或模型收益。CPU初次ICMP空切片已另版纠正，原结果保存。':'V157仅复现固定V146 A完整函数；不证明跨来源因果、不增标签/模型收益。初次CPU ICMP空切片另版纠正，原结果保存。'
    }
    assert all(key in project['known_limits'] for key in replacements)
    project['known_limits']=[replacements.get(r,r) for r in project['known_limits']]
    for r in cat['documents']:
        if r['id'] in historic:assert r==historic[r['id']]
    data=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(data)<=256*1024,(len(data),'catalog remains bounded')
    OUT.mkdir()
    for p in mutable:
        target=OUT/'previous'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    shutil.copyfile(REPORT,snapshot)
    summary=dict(status='three_role_finite_equality_restoration_mixed_failed_all_role_gate',actual_new_heads=338,actual_new_features=338,actual_new_full_class_gradients=0,actual_new_fixed_error_target_gradients=0,actual_new_margin_gradients=32,actual_finite_proposals=9,actual_restoration_solves=10,new_fits=0,permanent_updates=0,cumulative_heads=17246,cumulative_features=17246,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=12,cumulative_margin_gradients=42,cumulative_all_complete_derivatives=422,role_finite_pass=[False,True,False],safe_candidate_pure_repairs=8,all_parameters_restored=True,all_joint_TRAIN_retention_passed=True,latest_complete_trained_model='V159',latest_complete_trained_quality='failed',root_goal_status='active',short_training_gate_passed=False,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,*audits,independent_bindings]})
    save(OUT/'actual_result_summary.json',summary);mutable[2].write_bytes(data)
    lead='最新机制：[V162三角色真实有限恢复](docs/V162_COMPLETE_FINITE_RESTORATION_RESULTS_20261002.md)。角色1修复8条且0退化，角色0/2失败；0新fit/永久更新。累计17246头/368原类/12旧错/42间隔梯度。短程训练前提未满足；最新完整训练仍V159质量失败。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V162真实有限恢复阶段完成\n\n'+lead+'MCP delivery=v159-delivery，direction=v162-review。角色2负间隔约按1/64收缩，预算末端仍失败，不证明机制无效；角色0局部单侧约束对照不替代真实有限验收。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace('v161-review','v162-review');mutable[3].write_text(tests,encoding='utf-8')
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(data),historic_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v162-review',project_wording_compaction=replacements,new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}))
    print(json.dumps(dict(status='V162_mixed_actual_results_published',catalog_bytes=len(data),historic_records_preserved=len(historic))))

if __name__=='__main__':main()
