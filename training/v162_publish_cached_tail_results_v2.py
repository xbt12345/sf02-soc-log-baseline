"""Publish bounded actual continuation, retain all prior catalog facts."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v162_cached_tail_results_20261002'
REPORT=ROOT/'docs/V162_CACHED_TAIL_ACTUAL_RESULTS_20261002.md'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert OUT.exists() and REPORT.exists() and not (OUT/'publication.json').exists() and not (OUT/'actual_result_summary.json').exists()
    assert sha(REPORT)==sha(OUT/'review.md')
    audit=ROOT/'artifacts/v162_saved_cached_restoration_tail_audit_v2_20261002/audit.json';independent=ROOT/'artifacts/v162_independent_cached_restoration_tail_review_20261002/review.json';ib=independent.parent/'pre_review_bindings.json';a=read(audit);b=read(independent)
    check_bindings(a['source_sha256']);check_bindings(read(ib)['source_sha256']);assert a['actual_new_heads']==b['new_heads_and_features']==88 and a['actual_finite_pass'] and b['actual_finite_pass'] and a['new_fits']==a['permanent_updates']==0
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];cat=read(mutable[2]);historic={r['id']:dict(r) for r in cat['documents']};limits=list(cat['project']['known_limits']);assert cat['project']['authoritative_delivery_id']=='v159-delivery'
    for r in cat['documents']:assert sha(ROOT/r['path'])==r['sha256']
    body='''# V162 两次缓存恢复尾段实际结果

角色2在总第六次恢复通过真实有限保护，0旧错修复，参数已恢复，0新拟合/永久更新。角色0仍未通过有限机制；短程训练前提未满足。最新完整训练仍V159，六fit完整质量失败，三个用户问题均未关闭。

此前四次候选完整保存。本次另行封存最多两个同函数候选，上限88头/特征、0新增梯度/法向/拟合/更新，不修改原算法、起点、16eps/Armijo或全原行保护。第五次同函数间隔−5.8797411384e−13，仍新增同两条S错误；第六次+3.7025937871e−13，0保护退化，OOF纯错误回1586、完整错误回1698，旧错修复0。部署及V138/V140/V142联合保护全部通过，模型身份完全恢复。

本阶段实际88头/特征、2候选/恢复、0新增完整导数。累计17334头/特征、368原类别梯度、12固定错误目标梯度、42间隔梯度，完整参数导数422。旧V159的6拟合/165永久更新不变。两个额外候选的真实结果证明旧四次上限截断了本次同函数恢复，不证明已学会分类或迁移有效。V162角色1此前候选安全修复8条（4M+4S、3来源、3local），仍未提交为训练更新。

下一阶段仅准备单侧联合恢复：三角色同一算法、最多6候选、582头/特征、102新增成对间隔导数，0目标/原类别梯度/拟合/更新，必须独立绑定源与预算后才执行。正式短程学习仍需全三角色实际有限资格与单独封存。

实际入口与封存：`training/v162_cached_function_restoration_tail.py`、`artifacts/v162_cached_function_restoration_tail_20261002/run_seal.json`。原正式逐候选q/logq、原行、来源、完整位移/修正、法向和账本保存在同目录role2。自审：`artifacts/v162_saved_cached_restoration_tail_audit_v2_20261002/audit.json`；独立原gold复算：`artifacts/v162_independent_cached_restoration_tail_review_20261002/review.json`。首版自审把DataFrame传入要求row_position Series的纯辅助函数而失败；v2修复审查器类型，原失败保存，没有重跑模型。首版封存器的项目外路径转换失败也保持，v2修复后才产生正式封存和调用。

## 既有证据限制完整保留

只读目录接近既有256KiB限制；本次将项目级旧版本限制原文移入本附录，并在当前状态保留直接影响下一步的限制。所有旧文档ID、路径、标题、sha和元数据原样保留；原目录另存快照，不增加工具上限。以下每条来自更新前项目known_limits，均保留：

'''+''.join('- '+line+'\n' for line in limits)
    assert REPORT.read_text(encoding='utf-8')==body
    snapshot=OUT/'review.md';cat['documents'].insert(0,dict(id='v162-tail-review',title='V162缓存尾段实际结果',path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(REPORT),category='review_evidence',keywords=['V162']))
    project=cat['project'];project['authoritative_direction_id']='v162-tail-review';project['current_summary']='V159六fit完整验收失败。V162角色1候选修复8条、角色2第6恢复有限通过，角色0失败；0新fit/更新。累计17334头/422完整导数。'
    project['current_direction']=['先统一六次单侧联合有限恢复；全角色资格前不训练。']
    project['known_limits']=['V159完整质量失败；V162有限候选尚未成为训练更新，角色0未过。','训练分类掌握、细行为同类支持、来源外稳定性三个目标均未关闭。','V158及当前已看开发来源，非独立盲验/外部部署；N无新监督。','V159前3实际评价未存完整q；FIT重放与全q精确argmax控制流证据另列，冻结q来源单列。','原混合冲突、全部原行、225558部署正确保护及V138/V140/V142联合能力保留。','角色1的8修复对应3来源/3local；角色2恢复0旧错修复。','全部旧逐版限制原文保存在当前方向文档附录；历史目录元数据不变。']
    for r in cat['documents']:
        if r['id'] in historic:assert r==historic[r['id']]
    data=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(data)<=256*1024
    for p in mutable:
        target=OUT/'previous'/p.relative_to(ROOT);assert sha(target)==sha(p)
    assert sha(REPORT)==cat['documents'][0]['sha256']
    summary=dict(status='role2_two_cached_tail_restorations_actual_finite_pass_not_learning',actual_new_heads=88,actual_new_features=88,actual_new_gradients=0,actual_finite_proposals=2,actual_restoration_solves=2,new_fits=0,permanent_updates=0,cumulative_heads=17334,cumulative_features=17334,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=12,cumulative_margin_gradients=42,cumulative_all_complete_derivatives=422,role2_safe_classification_repairs=0,role1_prior_safe_candidate_repairs=8,role0_actual_finite_pass=False,short_training_gate_passed=False,all_parameters_restored=True,latest_complete_trained_model='V159',latest_complete_trained_quality='failed',root_goal_status='active',quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,audit,independent,ib]})
    save(OUT/'actual_result_summary.json',summary);mutable[2].write_bytes(data)
    lead='最新实际：[V162两次缓存恢复尾段](docs/V162_CACHED_TAIL_ACTUAL_RESULTS_20261002.md)。角色2第6恢复通过但0旧错修复，角色0仍失败；0新fit/更新，累计17334头/422完整导数。短程训练前提未满足。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V162缓存同函数尾段完成\n\n'+lead+'原全部旧版本证据限制已在当前MCP方向附录完整保留；旧文档元数据未改，delivery=v159-delivery，direction=v162-tail-review。\n',encoding='utf-8');mutable[3].write_text(mutable[3].read_text(encoding='utf-8').replace('v162-review','v162-tail-review'),encoding='utf-8')
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(data),historic_document_metadata_preserved=len(historic),prior_project_limits_retained_exactly_in_current_direction=len(limits),authoritative_delivery='v159-delivery',authoritative_direction='v162-tail-review',new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}));print(json.dumps(dict(status='V162_cached_tail_actual_results_published',catalog_bytes=len(data),historic_records_preserved=len(historic))))

if __name__=='__main__':main()
