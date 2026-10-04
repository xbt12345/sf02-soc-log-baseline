"""Publish three-role finite qualification; preserve trained-model authority."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v163_results_20261002'
REPORT=ROOT/'docs/V163_COMPLETE_JOINT_FINITE_RESULTS_20261002.md'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists() and not REPORT.exists()
    own=ROOT/'artifacts/v163_saved_joint_restoration_actual_audit_20261002/audit.json';ind=ROOT/'artifacts/v163_independent_actual_restoration_review_20261002/review.json';ib=ind.parent/'pre_review_bindings.json';a,b=read(own),read(ind)
    check_bindings(a['source_sha256']);check_bindings(read(ib)['source_sha256']);assert a['all_three_actual_finite_pass'] and b['all_three_finite_qualified'] and a['actual_new_heads']==b['new_heads_and_features']==328 and a['actual_new_margin_gradients']==b['new_margin_gradients']==0 and a['actual_QP_solves']==b['new_QP_solves']==10 and a['actual_optimizer_iterations']==b['new_optimizer_iterations']==20
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];cat=read(mutable[2]);historic={r['id']:dict(r) for r in cat['documents']};assert cat['project']['authoritative_delivery_id']=='v159-delivery'
    for r in cat['documents']:assert sha(ROOT/r['path'])==r['sha256']
    previous=ROOT/'artifacts/v162_cached_tail_results_20261002/review.md';retained=previous.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1]
    body='''# V163 三角色单侧联合恢复实际结果

三角色均通过真实有限保护与双类固定错误目标下降，参数全部恢复，0新拟合/永久更新。角色1安全修复8条（4M+4S、3来源、3local），角色0/2无旧错修复。该结果仅支持另行封存短程学习；训练侧分类掌握、细行为同类支持与跨来源稳定性均未关闭。最新完整实际训练交付仍V159，六fit完整质量失败。

## 实际范围与结果

保持三个固定B终点、原输入/成员、固定旧错目标与完整原类别分母、基础位移1/16及所有原行保护。新机制把等式恢复换为单侧不等式，并联合约束两类完整位移目标非增；求解后仍需严格可分辨负方向及原16eps/Armijo实际下降。三角色统一最多六个恢复候选，不按角色特供算法。

|角色|实际候选/恢复QP|内部优化迭代|头/特征|新完整导数|纯错误起点→候选|旧错修复/新增错误|
|---|---:|---:|---:|---:|---|---|
|0|2|4|88|0|3278→3278|0/0|
|1|2|4|64|0|1960→1952|8/0|
|2|6|12|176|0|1586→1586|0/0|

完整原行错误分别3300→3300、2032→2024、1698→1698；合计7030→7022、纯错误6824→6816。以上属于各角色有限候选，均未提交为永久训练状态，不能冒称新增8条独立行为或迁移收益。全部署与V138/V140/V142联合旧能力保护通过；退出恢复的完整参数身份与各自原终点相同。

本轮实际328次头/特征、10次QP/恢复、20次内部迭代；QP与恢复是同一外层子问题，不重复计费。新目标梯度、原类别梯度、间隔梯度、拟合、永久更新均0。注册上限582头/特征、102间隔导数、18候选/恢复QP；未扩增或重置。累计17662头/特征、368原类别+12固定错误+42间隔=422完整参数导数。此前V159的6拟合/165更新不变。

## 证据与下一边界

入口 `training/v163_fixed_endpoint_one_sided_restoration_v2.py`，合同 `training/review_policy/v163_fixed_endpoint_one_sided_restoration_contract.json`，8259个物理来源的事前封存 `artifacts/v163_fixed_endpoint_one_sided_restoration_20261002/run_seal.json`。真实全部候选q/logq、原行、来源、完整位移/修正、原单位约束、QP迭代、阻挡和成本保存在同目录role0/1/2。自审 `artifacts/v163_saved_joint_restoration_actual_audit_20261002/audit.json` 与独立原gold/完整向量复算 `artifacts/v163_independent_actual_restoration_review_20261002/review.json` 均实际通过，审查0官方调用。

软件返回成功不是QP最优性或非线性安全证据。全部真实候选包括拒绝候选保留；原单位约束、严格共同下降、完整候选参数哈希和每原行保护实际复算后才给予有限资格。角色2仍需要第六次恢复，保留此前V162四次失败与两次尾段的独立账本。

下一项为每角色最多10个真实接受更新的短程，须先绑定新入口、每状态完整梯度/法向、全原行累计正确保护和独立调用预算。首个实际提交后的新修复进入累计保护；旧错目标不重新挑选，原类别分母不改。每次真实接受状态保存四参数段的梯度/变化、完整原行分类/风险/来源；拒绝、停止与固定终点同样报告。本次封存没有任何训练或正式晋升权限，当前没有新训练完成结论。

## 既有证据限制完整保留

'''+retained
    OUT.mkdir()
    for p in mutable:
        target=OUT/'previous'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    REPORT.write_bytes(body.encode('utf-8'));snapshot=OUT/'review.md';shutil.copyfile(REPORT,snapshot)
    cat['documents'].insert(0,dict(id='v163-review',title='V163三角色有限资格',path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(snapshot),category='review_evidence',keywords=['V163']))
    project=cat['project'];project['authoritative_direction_id']='v163-review';project['current_summary']='V159六fit完整验收失败。V163三角色有限通过，角色1候选8修复，0新fit/更新；累计17662头/422完整导数。下一独立封存短程。';project['current_direction']=['另行封存每角色最多10个真实接受更新，累计保护新修复。'];project['known_limits'][0]='V159完整质量失败；V163有限候选尚未成为训练更新。';project['known_limits'][5]='V163角色1的8修复对应3来源/3local；角色0/2无旧错修复。'
    for r in cat['documents']:
        if r['id'] in historic:assert r==historic[r['id']]
    data=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(data)<=256*1024
    summary=dict(status='all_three_roles_actual_one_sided_joint_finite_qualified_not_trained',actual_new_heads=328,actual_new_features=328,actual_new_gradients=0,actual_QP_and_restoration_solves=10,actual_optimizer_iterations=20,actual_finite_proposals=10,new_fits=0,permanent_updates=0,cumulative_heads=17662,cumulative_features=17662,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=12,cumulative_margin_gradients=42,cumulative_all_complete_derivatives=422,safe_candidate_pure_repairs=[0,8,0],all_three_actual_finite_pass=True,all_parameters_restored=True,all_joint_TRAIN_retention_passed=True,latest_complete_trained_model='V159',latest_complete_trained_quality='failed',root_goal_status='active',next='separately sealed at most ten accepted updates per role with cumulative correct protection',no_short_training_authority_from_this_finite_seal=True,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,own,ind,ib,previous]});save(OUT/'actual_result_summary.json',summary);mutable[2].write_bytes(data)
    lead='最新实际：[V163三角色单侧联合有限资格](docs/V163_COMPLETE_JOINT_FINITE_RESULTS_20261002.md)。三角色保护与双类目标下降均通过，角色1候选修复8条；0新fit/更新，累计17662头/422完整导数。下一另封存每角色最多10真实接受更新；训练掌握尚未通过。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V163统一单侧联合有限机制通过\n\n'+lead+'delivery=v159-delivery，direction=v163-review；有限候选与实际训练严格区分，全部旧元数据和证据限制保留。\n',encoding='utf-8');mutable[3].write_text(mutable[3].read_text(encoding='utf-8').replace('v162-tail-review','v163-review'),encoding='utf-8')
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(data),historic_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v163-review',new_model_calls=0,new_gradients=0,new_fits=0,new_updates=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}));print(json.dumps(dict(status='V163_actual_joint_finite_results_published',catalog_bytes=len(data),historic_records_preserved=len(historic))))

if __name__=='__main__':main()
