"""Publish precise partial-repair status to the existing read-only catalog."""
import datetime
from pathlib import Path
from v61_common import read,save,sha
from train_v65_rank_heads import ROOT


def main():
    out=ROOT/'artifacts/v66_issue_resolution_20260920'
    evidence=ROOT/'evidence/2026-09-20/v66_issue_resolution';evidence.mkdir(parents=True,exist_ok=True)
    delivery=evidence/'delivery.json';assert not delivery.exists()
    audit=read(out/'audit.json');verification=read(out/'verification.json')
    gate=read(out/'normal_gate/analysis.json');repeat=read(out/'supported_normal_gate/analysis.json')
    assert verification['normal_models_restored']==9 and verification['verification_passed']
    assert repeat['normal_control_repair_passed'] and all(g['control_repair_passed'] for g in gate['models'].values())
    issues=[
      {'id':'selection_asymmetry','state':'implementation_repaired_and_historical_replay_verified'},
      {'id':'missing_normal_guard','state':'implementation_repaired_missing_support_fails_closed'},
      {'id':'empty_normal_validation_fold','state':'new_whole_source_manifest_verified_normal_gate_refitted'},
      {'id':'normal_control_classification','state':'narrow_control_repair_verified_not_full_SOC'},
      {'id':'neighbor_cap_as_main_cause','state':'ruled_out_for_current_626_failed_hard_S_rows'},
      {'id':'redacted_port_and_category_OOV','state':'distinguished_unknowns_preserved_no_value_imputation'},
      {'id':'contradictory_observable_labels','state':'diagnosed_original_rows_and_labels_retained_underlying_criteria_unresolved'},
      {'id':'structured_body_endpoint_mapping','state':'ambiguous_maps_quarantined_local_bijections_challenged_no_qualified_new_context'},
      {'id':'ASA_M_S_new_source_transfer','state':'unresolved_no_new_M_S_gain_claim'}]
    save(out/'issue_ledger.json',{'issues':issues,'all_issues_solved':False,'quality_acceptance':False})
    sources=[ROOT/'training'/name for name in ['v66_selection.py','audit_v66_failures.py','probe_v66_normal_gate.py',
      'prepare_v66_supported_folds.py','replicate_v66_normal_gate.py','audit_v66_context_coverage.py',
      'audit_v66_endpoint_mapping.py','audit_v66_context_bridge.py','verify_v66_repairs.py','test_v66_repairs.py','finalize_v66_review.py']]
    files=[p for p in out.rglob('*') if p.is_file()]+sources+[ROOT/'docs/V66_CAUSES_AND_REPAIRS.md']
    result={'status':'targeted_repairs_and_normal_gate_verified_MS_unresolved_no_model_promoted',
      'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'quality_acceptance':False,
      'validation_scope':'Original-fit development only: 59,640 official raw-message replays; symmetric selection and normal guard replay; nine logistic normal-gate fits and weight reloads; new three-source supported normal folds. Normal controls fixed 10/12 to 0/12 without new threat-to-B errors. No new M/S improvement, blind evaluation, full SOC acceptance or operational FPR claim.',
      'linear_normal_gate_fits':9,'new_M_S_models_trained':0,'neural_training_executed':False,'platform_used':False,
      'all_issues_solved':False,'issues':issues,'verification':verification,'implementation_tests_passed':8,
      'normal_control_original_errors':10,'normal_control_errors_after_repair':0,'normal_control_rows':12,'normal_source_symbols':3,
      'added_threat_to_B':0,'threat_rows_evaluated':59628,'new_manifest':read(out/'supported_folds.json'),
      'key_failure_slices':audit['slices'],'structured_body_audit':read(out/'context_coverage.json'),
      'endpoint_mapping':read(out/'endpoint_mapping.json'),'bridge_challenge':read(out/'context_bridge.json'),
      'next_direction':'Retain proven selector/normal-control repairs; resolve observable M/S criteria and endpoint provenance before broader context integration or another loss/epoch expansion. No literal-ID shortcut or automatic relabeling.',
      'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}}
    save(delivery,result)
    cp=ROOT/'mcp_readonly/catalog.json';catalog=read(cp);project=catalog['project']
    project.update(as_of='2026-09-20',authoritative_delivery_id='v66-delivery',authoritative_direction_id='v66-direction-review',
      current_summary='v6.6 已完成9次正常识别门拟合，正常控制错误 10/12→0/12，未新增威胁→正常；已修复选模一致性、正常保护和空正常验证折。地址映射与上下文审计发现证据边界，核心 M/S 仍未通过，无新模型晋升。',
      current_direction=['保留统一选模、正常控制拒绝条件和覆盖合格的新源划分；旧 M/S 权重不能当作新划分已重训结果。',
        '正常门仅在三个源主体的一种控制语法上修复，不作为全 SOC 正常检测器发布。',
        '停止本轮条件排序与加大邻居上限扩训；当前困难错误没有 cap 截断。',
        '结构化地址与正文端点分开处理；多对多映射与跨 collector 关联缺少依据，不自动补上下文。',
        '核心 M/S 仍需可观察判据或合格的新证据；不自动改标签，不将证据不足等同 suspicious。'],
      known_limits=['反复观察的开发数据；没有独立外部验证或全 SOC 质量验收。',
        '正常门可能主要识别语法差异；12 条控制不足以估计运营误报率。',
        '精确支持不足与冲突审计不证明所有非线性泛化不可能；578 条非冲突困难 S 错误仍未解决。',
        '本轮没有测试远端 ChatGPT Tunnel。'])
    for entry in catalog['documents']:
        if entry['id'] in ['v65-direction-review','v65-delivery']:
            entry['category']='historical_review' if 'review' in entry['id'] else 'historical_evidence'
            entry['title']=entry['title'].replace('当前','历史')
            entry['keywords']=[s for s in entry['keywords'] if s not in ['最新','当前','下一步']]
    entries=[{'id':'v66-direction-review','title':'v6.6 当前逐项根因与修复结果','path':'docs/V66_CAUSES_AND_REPAIRS.md',
      'category':'current_review','summary':'选模与正常控制已修复；九次真实拟合，正常 10/12→0/12；逐项核对输入、冲突和错误地址关联，核心 M/S 仍未解决。',
      'keywords':['当前','最新','下一步','方向','模型能力','v6.6','ASA','根因','修复','未通过']},
      {'id':'v66-delivery','title':'v6.6 当前部分修复交付证据','path':delivery.relative_to(ROOT).as_posix(),
       'category':'current_evidence','summary':'九个正常门重载、三折覆盖修复、原文及映射审计；quality_acceptance=false，核心 M/S 无新收益。',
       'keywords':['当前','最新','v6.6','证据','结果','delivery','quality_acceptance']}]
    for entry in entries:entry['sha256']=sha(ROOT/entry['path'])
    catalog['documents']=entries+catalog['documents'];save(cp,catalog)
    print({'status':result['status'],'bound_artifacts':len(files),'all_issues_solved':False})


if __name__=='__main__':main()
