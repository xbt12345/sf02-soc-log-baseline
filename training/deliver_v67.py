"""Bind completed experiments and publish honest read-only project status."""
from datetime import datetime,timezone
from pathlib import Path
from train_v65_rank_heads import ROOT
from v61_common import read,save,sha


def main():
    final=ROOT/'artifacts/v67_delivery_tables_20260920';check=read(final/'verification.json')
    assert check['verification_passed'] and check['all_v67_models_verified']==96
    assert check['accepted_repairs']==0 and check['accepted_remaining']==578
    out=ROOT/'evidence/2026-09-20/v67_target_578';out.mkdir(parents=True,exist_ok=True)
    destination=out/'delivery.json';assert not destination.exists()
    roots=[ROOT/'artifacts'/name for name in ['v67_targeted_20260920','v67_specialist_20260920',
        'v67_expansion_20260920','v67_row_review_20260920','v67_verification_r2_20260920',
        'v67_protocol_20260920','v67_dual_budget_20260920','v67_final_evidence_20260920',
        'v67_scale_free_20260920','v67_delivery_tables_20260920']]
    sources=[ROOT/'training'/name for name in ['run_v67_targeted.py','run_v67_specialist.py','run_v67_expansion.py',
        'audit_v67_rows.py','verify_v67.py','test_v67.py','refine_v67_protocol.py','run_v67_dual_budget.py',
        'finish_v67_evidence.py','run_v67_scale_free.py','verify_v67_scale_free.py','deliver_v67.py']]
    files=sources+[ROOT/'docs/V67_TARGET_578_REVIEW.md']
    for root in roots:files += [p for p in root.rglob('*') if p.is_file()]
    previous=read(ROOT/'evidence/2026-09-20/v66_issue_resolution/delivery.json')
    for name,digest in previous['artifact_sha256'].items():assert sha(ROOT/name)==digest
    result={'status':'target_578_audited_96_fits_failed_quality_no_model_promoted',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'quality_acceptance':False,
        'validation_scope':'Repeatedly observed official development data only: 59640 source-held-out rows, fixed 578 hard-S target, 96 fitted estimators and nested calibration. Auxiliary historical validation roles retired for expanded training. No independent external or full-SOC acceptance; no new model promoted.',
        'all_issues_solved':False,'target_rows':578,'target_sources':162,'accepted_repairs':0,'accepted_remaining':578,
        'new_model_fits':96,'model_or_decision_configurations':17,'verification':check,
        'implementation_tests_passed':5,'platform_used':False,'external_training_data_used':False,
        'prior_v66_bound_files_unchanged':len(previous['artifact_sha256']),
        'evidence_partition':{'same_encoded_input_seen_as_M_in_training':10,'novel_without_context':236,'novel_with_context':332},
        'main_findings':['Original nonconflict description was only within validation folds; ten cross-fold M counterexamples verified.',
            'Native tree controls, hard specialists, official-data expansion, protocol-specific calibration, dual row/source budgets, and absolute-count ablation actually executed.',
            'Maximum target repair in one candidate is 58, but with 520 newly wrong M and 16 newly wrong S; rejected.',
            'Official support expansion without exact ports fixes 48 targets in four sources, but breaks 150 M and 16 S; rejected.',
            'Count-free retraining fixes 18 targets and breaks 144 M and 16 S; rejected.',
            'Union of isolated repairs across all candidates is 88, not a deployable model; 490 fail every tested candidate.'],
        'retired_historical_validation_roles':['v61 selection','v61 evaluation'],
        'role_change_scope':'39758 already-inspected official records from 3166 disjoint sources admitted as auxiliary training for expansion branches only; their historical validation roles retired. Original 59640-row source folds retained as adaptive development evaluation.',
        'next_direction':'Do not expand these failed configurations. Require new observable evidence for ten cross-fold conflicts, a single-event path for empty-context novelty, and replicated source coverage for context-bearing novelty. Keep official labels and all evaluation rows.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}}
    save(destination,result)
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    project=catalog['project'];project.update(as_of='2026-09-20',
        authoritative_delivery_id='v67-delivery',authoritative_direction_id='v67-direction-review',
        current_summary='v6.7 已完成96次真实拟合及578条逐行核查；候选最大修复58条却新增520条M错误，质量未通过，无新模型晋升。可验收修复0条，原578条仍未解决。',
        current_direction=['保留原标签和全部评价行，10条跨折同输入M/S反例单列。',
            '236条无上下文的新输入与332条有上下文的新输入分开处理；不再统一扩训练轮数。',
            '保留协议分开、行与主体双预算的训练侧校准及完整外折拒绝条件；校准通过不是迁移保证。',
            '当前模型、权重平衡、官方支持扩充和去规模配置均未通过；不原样扩训或接入。'],
        known_limits=['这些是反复观察的适应性开发结果，不是独立外部验证。',
            '辅助训练使用过的39758条历史开发记录已退役其验证角色。',
            '仅12条正常控制且原一折无正常支持，不能估计运营误报率。',
            '88条候选修复并集不能作为一个模型成绩；没有可验收修复。',
            '本轮仅更新本地只读目录并测试；未测试远端ChatGPT Tunnel。'])
    for doc in catalog['documents']:
        if doc['id'] in ['v66-direction-review','v66-delivery']:
            doc['category']='historical_review' if doc['id'].endswith('review') else 'historical_evidence'
            doc['title']=doc['title'].replace('当前','历史')
            doc['keywords']=[k for k in doc['keywords'] if k not in ['当前','最新','下一步']]
    entries=[{'id':'v67-direction-review','title':'v6.7 当前578条困难S逐条核查与训练结果',
        'path':'docs/V67_TARGET_578_REVIEW.md','category':'current_review',
        'summary':'96次拟合和17组判定对照均未通过；逐条清单、三类证据、真实修复与退化及下一步。',
        'keywords':['当前','最新','下一步','方向','模型能力','578','ASA','v6.7','训练','未通过']},
        {'id':'v67-delivery','title':'v6.7 当前578条核查交付证据',
        'path':destination.relative_to(ROOT).as_posix(),'category':'current_evidence',
        'summary':'96个模型重载、1420401行回放；quality_acceptance=false，accepted_repairs=0。',
        'keywords':['当前','最新','v6.7','证据','结果','delivery','578','quality_acceptance']}]
    for doc in entries:doc['sha256']=sha(ROOT/doc['path'])
    catalog['documents']=entries+catalog['documents'];save(catalog_path,catalog)
    print({'bound_files':len(files),'status':result['status'],'delivery_bytes':destination.stat().st_size})


if __name__=='__main__':main()
