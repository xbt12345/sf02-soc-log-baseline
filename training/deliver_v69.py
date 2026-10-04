"""Publish verified results and advance the existing local read-only catalog."""
from datetime import datetime, timezone
from pathlib import Path
from v61_common import read, save, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/'artifacts/v69_support_control_20260921'
    verified=read(run/'verification.json'); result=read(run/'analysis.json')
    assert verified['verification_passed'] and verified['models_restored']==9
    assert not any(x['eligible_for_further_validation'] for x in result['hypothesis_comparisons'].values())
    folder=ROOT/'evidence/2026-09-21/v69_support_control';folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'delivery.json';assert not destination.exists()
    sources=[ROOT/'training'/n for n in ['run_v69_support_control.py','verify_v69_support_control.py','audit_v69_target_results.py','deliver_v69.py']]
    files=sources+[ROOT/'docs/V69_SUPPORT_CONTROL_RESULTS.md']+[p for p in run.rglob('*') if p.is_file()]
    delivery={'status':'source_support_9_fits_failed_quality_no_model_promoted',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'quality_acceptance':False,'all_issues_solved':False,
        'actual_new_fits':9,'target_rows':578,'accepted_repairs':0,'accepted_remaining':578,
        'primary_budget':.01,'platform_used':False,'external_training_data_used':False,
        'primary_candidate_B':result['arms']['B_small_sources']['budgets']['0.01']['historical_effects'],
        'hypothesis_comparisons':result['hypothesis_comparisons'],'verification':verified,
        'validation_scope':'Already inspected official adaptive development: 59640 original rows; fixed 578 diagnostic errors; frozen nonhard historical predictions. Nine support-controlled fits with common held-source calibration and no post-calibration refit. No fresh independent, external or whole-SOC acceptance.',
        'next_direction':'Stop expansion of this observation/model/support-threshold family. New training requires concrete observable discriminative evidence or a falsifiable representation/inductive-bias hypothesis; do not repeat weighting/threshold searches.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}}
    save(destination,delivery)
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    old_catalog_sha=sha(catalog_path)
    catalog['project'].update(as_of='2026-09-21',authoritative_delivery_id='v69-delivery',authoritative_direction_id='v69-direction-review',
        current_summary='v6.9 已完成9次真实拟合及独立回放。小来源方案在固定主操作点修复62条目标，但相对历史基线新增450条M错误，质量未通过，无新模型晋升。可验收修复0条，578条仍未解决。',
        current_direction=['停止当前同表示下的支持量、权重与阈值扩训，两个预登记假设均未通过继续条件。',
            '保留对称比较、共同校准来源和校准后冻结预测器；经验预算不是外折保证。',
            '新增可观察判别依据或可否证表示假设后，才启动新的能力实验；不恢复身份查表或擅改标签。'],
        known_limits=['同一官方来源开发资料已反复查看，不是新盲测或跨企业迁移验证。',
            '全开发Macro-F1为98.39%仍掩盖困难S源平均召回5.35%；12条B不能代表SOC误报率。',
            '局部修复62条不是可接入收益；主操作点三个条件并集74条也不是模型。',
            '源符号不保证真实主体或独立事件；同输入冲突不证明官方标签错误。',
            '仅本地只读目录更新与测试，未测试远端ChatGPT Tunnel。'])
    for doc in catalog['documents']:
        if doc['id'] in ['v67-direction-review','v67-delivery']:
            doc['category']='historical_review' if doc['id'].endswith('review') else 'historical_evidence'
            doc['title']=doc['title'].replace('当前','历史')
            doc['keywords']=[v for v in doc['keywords'] if v not in ['当前','最新','下一步']]
    entries=[{'id':'v69-direction-review','title':'v6.9 当前来源支持对照真实训练结果',
              'path':'docs/V69_SUPPORT_CONTROL_RESULTS.md','category':'current_review',
              'summary':'9次拟合；共同校准、冻结预测器。主操作点62目标修复仍新增450个M错误，停止该配置扩训。',
              'keywords':['当前','最新','方向','下一步','v6.9','训练','ASA','578','未通过']},
             {'id':'v69-delivery','title':'v6.9 当前训练与验证证据',
              'path':destination.relative_to(ROOT).as_posix(),'category':'current_evidence',
              'summary':'9模型、648405行回放、90阈值独立复算。quality_acceptance=false，accepted_repairs=0。',
              'keywords':['当前','最新','v6.9','证据','delivery','结果','quality_acceptance']},
             {'id':'v68-review-plan','title':'v6.8 来源支持对照前的整体复盘与规划',
              'path':'docs/V68_REVIEW_AND_PLAN.md','category':'historical_plan',
              'summary':'来源集中度、判定条件、校准转移和对称比较的规划依据，执行结果见v6.9。',
              'keywords':['v6.8','规划','第一性原理','来源支持']}]
    for doc in entries:doc['sha256']=sha(ROOT/doc['path'])
    assert not set(d['id'] for d in entries)&set(d['id'] for d in catalog['documents'])
    catalog['documents']=entries+catalog['documents'];save(catalog_path,catalog)
    save(folder/'publication.json',{'prior_catalog_sha256':old_catalog_sha,'catalog_sha256':sha(catalog_path),
         'delivery_sha256':sha(destination),'bound_files':len(files),'local_only':True,'remote_tunnel_tested':False})
    print({'bound_files':len(files),'status':delivery['status'],'quality_acceptance':False})


if __name__=='__main__':main()
