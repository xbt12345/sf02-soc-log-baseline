"""Publish the actual V121 failure without changing its sealed training evidence."""
import json
from pathlib import Path

from experiment_review import ROOT, read, sha, check_bindings
from v121_train import DEST, PLAN


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    delivery=DEST/'delivery.json'
    if delivery.exists():raise FileExistsError('Existing delivery is immutable.')
    check_bindings(read(DEST/'run_seal.json')['source_sha256'])
    check_bindings(read(DEST/'postfit_audit.json')['source_sha256'])
    result=read(DEST/'primary_evaluation.json')
    audit=read(DEST/'postfit_audit.json')
    if (result['classifier_fits_new']!=6 or result['confirmation_allowed'] or
            audit['optimizer_steps']!=8900 or audit['confirmation_fits']!=0 or
            list(DEST.glob('seed10202_*')) or list(DEST.glob('seed10203_*'))):
        raise ValueError('Registered stop decision or fit count changed.')
    summary=('V121按V120完成三折两臂6次新拟合、8900次更新、0次校准。固定25轮新A的ASA错误2560，'
      '质量守恒分批B为4416；M错310→1062，S错2250→3354，三个折均无收益。'
      '完整任务错误2667→4523；逐行真值、模型重放和训练质量守恒通过。主质量门槛失败，'
      '预登记的后12次种子确认未启动，无模型晋升。最新实际训练为V121；下一步先核查支持缺口及跨来源M/S的可观察判据。')
    report=ROOT/'docs/V121_BATCH_TRAINING_RESULTS_AND_STOP.md'
    case=ROOT/'training/review_policy/v121_batch_failure_case.json'
    files=[report,case,PLAN,DEST/'run_seal.json',DEST/'registration.json',
           DEST/'primary_evaluation.json',DEST/'postfit_audit.json',
           DEST/'expert_ASA_predictions.parquet',DEST/'full_prediction_ledger.parquet',
           DEST/'support_results.json',DEST/'source_group_changes.csv',
           ROOT/'training/v121_train.py',ROOT/'training/v121_evaluate.py',
           ROOT/'training/v121_confirm.py',ROOT/'training/v121_postfit_audit.py',Path(__file__)]
    for fold in range(3):
        for arm in ('A','B'):
            folder=DEST/f'fold{fold}_{arm}'
            files += [folder/'fit.json',folder/'started.json',folder/'progress.json',
                      folder/'checkpoints.json',folder/'epoch25_model.pt',folder/'epoch25_prob.npy']
    save(delivery,{'status':'v121_paired_batch_executed_failed_quality_gates',
        'summary':summary,'latest_actual_training':'V121','classifier_fits_new':6,
        'optimizer_steps':8900,'calibration_fits':0,'confirmation_fits':0,
        'quality_acceptance':False,'model_promoted':False,
        'validation_scope':'Actual six-fit matched source-closed official development folds; full original-row three-class replay, independent official-label recount and model replay. No external blind test.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    for rel,link in [('README.md','docs/'+report.name),('docs/TRAINING_PLAN.md',report.name)]:
        p=ROOT/rel;s=p.read_text(encoding='utf-8')
        if report.name in s:raise ValueError('Current summary already present: '+rel)
        s=s.replace('当前下一轮训练方案（V120）','历史训练方案（V120；V121已执行）',1)
        head,body=s.split('\n',1)
        p.write_text(head+'\n\n最新实际训练与停止决定（V121）：**'+summary+'** [六次拟合、实际错误与失败归因]('+link+')。\n'+body,encoding='utf-8')
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    project=catalog['project']
    if project['authoritative_delivery_id']!='v116-delivery' or project['authoritative_direction_id']!='v120-direction-review':
        raise ValueError('Unexpected prior MCP authority.')
    project.update(current_summary=summary,authoritative_delivery_id='v121-delivery',
                   authoritative_direction_id='v121-review',
                   current_direction=['停止V120质量守恒批次配置；六次主对照三个折均退化，后12次未启动。',
                      '保存匹配A/B及原始逐行证据，分析未见参数、同类支持缺口与两个最大退化关联组。',
                      '新方法先提出可观察且跨组一致的M/S判据，再登记单因素训练和完整三分类验收。'],
                   known_limits=['当前结果是已查看的官方开发折，不证明其他真实环境的分类能力。',
                     '注册与重放保证这一轮执行可核对，不证明未知M/S标签依据已经可观察。',
                     '较低训练误差、CE或冻结梯度方差不保证来源外收益；数值非确定性仍须在未来确认。'])
    for e in catalog['documents']:
        if e['id']=='v120-direction-review':e['category']='historical_plan'
        if e['path'] in ('README.md','docs/TRAINING_PLAN.md'):
            e.update(summary=summary,sha256=sha(ROOT/e['path']))
    entries=[('v121-review',report,'V121 六次拟合结果、失败归因与停止决定'),
             ('v121-delivery',delivery,'V121 训练交付与身份凭据'),
             ('v121-evaluation',DEST/'primary_evaluation.json','V121 完整任务与ASA质量复算'),
             ('v121-postfit',DEST/'postfit_audit.json','V121 独立逐行审查与执行检查'),
             ('v121-historical-case',case,'V121 分批机制训练反例')]
    catalog['documents']=[{'id':ident,'title':title,'path':path.relative_to(ROOT).as_posix(),
        'category':'current_review' if ident=='v121-review' else 'current_evidence',
        'summary':summary,'sha256':sha(path),
        'keywords':['V121','当前','最新','实际训练','结果','错误','停止决定','下一步']}
        for ident,path,title in entries]+catalog['documents']
    save(catalog_path,catalog)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=test.read_text(encoding='utf-8')
    if 'v120-direction-review' not in s:raise ValueError('MCP test has unexpected authority.')
    s=s.replace('v120-direction-review','v121-review').replace('v116-delivery','v121-delivery')
    s=s.replace('v116_nested_selection_executed_failed_quality_gates','v121_paired_batch_executed_failed_quality_gates')
    s=s.replace("self.assertIn('3次内层、3次外层', status.current_summary)",
                "self.assertIn('三折两臂', status.current_summary)")
    test.write_text(s,encoding='utf-8')
    print(json.dumps({'round':'V121','classifier_fits_new':6,'confirmation_fits':0,
                      'quality_acceptance':False,'M_errors':[310,1062],
                      'S_errors':[2250,3354]},ensure_ascii=False))


if __name__=='__main__':main()
