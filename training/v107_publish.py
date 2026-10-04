"""Publish verified V107 developmental training without promoting a model."""
import json
from pathlib import Path

from run_v75 import ROOT, save, sha
from v107_matched_training import DEST


def main():
    e=json.loads((DEST/'evaluation.json').read_text(encoding='utf-8'))
    a=json.loads((DEST/'adversarial_decision.json').read_text(encoding='utf-8'))
    reg=json.loads((DEST/'registration.json').read_text(encoding='utf-8'))
    assert e['first_wave_fits']==a['training_fits']==12
    assert e['continue_replication'] and not a['decision']['replicate_N2_seeds']
    assert e['model_promoted'] is False and a['decision']['promote_N2'] is False
    assert e['registration_sha256']==sha(DEST/'registration.json')
    assert e['source_sha256']==sha(ROOT/'training/v107_evaluate.py')
    assert a['source_sha256']==sha(ROOT/'training/v107_adversarial_review.py')
    assert reg['source_sha256']==sha(ROOT/'training/v107_matched_training.py')
    fits={}
    for fold in (0,1,2):
        for view in ('N1','N2'):
            for arm in ('teacher','TabM25_seed10201'):
                p=DEST/f'fold{fold}_{view}_{arm}'
                fit=json.loads((p/'fit.json').read_text(encoding='utf-8'))
                model=p/('teacher.joblib' if arm=='teacher' else 'model.pt')
                pred=p/('scores_all_input_ids.npy' if arm=='teacher' else 'ASA_input_prob.npy')
                assert fit['model_sha256']==sha(model)
                assert fit['scores_sha256' if arm=='teacher' else 'prob_sha256']==sha(pred)
                assert fit['heldout_gradient_rows']==0
                fits[p.name]={'fit_sha256':sha(p/'fit.json'),
                              'model_sha256':fit['model_sha256'],
                              'prediction_sha256':sha(pred)}
    assert len(fits)==12
    title='v10.7 来源闭合折匹配训练与对抗性停止决定'
    summary=('v10.7完成N1/N2×线性/TabM25三折共12次拟合、0次校准。'
             '同架构匹配对照中，N2 TabM 的ASA错误2412→5490，S判对31965→28865；'
             '原冻结门槛虽全通过，但遗漏同架构对照，已记录并停止N2扩种子。'
             'N1仍有198/233个S来源组零召回；质量未通过、无模型晋升。'
             '所有结果属于已观察官方数据的开发折外评估，未证明外部迁移。')
    doc='docs/V107_MATCHED_TRAINING_RESULTS_AND_ADVERSARIAL_DECISION.md'
    for path,link in ((ROOT/'README.md',doc),
                      (ROOT/'docs/TRAINING_PLAN.md',Path(doc).name)):
        content=path.read_text(encoding='utf-8')
        heading,rest=content.split('\n',1)
        path.write_text(heading+'\n\n当前训练执行与决定（2026-09-28）：**'+summary+
                        '** ['+title+']('+link+')。下方保留历史阶段。\n'+rest,
                        encoding='utf-8')
    delivery_path=ROOT/'evidence/2026-09-28/v107_matched_round/delivery.json'
    assert not delivery_path.exists()
    delivery_path.parent.mkdir(parents=True,exist_ok=True)
    artifacts=[doc,'training/v107_matched_training.py','training/v107_evaluate.py',
               'training/v107_adversarial_review.py',
               'artifacts/v107_matched_training_20260928/registration.json',
               'artifacts/v107_matched_training_20260928/evaluation.json',
               'artifacts/v107_matched_training_20260928/adversarial_decision.json',
               'artifacts/v107_matched_training_20260928/OOF_ASA_ledger.parquet']
    delivery={'status':'v107_matched_training_executed_no_promotion',
              'actual_classifier_fits_this_user_request':12,
              'actual_calibration_fits':0,'official_training_rows':2056871,
              'same_ASA_OOF_rows':112807,'quality_acceptance':False,
              'model_promoted':False,'platform_used':False,
              'external_training_data_used':False,'private_answer_used':False,
              'current_report':doc,
              'validation_scope':'Body-source-closed three-fold internal OOF, all folds previously studied; no external blind validation.',
              'frozen_gate_passed_but_was_insufficient':True,
              'matched_N2_vs_N1_extra_ASA_errors':3078,
              'matched_N2_vs_N1_S_correct_delta':-3100,
              'S_zero_recall_groups_N1':198,'S_zero_recall_groups_N2':214,
              'conditional_replication_executed':False,
              'artifact_sha256':{p:sha(ROOT/p) for p in artifacts},
              'fit_receipts':fits,'source_sha256':sha(__file__)}
    save(delivery_path,delivery)
    cp=ROOT/'mcp_readonly/catalog.json'
    cat=json.loads(cp.read_text(encoding='utf-8'))
    assert cat['project']['authoritative_delivery_id']=='v104-delivery'
    assert cat['project']['authoritative_direction_id']=='v106-direction-review'
    cat['project'].update(current_summary=summary,
      authoritative_delivery_id='v107-delivery',
      authoritative_direction_id='v107-direction-review',
      current_direction=[
        '停止N2脱敏包装合并分支和条件性扩种子；同架构同折对照显示S严重退化。',
        '保留N1 TabM作为研究基线但不晋升；先逐组核查剩余稳定行为证据和独立双类来源支持。',
        '新拟合须先冻结同架构对照、M/S逐类与来源组门槛；外部迁移需未知来源或官方隐藏测试。'],
      known_limits=[
        '原冻结门槛全部通过但漏掉N2对N1 TabM同架构比较，不能据此宣称N2有效。',
        'N1的S零召回来源组仍为198/233；大组和人工脱敏外观可抬高行级成绩。',
        '正文源地址未经真实实体认证，官方开发三折不是新环境；无模型晋升。'])
    entries=[
      ('v107-direction-review',title,doc,'current_review',summary),
      ('v107-delivery','v10.7 匹配训练交付凭据',str(delivery_path.relative_to(ROOT)).replace('\\','/'),
       'current_delivery','12次拟合的模型/预测哈希、质量边界与停止决定。'),
      ('v107-evaluation','v10.7 逐类折外验证',
       'artifacts/v107_matched_training_20260928/evaluation.json','current_evidence',
       '四臂逐类、逐折、逐组及全任务组合结果；保留原冻结门槛通过记录。'),
      ('v107-adversarial-decision','v10.7 同架构负对照审查',
       'artifacts/v107_matched_training_20260928/adversarial_decision.json','current_evidence',
       'N2相对N1同模型显著退化；停止额外种子与模型晋升。')]
    new=[{'id':i,'title':t,'path':p,'category':c,'summary':s,
          'keywords':['v10.7','当前','最新','匹配训练','脱敏','M/S','来源组'],
          'sha256':sha(ROOT/p)} for i,t,p,c,s in entries]
    for entry in cat['documents']:
        if entry['id']=='v106-direction-review':entry['category']='historical_review'
        if entry['id'] in ('v106-diagnosis','v106-source-audit'):entry['category']='historical_evidence'
        if entry['id']=='v106-next-contract':entry['category']='executed_plan'
        if entry['id']=='v104-delivery':entry['category']='historical_delivery'
        if entry['id']=='training-plan':entry.update(summary=summary,sha256=sha(ROOT/entry['path']))
    cat['documents']=new+cat['documents']
    save(cp,cat)
    tests=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    content=tests.read_text(encoding='utf-8')
    content=content.replace('v106-direction-review','v107-direction-review')
    content=content.replace('v104-delivery','v107-delivery')
    content=content.replace('v104_official_coverage_and_independent_models_executed_no_promotion',
                            'v107_matched_training_executed_no_promotion')
    content=content.replace('根因复核','匹配训练')
    content=content.replace('15次分类器拟合','12次拟合')
    content=content.replace("        self.assertIn('完整开发回放仍为v7.9', status.current_summary)\n",'')
    tests.write_text(content,encoding='utf-8')
    print(json.dumps({'stage':'v107_published','fits':len(fits),
                      'delivery_sha256':sha(delivery_path),'summary':summary},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
