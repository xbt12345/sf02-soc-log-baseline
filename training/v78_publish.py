"""Publish actual execution receipts, classwise error ledger, and read-only status."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, OUT, read, save, sha, NAMES
from v78_boundary import DEST


def main():
    p=pd.read_parquet(DEST/'frozen_development/predictions.parquet')
    ch=pd.read_parquet(DEST/'denial_adapter_v4/changed_predictions.parquet')
    a=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet').set_index('event_id').label_binary
    y=a.reindex(p.event_id).map(dict(zip(NAMES,range(3)))).to_numpy();tables=[]
    for n in ['K_base','O_sgd','O_lbfgs','shadow']:
        z=p[n+'_pred'].to_numpy().copy();z[ch.row_position]=ch[n+'_pred'].to_numpy()
        bad=z!=y;d=p.loc[bad,['row_position','event_id','route']].copy();d['model']=n
        d['true_label']=np.array(NAMES)[y[bad]];d['predicted_label']=np.array(NAMES)[z[bad]]
        d['new_parser_applied']=d.row_position.isin(ch.row_position)
        d['scope']='Inspected development error; root cause not automatically inferred from route.'
        tables.append(d)
    errors=pd.concat(tables,ignore_index=True);errors.to_parquet(DEST/'remaining_error_ledger.parquet',index=False)
    errors.groupby(['model','route','true_label','predicted_label','new_parser_applied']).size().rename('rows').reset_index().to_csv(DEST/'remaining_error_counts.csv',index=False)
    assert len(errors[errors.model.eq('O_sgd')])==4823
    summary='v7.8本轮新增4次分类器拟合＋1次校准拟合。固定解析修复使诊断模型困难恶意正确4/6226→6008/6226，完整恶意召回78.47%；旧全量D/F修复后仅30.70%/36.44%。扩训稳定性和质量未通过，无新模型晋升。'
    entry='当前执行（2026-09-22）：**'+summary+'** 逐类统计、解析覆盖、内部/完整回放与全量模型对照已完成；剩余VPC-M 2,664条、ASA-S 1,645条及其他稀缺/正常误报继续单独监督。[本轮真实结果和下一步]({link})。以下为历史阶段。\n\n'
    for path,link in [(ROOT/'README.md','docs/V78_BOUNDARY_EXECUTION.md'),(ROOT/'docs/TRAINING_PLAN.md','V78_BOUNDARY_EXECUTION.md')]:
        s=path.read_text(encoding='utf8');head,tail=s.split('\n\n',1)
        assert 'v7.8本轮新增' not in s
        path.write_text(head+'\n\n'+entry.format(link=link)+tail,encoding='utf8')
    dest=ROOT/'evidence/2026-09-22/v78_boundary';dest.mkdir(parents=True,exist_ok=False)
    paths=list(DEST.rglob('*.json'))+list(DEST.rglob('*.csv'))+list(DEST.rglob('*.parquet'))+list(DEST.glob('*.joblib'))
    paths+=list((ROOT/'training').glob('v78*.py'))+[ROOT/'docs/V78_BOUNDARY_EXECUTION.md']
    paths += [OUT/'facts_encoder.joblib',ROOT/'training/run_v75.py',ROOT/'training/v75_views.py',ROOT/'training/v75_corrective.py',ROOT/'training/v75_metadata.py']
    before=read(DEST/'frozen_development/scoring.json');after=read(DEST/'denial_adapter_v4/scoring.json')
    save(dest/'delivery.json',{'status':'boundary_training_and_parser_repair_completed_quality_not_promoted',
        'quality_acceptance':False,'all_issues_solved':False,'actual_new_fits':5,'actual_classifier_fits':4,'actual_calibration_fits':1,
        'new_full_data_final_fit':False,'full_input_replay_rows':2014052,
        'roles':read(DEST/'contract.json')['role_counts'],'optimizer_subset_rows':104544,'shadow_subset_rows':151647,
        'calibration_candidate_exists':False,'shadow_passed':False,
        'reference_model_for_reporting':'O_sgd diagnostic control; not selected or promoted using development answers',
        'before_parser':before['totals'],'after_parser':after['totals'],
        'parser_evidence':read(DEST/'denial_adapter_v4/independent_verification.json'),
        'previous_full_models':read(DEST/'full_model_adapter/scoring.json')['totals'],
        'implementation_verification':read(DEST/'independent_verification.json'),
        'remaining_reference_errors':4823,'promoted_models':[],
        'validation_scope':'Historical inspected train/development splits and private-answer regression, including post-result parser repair. No fresh blind, external environment, or new all-data final training acceptance.',
        'platform_used':False,'external_training_data_used':False,
        'source_note':'Frozen replay first completed inference; a print-only missing-json import was corrected before scoring rerun. Original bound inference source is retained in frozen_development/inference_source.py. No model or predictions changed.',
        'next_direction':'Preserve evidenced parser fix; controlled normal-pool/selection-objective and shadow stability comparisons; separate VPC zero-M-support transfer from ASA M/S boundary; protect every rare class and normal negatives.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}})
    cat=read(ROOT/'mcp_readonly/catalog.json');cat['project'].update(as_of='2026-09-22',current_summary=summary,
        authoritative_delivery_id='v78-delivery',authoritative_direction_id='v78-direction-review',
        current_direction=['保留可核验的解析修复，按真实类别和载体逐项验收。',
        '同输入下区分正常抽样/目标先验/优化收敛，并通过模拟扩训稳定性后才全量最终训练。',
        'VPC零M监督先做跨格式迁移对照；ASA同时保护M与S，保留全部官方样本及稀缺/正常负例。'],
        known_limits=['4次分类器和1次校准已实际完成；扩训稳定性未通过。',
        '6004条净修复是已查看开发数据上的输入修复收益，未证明外部泛化。',
        '新激活语法在这批开发数据全为M，缺少新增变体的真实B/S反例；不能把deny作为标签规则。',
        '诊断模型M召回78.47%不代表旧全量模型；后者修复后仅30.70%/36.44%。',
        'VPC 2664条M仍全判S，ASA 1645条S判M，51条正常误报及其他稀缺错误尚未解决。'])
    for e in cat['documents']:
        if e['id']=='v77-direction-review':e['category']='historical_review'
        if e['id']=='v75-delivery':e['category']='historical_evidence'
        if e['id']=='training-plan':e['sha256']=sha(ROOT/e['path'])
    cat['documents'][:0]=[
        {'id':'v78-direction-review','title':'v7.8 逐类边界训练、解析缺口修复与真实收益','path':'docs/V78_BOUNDARY_EXECUTION.md',
        'category':'current_review','summary':summary,'keywords':['当前','最新','方向','下一步','训练','恶意','可疑','边界','解析','泛化','v7.8'],
        'sha256':sha(ROOT/'docs/V78_BOUNDARY_EXECUTION.md')},
        {'id':'v78-delivery','title':'v7.8 五次拟合及固定解析修复交付证据','path':'evidence/2026-09-22/v78_boundary/delivery.json',
        'category':'current_evidence','summary':'4分类器＋1校准，6004条开发诊断净修复；全量模型另列，未晋升。',
        'keywords':['实际结果','逐类','召回率','精确率','验证','训练','v7.8'],'sha256':sha(dest/'delivery.json')}]
    save(ROOT/'mcp_readonly/catalog.json',cat)
    path=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=path.read_text(encoding='utf8')
    s=s.replace('v77-direction-review','v78-direction-review').replace('v75-delivery','v78-delivery')
    s=s.replace('four_arm_and_rare_followups_completed_quality_not_promoted','boundary_training_and_parser_repair_completed_quality_not_promoted')
    s=s.replace('逐项验收与对抗性审查','多轮对抗性审查与结论').replace('本轮新增0次拟合','本轮新增4次分类器拟合').replace('v7.5新增8次拟合','1次校准拟合')
    path.write_text(s,encoding='utf8')
    print(json.dumps({'actual_classifier_fits':4,'actual_calibration_fits':1,'quality_acceptance':False,'hashed_files':len(set(paths)),'error_ledger_rows':len(errors)}))


if __name__=='__main__':main()
