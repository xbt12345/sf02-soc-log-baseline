"""Publish no-fit direction review without changing the latest training receipt."""
import json
from datetime import datetime
from pathlib import Path
from run_v75 import ROOT, save, sha
from v108_root_evidence_audit import DEST


def main():
    verification=json.loads((DEST/'verification.json').read_text(encoding='utf-8'))
    assert verification['all_checks_passed']
    assert all(sha(DEST/name)==value for name,value in verification['artifact_sha256'].items())
    report='docs/V108_ROOT_CAUSE_RESEARCH_AND_REVISED_PLAN.md'
    summary=('最新实际训练仍为v10.7：12次拟合、0次校准，质量未通过，无模型晋升。'
      'v10.8仅做零拟合根因审查：全部112807条ASA追踪；前三S组占91.186%；'
      'N2新增3102条S错误，统一阈值保持318条M错误预算时最多判对28869条S。'
      '低输入冲突不能证明稳定语义可辨。下一步先做固定频次的冻结表示读出诊断，'
      '再按真实跨来源证据决定是否重训表示；不盲目扩种子、调阈值或复制旧配权方案。')
    for name,link in [('README.md',report),('docs/TRAINING_PLAN.md',Path(report).name)]:
        p=ROOT/name;old=p.read_text(encoding='utf-8');head,body=old.split('\n',1)
        assert 'V108_ROOT_CAUSE' not in old
        p.write_text(head+'\n\n当前方向审查（v10.8，未新增训练）：**'+summary+'** [根因证据与下一轮方案]('+link+')。\n'+body,encoding='utf-8')
    receipt_path=ROOT/'evidence/2026-09-28/v108_root_review/review.json'
    assert not receipt_path.exists();receipt_path.parent.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/report,DEST/'verification.json',DEST/'root_evidence.json',DEST/'raw_trace_review.json',DEST/'resolution_and_metadata_audit.json']
    paths.extend(ROOT/'training'/n for n in ['v108_root_evidence_audit.py','v108_raw_trace_review.py','v108_resolution_audit.py','v108_verify_review.py'])
    receipt={'status':'v108_root_review_completed_no_fit','created_at':datetime.now().astimezone().isoformat(),
      'actual_classifier_fits_this_user_request':0,'actual_calibration_fits':0,'model_promoted':False,
      'quality_acceptance':False,'latest_actual_training_delivery':'v107-delivery',
      'review_scope':'All ASA training-file rows, studied OOF and in-fit predictions; not a fresh blind test.',
      'private_answer_used':False,'platform_used':False,'external_training_data_used':False,
      'source_sha256':sha(__file__),
      'execution_notes':['Coarse behavior support was corrected to retain known protocol/roles when a complete port is unavailable; only the corrected no-fit outputs are published.',
         'Verification first hit the Windows default GBK decoder on a UTF-8 Chinese JSON; explicit UTF-8 fixed it and all checks were rerun.'],
      'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    save(receipt_path,receipt)
    cp=ROOT/'mcp_readonly/catalog.json';cat=json.loads(cp.read_text(encoding='utf-8'))
    assert cat['project']['authoritative_delivery_id']=='v107-delivery'
    assert cat['project']['authoritative_direction_id']=='v107-direction-review'
    cat['project'].update(current_summary=summary,authoritative_direction_id='v108-direction-review',
      current_direction=['先固定V107匹配基线，以冻结表示的收敛M/S读出探针区分学习与读出缺口；尚未拟合。',
        '真实原文、正文事实和记录端口分开审计；保留全部官方行和标签，不以输入唯一性替代安全证据。',
        'CNC/DFR及配权只在适用条件有新证据时复开；研究继续与正式晋升分开判断。'],
      known_limits=['前三个S来源组占91.186%；N1/N2在其他3002条S中仅判对1629/1595，平均成绩掩盖广泛缺口。',
        '3102条新退化S中652条目的端口已脱敏，2232条缺少训练侧相同完整行为的S来源；粗行为两类支持仍在。',
        '全部现有折已用于研究；官方没有提供具体M/S判定规则和脱敏实体时钟关联保证；未证明外部迁移。'])
    new=[('v108-direction-review','v10.8 根因剖析与分阶段方案',report,'current_review',summary),
         ('v108-evidence','v10.8 训练支持与排序根因核查','artifacts/v108_root_evidence_review_20260928/root_evidence.json','current_evidence','0次拟合，当前新折训练支持、原行计数、组集中度和非部署阈值上界。'),
         ('v108-resolution','v10.8 正文事实与记录源端口审查','artifacts/v108_root_evidence_review_20260928/resolution_and_metadata_audit.json','current_evidence','分解477维正文事实和18维记录源端口；低完整向量冲突不等于稳定语义可辨。'),
         ('v108-review-receipt','v10.8 零拟合审查凭据',receipt_path.relative_to(ROOT).as_posix(),'current_evidence','方向审查凭据；最新实际训练仍由v107-delivery负责。')]
    for e in cat['documents']:
        if e['id']=='v107-direction-review':e['category']='historical_review'
        if e['id']=='training-plan':e.update(summary=summary,sha256=sha(ROOT/e['path']))
    cat['documents']=[{'id':i,'title':t,'path':p,'category':c,'summary':s,'keywords':['v10.8','当前','根因','M/S','来源','读出','匹配训练'],'sha256':sha(ROOT/p)} for i,t,p,c,s in new]+cat['documents']
    save(cp,cat)
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    text=test.read_text(encoding='utf-8').replace('v107-direction-review','v108-direction-review')
    test.write_text(text,encoding='utf-8')
    print(json.dumps({'direction':'v108-direction-review','latest_training':'v107-delivery','new_fits':0,'review_sha256':sha(receipt_path)},ensure_ascii=False))


if __name__=='__main__':main()
