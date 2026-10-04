"""Bind completed evidence and refresh the read-only project catalog once."""
import datetime
import importlib.metadata
import json
import platform
from pathlib import Path
import torch
from v61_common import read,save,sha


ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'artifacts/v65_rank_heads_20260920'
destination=ROOT/'evidence/2026-09-20/v65_rank_heads/delivery.json'
assert not destination.exists(), 'Never overwrite a bound delivery'
a=read(run/'analysis.json');registered=read(run/'preregistered.json')
assert not a['gate']['passed'] and not a['quality_acceptance']
assert a['all_checkpoint_metrics_replayed']==90 and len(a['weight_replays'])==15
assert a['pair_exposures_verified']==31584
for name,digest in registered['sources'].items():assert sha(ROOT/'training'/name)==digest
assert sha(ROOT/'training/analyze_v65_rank_heads.py')==a['analysis_source_sha256']
previous=read(ROOT/'evidence/2026-09-20/v64_evidence/delivery.json')
for name,digest in previous['artifact_sha256'].items():assert sha(ROOT/name)==digest,name
testlog=(run/'tests.log').read_text(encoding='utf-16' if (run/'tests.log').read_bytes().startswith(b'\xff\xfe') else 'utf-8')
assert 'Ran 10 tests' in testlog and '\nOK' in testlog
files=[p for p in run.rglob('*') if p.is_file()]+[
    ROOT/'training'/name for name in ['train_v65_rank_heads.py','analyze_v65_rank_heads.py',
        'test_v65_rank_heads.py','test_v65_analysis.py','finalize_v65_review.py']]
files.append(ROOT/'docs/V65_EXECUTION_REVIEW.md')
receipt={'status':a['status'],'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'quality_acceptance':False,'platform_used':False,'neural_training_executed':True,
    'validation_scope':'Three source-disjoint inner folds in original fit, 59,640 rows; six frozen-encoder head fits, 12 epochs each, 90 checkpoint metric replays, 15 weight/probability replays, 10 implementation tests. Adaptive development only, no old selection/outer evaluation, no model promotion or full SOC acceptance.',
    'training_device':torch.cuda.get_device_name(0),'python':platform.python_version(),
    'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','pandas','scikit-learn','pyarrow']},
    'head_training_loop_seconds':sum(t['seconds'] for f in a['folds'].values() for t in f['training'].values()),
    'six_head_fits_executed':True,'epochs_per_head':12,'original_labels_changed':False,
    'raw_port_information_retained':True,'gate':a['gate'],'aggregate_metrics':a['aggregate'],
    'matched_selector_S_delta':a['paired_source']['H1_selected-minus-H0_guarded']['source_S_delta'],
    'normal_control_note':'Selected H1 fails 10 of 12 normal controls; one inner validation fold has no normal rows. No operational false-positive-rate claim.',
    'verification':{'implementation_tests':10,'metric_replays':90,'weight_replays':15,
        'max_replayed_probability_difference':max(x['max_probability_difference'] for x in a['weight_replays'].values()),
        'pair_exposures_reconstructed':a['pair_exposures_verified'],
        'previous_v64_bound_artifacts_unchanged':len(previous['artifact_sha256'])},
    'stopped_branch':'Expansion of this fixed-representation CE plus conditional-ranking configuration',
    'next_direction':'Audit official raw-to-input evidence and safely supportable context before another loss/epoch expansion; use identical selectors on both arms and protect normal controls.',
    'limitations':a['limitations'],
    'artifact_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(files)}}
destination.parent.mkdir(parents=True,exist_ok=True)
save(destination,receipt)
catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
p=catalog['project'];p['as_of']='2026-09-20'
p['current_summary']='v6.5 已实际完成三折六次头部训练及复算；条件排序未通过 +5 pp 收益门槛。S 主体召回相对行最优参照仅 +1.27 pp，相同选模规则下低于普通头 0.46 pp；正常控制仍错 10/12，无新模型晋升。'
p['current_direction']=[
    '停止扩大本轮固定表示的条件排序训练；不追加 epochs 或调低成功门槛。',
    '按失败分支转向官方原日志、规范化输入和可安全关联上下文的证据核对；该后续审计尚未执行。',
    '未来两种训练目标使用相同受约束选模规则，并补充正常控制退化拒绝；不在评价折优化阈值。',
    '保留合法端口和官方标签；新增证据必须先验证跨主体支持，再做单变量输入对照。']
p['known_limits']=[
    '结果来自原 fit 内的自适应开发折，不是嵌套盲测或外部网络验证；没有重新评价旧 selection/outer。',
    '只有 12 条正常控制且一折无正常行，不能估计运营误报率；这是 ASA 定向实验，不是全 SOC 验收。',
    '没有官方 M/S 操作性判据；源符号相同不保证物理实体或真实时间关系。',
    '本次未测试 ChatGPT Tunnel 的远端连接。']
p['authoritative_delivery_id']='v65-delivery';p['authoritative_direction_id']='v65-direction-review'
for entry in catalog['documents']:
    if entry['id'] in ['v64-direction-review','v64-delivery']:
        entry['category']='historical_review' if 'review' in entry['id'] else 'historical_evidence'
        entry['title']=entry['title'].replace('当前','历史')
        entry['keywords']=[k for k in entry['keywords'] if k not in ['最新','当前','下一步']]
new=[{'id':'v65-direction-review','title':'v6.5 当前训练结果与下一步方向审查',
      'path':'docs/V65_EXECUTION_REVIEW.md','category':'current_review',
      'summary':'三折六次条件排序头训练失败；同规则选模无 S 主体优势，正常控制仍失败，停止扩训并转入官方证据核查。',
      'keywords':['最新','当前','下一步','方向','模型能力','v6.5','训练','选模','ASA','M/S','未通过']},
     {'id':'v65-delivery','title':'v6.5 当前交付证据（真实训练，无模型晋升）',
      'path':str(destination.relative_to(ROOT)).replace('\\','/'),'category':'current_evidence',
      'summary':'六个头各 12 epochs；90 检查点指标、15 权重回放、10 测试；排序未通过，正常控制不合格。',
      'keywords':['最新','当前','证据','delivery','quality_acceptance','v6.5','结果','训练']}]
for entry in new:entry['sha256']=sha(ROOT/entry['path'])
catalog['documents']=new+catalog['documents'];save(catalog_path,catalog)
print(json.dumps({'status':a['status'],'delivery':str(destination),'bound_files':len(files),
                  'prior_files_unchanged':len(previous['artifact_sha256'])},ensure_ascii=False))
