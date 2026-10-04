"""Actual failed preseal and post-CUDA resource snapshot, no training claim."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v169_actual_resource_wait_direction_20261002'
ID='v169-resource-wait'
LOG=ROOT/'artifacts/v169_actual_resource_wait_MCP_tests_original_console_20261002.txt'

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def publish():
    assert not OUT.exists();previous=ROOT/'artifacts/v169_commit_ready_direction_20261002';check_bindings(read(previous/'validation.json')['source_sha256'])
    snapshot=ROOT/'artifacts/v169_actual_preseal_resource_shortfall_20261002/snapshot.json';s=read(snapshot);check_bindings(s['source_sha256']);assert not s['formal_OUT_exists'] and not s['formal_PLAN_exists'] and s['resources']['physical_RAM']['shortfall_bytes']>0
    root=ROOT/'artifacts/v169_root_final_preseal_review_20261002/review.json';r=read(root);check_bindings(r['source_sha256']);assert r['supports_physical_seal'] is True and not r['execution_authority']
    original=ROOT/'docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md';body=original.read_bytes()+('\n\n## 当前实际状态：设计审查通过，封存因RAM门槛拒绝\n\n'
      '根实际独立审查已通过最终v13、v5完整依赖、v7资源、契约候选v3和封存器v4，完整18767项哈希及93模块闭包；只支持满足实际资源之后封存。随后实际执行封存器停在physical RAM检查，未生成正式OUT、PLAN、run_seal或registration，0官方头/特征/导数/fit/永久更新，最新训练V164、完整交付V159；三个目标未完成。\n\n'
      'CUDA初始化后快照：可用RAM5,762,314,240 bytes，须6,442,450,944，缺680,136,704 bytes（约649MiB）；commit9,966,776,320≥6,710,886,400，显存7,451,181,056≥536,870,912，磁盘14,338,809,856≥14,283,971,948。这是保存时点的实测，启动前必须再次读取。不得降低门槛、关闭用户应用或在资源未变化时盲重试。未写正式目录/契约，资源变化满足后可按同一根审查、同一封存器重试；若以后已写目录后失败，保留失败状态且不能从中重启。\n\n').encode('utf-8')
    for path in [root,snapshot]:body+=(f'\n### 来源 {path.relative_to(ROOT).as_posix()}；SHA256 `{sha(path)}`\n\n'.encode('utf-8')+path.read_bytes())
    body+=('\n\n## 完整上一方向、原始研究、草案及全部历史限制保留\n\n').encode('utf-8')+(previous/'plan.md').read_bytes();assert len(body)<=256*1024
    mutables=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutables[2]);current=load_catalog(ROOT,mutables[2],256*1024);history={row['id']:row for row in current['documents']};assert len(history)==458 and current['project']['authoritative_direction_id']=='v169-commit-ready'
    for row in history.values():assert sha(ROOT/row['path'])==row['sha256']
    catalog['documents'].insert(0,dict(id=ID,title='V169根审查通过与实际RAM封存拒绝',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V169','实际资源门槛','RAM']))
    project=catalog['project'];project['authoritative_direction_id']=ID;project['current_summary']='V169最终源与资源设计根独立审查通过；实际封存因物理RAM不足拒绝，未写OUT/PLAN，0官方调用/fit。CUDA后RAM缺约649MiB，其余门槛通过，等待实际资源变化，禁止降低门槛或盲重试。最新训练V164、完整交付V159，历史19178头/768导数、9fit/170更新保留，分类未掌握，三目标未完成。';project['current_direction']=['保留v13/契约候选v3/资源v7/整包v5/封存器v4及根实际审查；资源满足后同一版本封存，再执行既定point0和最多6×20配对训练；完整逐行质量仍待真实训练及根独立验收。'];project['known_limits']=['实际封存未成功，无正式执行契约/运行目录/封存，V169正式模型头/特征/导数/fit/更新均0。','RAM6GiB、可用commit6.25GiB、GPU512MiB、磁盘整轮加2GiB固定reserve，CUDA初始化后均须实际满足；缺约649MiB为保存时点快照，不可据此预测可用时刻。','根设计审查支持条件封存，不等同实际资源通过、真实零步、分类收益或完整三目标；所有旧限制、失败证据与历史元数据完整保留。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024;OUT.mkdir()
    for path in mutables:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body);lead='当前状态：[V169根独立审查通过与实际RAM封存拒绝](artifacts/v169_actual_resource_wait_direction_20261002/plan.md)。实际封存未写正式OUT/PLAN，0新官方调用/fit；CUDA后可用物理RAM缺約649MiB，等待资源变化，保持全部门槛与既定源。最新训练V164、完整交付V159，三目标未完成。\n\n'
    mutables[0].write_text(lead+mutables[0].read_text(encoding='utf-8'),encoding='utf-8');mutables[1].write_text(mutables[1].read_text(encoding='utf-8')+'\n\n## V169实际封存资源拒绝\n\n'+lead+'当前direction=v169-resource-wait，458旧文档保留。失败日志v169_seal_prior_pair_training_v4_original_console_20261002.txt；快照v169_actual_preseal_resource_shortfall_20261002/snapshot.json。根支持条件封存，资源外部变化前不重复尝试，不能按未封存状态调用正式模型。\n',encoding='utf-8');mutables[2].write_bytes(encoded);mutables[3].write_text(mutables[3].read_text(encoding='utf-8').replace('v169-commit-ready',ID),encoding='utf-8')
    now={row['id']:row for row in load_catalog(ROOT,mutables[2],256*1024)['documents']};assert len(now)==459 and all(now[k]==v for k,v in history.items())
    paths=[Path(__file__).resolve(),original,root,snapshot,previous/'plan.md',previous/'validation.json',OUT/'plan.md',*mutables];save(OUT/'publication.json',dict(status='V169_actual_failed_preseal_resource_wait_published',historic_records_preserved=458,effective_documents=459,current_direction=ID,latest_actual_training='V164',latest_complete_delivery='V159',official_calls=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}));print(json.dumps(dict(status='V169_actual_resource_wait_published',documents=459,old_records=458,official_calls=0)))

def verify():
    target=OUT/'validation.json';assert not target.exists();check_bindings(read(OUT/'publication.json')['source_sha256']);current=load_catalog(ROOT,ROOT/'mcp_readonly/catalog.json',256*1024);old={r['id']:r for r in load_catalog(ROOT,OUT/'previous/mcp_readonly/catalog.json',256*1024)['documents']};now={r['id']:r for r in current['documents']};assert len(old)==458 and len(now)==459 and all(now[k]==v for k,v in old.items())
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    assert current['project']['authoritative_direction_id']==ID and current['project']['authoritative_delivery_id']=='v159-delivery';log=LOG.read_text(encoding='utf-8');assert 'Ran 23 tests' in log and log.rstrip().endswith('OK')
    assert not (ROOT/'artifacts/v169_prior_pair_training').exists() and not (ROOT/'training/review_policy/v169_prior_pair_execution_contract.json').exists()
    paths=[Path(__file__).resolve(),OUT/'publication.json',OUT/'plan.md',LOG,ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py'];save(target,dict(status='V169_actual_resource_wait_all458_historic_records_and23_local_MCP_tests_verified',current_direction=ID,MCP_tests_passed=23,effective_documents=459,historic_records_preserved=458,latest_actual_training='V164',latest_complete_delivery='V159',official_calls=0,fits=0,permanent_updates=0,execution_authority=False,quality_acceptance=False,classification_mastery=False,three_goals_complete=False,live_MCP_not_verified=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}));print(json.dumps(dict(status='V169_actual_resource_wait_verified',MCP_tests=23,official_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['publish','verify']);args=parser.parse_args();(publish if args.mode=='publish' else verify)()
