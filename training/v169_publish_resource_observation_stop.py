"""Publish completed observation and external-resource stop; no new diagnostics."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog
OUT=ROOT/'artifacts/v169_resource_observation_stop_direction_20261002'
ID='v169-resource-observation-wait'
LOG=ROOT/'artifacts/v169_resource_observation_stop_MCP_tests_original_console_20261002.txt'

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    assert not path.exists();path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def bindings(value):
    for key,expected in value.items():assert sha(ROOT/key)==expected

def publish():
    assert not OUT.exists()
    previous=ROOT/'artifacts/v169_registered_startup_wait_direction_20261002'
    bindings(read(previous/'validation.json')['source_sha256'])
    diag=ROOT/'artifacts/v169_same_process_initial_resource_diagnostic_v2_20261002'
    report=read(diag/'review.json');decision=read(diag/'decision.json')
    bindings(report['source_sha256']);bindings(decision['source_sha256'])
    assert report['actual_require_calls']==1 and report['elapsed_seconds']<180
    assert all(report[k]==decision[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates'])
    assert sorted(p.name for p in (ROOT/'artifacts/v169_prior_pair_training').iterdir())==['registration.json','run_seal.json']
    mutables=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    current=load_catalog(ROOT,mutables[2],256*1024);old={r['id']:r for r in current['documents']}
    assert len(old)==460 and current['project']['authoritative_direction_id']=='v169-registered-startup-wait'
    for row in old.values():assert sha(ROOT/row['path'])==row['sha256']
    body=('# V169当前实际停止：真实RAM和commit不足，未进入任何训练\n\n'
          '同一已封存v14入口的首次启动及retry1均在initial检查拒绝，未建role/backend、未完成真实零步；0新官方头/特征/导数/fit/更新。不能将registration、进程启动或0调用资源观察算成训练实验。\n\n'
          '根授权一次同进程实际require(initial)观察。首版广域trace因观察器开销过大，经核实PID归属后只停止该诊断和包装进程，原source/console及终止凭据保留。纠正版按v2/v5函数code对象直接过滤，非目标frame返回None，一次require在54.7秒内完成，满足180秒上限。没有改变任何gate返回值、系统状态、阈值、模型/训练源或预算。\n\n'
          '实际v2前CUDA读取瞬间RAM3294277632<5637144576，commit4405325824<6710886400bytes；诊断开始时两项已不足。完整封存哈希及继承基础契约核验后，进程常驻仅增加7159808bytes、私有commit增加7630848bytes。没有支持继续降低门槛、修复preCUDA检查器或第三次main盲重试的证据；后续v3/v5政策检查没有因前段哈希完成而自动通过。\n\n'
          'CUDA初始化后保存时点RAM还缺2433847296bytes（约2.27GiB）、commit还缺2483826688bytes（约2.31GiB）。这些是保存时点，不是未来 readiness，也不重构retry1原失败瞬间。只读应用私有工作集观察不授予关闭用户应用或未归属node服务的权限；SF02/v169可归属node/python helper复查为0。没有用户进程被关闭或trim。\n\n'
          '根要求现在停止新增训练、资源诊断和同构资格准备，等待人或外部明确释放足够RAM与commit并保持稳定。恢复信号后按已有授权核实完整源和即时RAM5.25GiB/commit6.25GiB/GPU512MiB/完整磁盘门槛，再继续同一sealed、0后台/0调用入口；起始资源失败不追加fit，进入正式fit后的失败仍严格执行原全局停止约束。原封存、数据/源码、失败日志全保留。\n\n'
          '历史19178头/768完整导数、V159以来9fits/170更新不变；最新实际训练V164、完整交付V159，三个总目标、完整三分类与独立来源质量均未完成，无模型晋升。\n\n').encode('utf-8')
    for path in [diag/'review.json',diag/'decision.json']:
        body+=(f'\n## 实际原始来源 {path.relative_to(ROOT).as_posix()}；SHA256 `{sha(path)}`\n\n'.encode('utf-8')+path.read_bytes())
    body+=b'\n\n## Previous complete direction and all inherited scientific constraints\n\n'+(previous/'plan.md').read_bytes()
    assert len(body)<=256*1024
    catalog=read(mutables[2]);catalog['documents'].insert(0,dict(id=ID,title='V169同进程真实资源拒绝与停止，0正式调用',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V169','真实资源读取','RAM','commit','停止']))
    project=catalog['project'];project['authoritative_direction_id']=ID
    project['current_summary']='V169已封存v14首次启动/retry1均initial资源拒绝，未建backend/role、0官方调用/fit/更新，真实零步未完成。纠正后的同进程一次实际require54.7s完成：真实读取RAM3294277632、commit4405325824均不足；核验常驻仅增7159808bytes，不支持改gate/门槛。CUDA后快照RAM缺约2.27GiB、commit缺约2.31GiB。根要求停止新增训练/资源诊断/同构准备，等待人或外部明确恢复并稳定后再核源及门槛；原封存、日志、全部旧限制保留。最新训练V164、完整交付V159，三目标未完成。'
    project['current_direction']=['只保留/发布实际停止状态和恢复条件，不继续同构资源诊断或盲重试main。待人或外部明确恢复RAM和commit并保持稳定后，已有恢复授权继续有效；核完整源/即时全门槛，起始0后台0调用资源失败不追加fit，进入过正式fit则遵守原全局停止约束。']
    project['known_limits']=['registration不是正式零步/训练证明；首次与retry1均前CUDA RAM拒绝，无role目录/官方调用/fit/更新。','纠正的同进程真实读取同时RAM/commit不足，核验自身常驻仅增约6.8MiB；没有降低资源/科学门槛或改检查器的证据。','RAM缺2.27GiB、commit缺2.31GiB只是CUDA后保存时点，恢复信号后须重新核实，不依据预查询预测后续入口；原失败、首版观察器终止、460旧文档与全部科学限制保留。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    OUT.mkdir()
    for path in mutables:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body)
    lead='当前实际：[V169真实资源读取拒绝与停止](artifacts/v169_resource_observation_stop_direction_20261002/plan.md)。已封存但两次入口均未进backend/真零步，0官方调用/fit/更新。纠正的同进程诊断54.7s完成，真实RAM/commit不足；停止新增训练及资源诊断，等人或外部明确恢复并稳定，再按已有授权核源/门槛。最新训练V164、完整交付V159，三目标未完成。\n\n'
    mutables[0].write_text(lead+mutables[0].read_text(encoding='utf-8'),encoding='utf-8')
    mutables[1].write_text(mutables[1].read_text(encoding='utf-8')+'\n\n## V169同进程真实资源不足，停止新增准备\n\n'+lead+'恢复条件：RAM5.25GiB/可用commit6.25GiB/GPU512MiB/整轮磁盘门槛全部真实满足并稳定。起始0调用无role目录失败不算fit；原封存与两个失败控制台保持，不重新sealer/改源/放宽阈值。\n',encoding='utf-8')
    mutables[2].write_bytes(encoded);mutables[3].write_text(mutables[3].read_text(encoding='utf-8').replace('v169-registered-startup-wait',ID),encoding='utf-8')
    now={r['id']:r for r in load_catalog(ROOT,mutables[2],256*1024)['documents']};assert len(now)==461 and all(now[k]==v for k,v in old.items())
    sources=[Path(__file__).resolve(),previous/'validation.json',previous/'plan.md',diag/'review.json',diag/'decision.json',diag/'decision.md',diag/'process_memory_snapshot.json',ROOT/'artifacts/v169_same_process_initial_resource_diagnostic_20261002/termination.json',OUT/'plan.md',*mutables]
    save(OUT/'publication.json',dict(status='V169_actual_resource_observation_stop_published',current_direction=ID,effective_documents=461,historic_records_preserved=460,new_training_and_resource_diagnostics_stopped=True,waiting_explicit_external_resource_recovery=True,official_calls=0,fits=0,permanent_updates=0,quality_acceptance=False,three_goals_complete=False,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
    print(json.dumps(dict(status='resource_observation_stop_published',documents=461,old_preserved=460,official_calls=0)))

def verify():
    bindings(read(OUT/'publication.json')['source_sha256']);current=load_catalog(ROOT,ROOT/'mcp_readonly/catalog.json',256*1024)
    old={r['id']:r for r in load_catalog(ROOT,OUT/'previous/mcp_readonly/catalog.json',256*1024)['documents']};now={r['id']:r for r in current['documents']}
    assert len(old)==460 and len(now)==461 and all(now[k]==v for k,v in old.items())
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    assert current['project']['authoritative_direction_id']==ID and current['project']['authoritative_delivery_id']=='v159-delivery'
    text=LOG.read_text(encoding='utf-8');assert 'Ran 23 tests' in text and text.rstrip().endswith('OK')
    paths=[Path(__file__).resolve(),OUT/'publication.json',OUT/'plan.md',LOG,ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py']
    save(OUT/'validation.json',dict(status='V169_resource_observation_stop_all460_old_records_and23_local_MCP_tests_verified',current_direction=ID,effective_documents=461,historic_records_preserved=460,MCP_tests_passed=23,official_calls=0,fits=0,permanent_updates=0,three_goals_complete=False,quality_acceptance=False,execution_authority=False,live_MCP_not_verified=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    print(json.dumps(dict(status='resource_observation_stop_verified',MCP_tests=23,official_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['publish','verify']);args=parser.parse_args();(publish if args.mode=='publish' else verify)()
