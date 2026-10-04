"""Publish actual sealed-but-unstarted terminal RAM refusal, no quality claim."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v169_registered_startup_wait_direction_20261002'
ID='v169-registered-startup-wait'
LOG=ROOT/'artifacts/v169_registered_startup_wait_MCP_tests_original_console_20261002.txt'

def save(path,value):
    assert not path.exists();path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def publish():
    assert not OUT.exists()
    previous=ROOT/'artifacts/v169_disk_headroom_current_status_20261002/record.json';check_bindings(read(previous)['source_sha256'])
    incident=ROOT/'artifacts/v169_registered_initial_resource_refusal_20261002/snapshot.json';failure=read(incident);check_bindings(failure['source_sha256'])
    assert failure['runner_observed_exit_code']==1 and not failure['official_zero_step_completed']
    assert failure['formal_registered_directory_contents']==['registration.json','run_seal.json']
    root_path=ROOT/'artifacts/v169_root_final_preseal_review_v2_20261002/review.json';root=read(root_path);check_bindings(root['source_sha256']);assert root['supports_physical_seal']
    registration=ROOT/'artifacts/v169_prior_pair_training/registration.json';reg=read(registration)
    assert all(reg[k]==failure[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates'])
    plan_path=ROOT/'training/review_policy/v169_prior_pair_execution_contract.json';plan=read(plan_path)
    assert plan['resources']['minimum_free_RAM_bytes']==5637144576 and plan['resources']['minimum_free_commit_bytes']==6710886400
    bundle_path=ROOT/'artifacts/v169_full_preseal_bundle_v6_20261002/bundle.json';bundle=read(bundle_path)
    old_plan=ROOT/'artifacts/v169_actual_resource_wait_direction_20261002/plan.md'
    body=('# V169当前实际：封存成功，首次启动在RAM检查拒绝\n\n'
          '根新最终独审通过18967项/35194537679bytes/93递归模块/18资格；v14只更换资源政策import，原数据、完整参数、逐类目标、20更新日程、预算、保护、数值与质量门槛全保持。资源v9原环境BLAS24/完整触页/原生SVD生命周期证据经根独立复算，资源v10与policy_v5登记物理RAM5.25GiB、可用commit6.25GiB，其余资源不变。原6GiB封存拒绝和BLAS4资格的混杂记录保留。\n\n'
          'sealer_v5实际exit0，registration绑定18971项。封存时RAM5874417664、commit10196668416、GPU7451181056、磁盘15053062144bytes过门槛。但随后v14第一次启动实际exit1，main首句require(initial)经v2的前CUDA物理RAM检查拒绝，未进入configure/ActualBackend、未建角色目录、未执行零步/官方前向/导数/拟合/更新。正式目录仅registration.json和run_seal.json；不能将registration或进程启动称为训练完成。\n\n'
          '失败后CUDA初始化快照为2026-10-02T07:53:22.194167+00:00：RAM4573757440，须5637144576，缺1063387136bytes；commit/GPU/磁盘通过。这是失败后的保存时点，不是故障瞬间值，也不能预测可用时刻。根允许0调用且未建角色目录的同一sealed入口在新真实全资源过门槛时启动重试，不追加fit；不得降低资源/科学门槛、改已封存源码或重复sealer。失败日志/封存全保留。已核查当前python/pythonw进程，未找到带SF02/v169命令的驻留helper，无对象被关闭。\n\n'
          '累计19178头/768完整导数，自V159以来9fits/170更新不变；最新实际训练V164、最新完整交付V159。当前仍解决第一问题，真实零步/配对训练/完整原行质量均未完成；三个总目标未完成，无模型晋升。\n\n').encode('utf-8')
    for path in [root_path,incident,registration]:body+=(f'\n## 实际原始来源 {path.relative_to(ROOT).as_posix()}；SHA256 `{sha(path)}`\n\n'.encode('utf-8')+path.read_bytes())
    body+=b'\n\n## Previous complete direction, original research, draft and all inherited limits\n\n'+old_plan.read_bytes()
    assert len(body)<=256*1024
    mutables=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    catalog=read(mutables[2]);before=load_catalog(ROOT,mutables[2],256*1024);history={r['id']:r for r in before['documents']}
    assert len(history)==459 and before['project']['authoritative_direction_id']=='v169-resource-wait'
    for row in history.values():assert sha(ROOT/row['path'])==row['sha256']
    catalog['documents'].insert(0,dict(id=ID,title='V169已封存但首次入口因RAM拒绝，0正式调用',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V169','实际启动拒绝','物理RAM','sealed']))
    project=catalog['project'];project['authoritative_direction_id']=ID
    project['current_summary']='V169资源专属v14/policy5/sealer5、candidate4/resource10/bundle6经根最终独审通过；实际封存18971项成功。随后首次v14启动在initial RAM门槛拒绝，未创建backend/role目录/真零步，0新官方头/导数/fit/更新。失败后快照缺1063387136bytes物理RAM，其他资源通过；保留sealed状态及失败证据，等待新真实全资源通过后仅按根允许的0调用入口重试。历史19178头/768导数、9fit/170更新不变；最新训练V164、完整交付V159，三目标未完成。'
    project['current_direction']=['保留同一已封存v14及全部门槛/预算/原行保护；核实无已结束驻留helper。真实RAM5.25GiB、commit6.25GiB、GPU512MiB、整轮磁盘门槛通过后按根允许重试同一0调用且无role目录的入口，不能重复sealer或改封存来源；随后真实point0及既定最多6×20训练和根完整独审。']
    project['known_limits']=['registration成功仅证明封存时资源通过；首次入口actual exit1、未进入backend，不证明零步、任何训练/分类收益或质量。','失败后快照RAM缺1063387136bytes是保存时点，须重读全部资源；不降低门槛，不关闭用户应用，不改已封存来源或重复封存。','0新官方调用/fit/永久更新；三个目标/完整三分类和独立来源泛化均未完成，459旧文档元数据及全部历史限制原文保留。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    OUT.mkdir()
    for path in mutables:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body)
    lead='当前实际：[V169已封存但首次入口因RAM拒绝](artifacts/v169_registered_startup_wait_direction_20261002/plan.md)。registration成功后v14首次启动actual exit1，未进入backend/真零步，0官方调用/fit/更新；失败后快照物理RAM缺约1GiB，等待真实资源恢复。原封存及失败日志保留，门槛5.25GiB RAM/6.25GiB commit不变。最新训练V164、完整交付V159，三目标未完成。\n\n'
    mutables[0].write_text(lead+mutables[0].read_text(encoding='utf-8'),encoding='utf-8')
    mutables[1].write_text(mutables[1].read_text(encoding='utf-8')+'\n\n## V169已封存首次启动RAM拒绝\n\n'+lead+'当前direction=v169-registered-startup-wait。根允许资源恢复后对同一sealed、0调用且无角色目录入口重试；不重复sealer，不改源或预算，不追加fit。\n',encoding='utf-8')
    mutables[2].write_bytes(encoded);mutables[3].write_text(mutables[3].read_text(encoding='utf-8').replace('v169-resource-wait',ID),encoding='utf-8')
    now={r['id']:r for r in load_catalog(ROOT,mutables[2],256*1024)['documents']}
    assert len(now)==460 and all(now[k]==v for k,v in history.items())
    paths=[Path(__file__).resolve(),previous,root_path,incident,registration,plan_path,bundle_path,old_plan,OUT/'plan.md',*mutables]
    save(OUT/'publication.json',dict(status='V169_actual_registered_startup_refusal_current_state_published',historic_records_preserved=459,effective_documents=460,current_direction=ID,official_calls=0,fits=0,permanent_updates=0,quality_acceptance=False,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    print(json.dumps(dict(status='registered_startup_wait_published',documents=460,official_calls=0)))

def verify():
    check_bindings(read(OUT/'publication.json')['source_sha256']);current=load_catalog(ROOT,ROOT/'mcp_readonly/catalog.json',256*1024)
    previous={r['id']:r for r in load_catalog(ROOT,OUT/'previous/mcp_readonly/catalog.json',256*1024)['documents']};now={r['id']:r for r in current['documents']}
    assert len(previous)==459 and len(now)==460 and all(now[k]==v for k,v in previous.items())
    for r in now.values():assert sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size<=256*1024
    assert current['project']['authoritative_direction_id']==ID and current['project']['authoritative_delivery_id']=='v159-delivery'
    log=LOG.read_text(encoding='utf-8');assert 'Ran 23 tests' in log and log.rstrip().endswith('OK')
    paths=[Path(__file__).resolve(),OUT/'publication.json',OUT/'plan.md',LOG,ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py']
    save(OUT/'validation.json',dict(status='V169_registered_startup_wait_all459_old_records_and23_local_MCP_tests_verified',current_direction=ID,effective_documents=460,historic_records_preserved=459,MCP_tests_passed=23,official_calls=0,fits=0,permanent_updates=0,quality_acceptance=False,three_goals_complete=False,live_MCP_not_verified=True,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    print(json.dumps(dict(status='registered_startup_wait_verified',MCP_tests=23,official_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['publish','verify']);args=parser.parse_args();(publish if args.mode=='publish' else verify)()
