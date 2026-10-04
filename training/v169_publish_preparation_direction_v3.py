"""Publish actual preparation while preserving every historical document."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v169_commit_ready_direction_20261002'
BUNDLE=ROOT/'artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json'
PLAN=ROOT/'docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md'
ID='v169-commit-ready'
LOG=ROOT/'artifacts/v169_commit_ready_direction_v3_MCP_tests_original_console_20261002.txt'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def publish():
    assert not OUT.exists();bundle=read(BUNDLE);assert bundle['entry_path']=='training/v169_prior_pair_training_entry_v13.py' and not bundle['execution_authority'];check_bindings(bundle['source_sha256'])
    previous=ROOT/'artifacts/v169_budget_scope_direction_20261002/plan.md';research=ROOT/'docs/V169_INCREMENTAL_RESEARCH_AND_TRAINING_REVIEW_20261002.md';findings=ROOT/'docs/V169_PRETRAINING_INDEPENDENT_FINDINGS_20261002.md';contract=ROOT/bundle['candidate_contract_path'];resource=ROOT/bundle['resource_review_path'];backend=ROOT/'artifacts/v169_actual_backend_synthetic_qualification_v4_20261002/qualification.json'
    assert sha(PLAN)=='6cb91a9b340f1b530ba14c224a4deb00de6769c06138da51e5c2b05d5c072f73' and sha(research)=='42c566eb2c7c09b1999ed57c2fb001ac27db0b1ab2d03582c583ee1a86bc563c'
    summary={k:v for k,v in bundle.items() if k!='source_sha256'};summary['complete_physical_manifest_path']=BUNDLE.relative_to(ROOT).as_posix();summary['complete_physical_manifest_sha256']=sha(BUNDLE)
    text=('\n\n## V169当前准备状态（本节覆盖旧附录的未实现状态）\n\n'
      '调用图已独立核算，完整参数、DataFrame接口、206条旧正确混类记录保护、实际CPU合成后台、终点同参数重放和全局故障停止已取得资格。最新入口v13；没有新增模型方法。原QP去掉一份完整矩阵副本；四次A/B完整64函数对照证明SVD输入、完整修正/位移及原单位验证逐位相同。物理存储/RAM工作区仍须根独立审查，执行预算尚未注册，根preseal/正式封存和真实零步尚未执行。最新实际训练V164、完整2056871行质量交付V159，训练分类未掌握，三个目标未完成。\n\n'
      '拟议新范围：56328头/特征、504固定目标导数、29184间隔导数、342QP、1146试探、6fit和最多120接受更新。历史19178头、768完整导数、9fit/170更新不重置；旧82174/2426只绑定各自旧试验。两臂20接受更新均含安全初始化，每次新方向最多8回溯/2纠错/64当前函数；完整永久保护不可缩减。非匹配20终点或技术故障为inconclusive，不选早期状态晋升。\n\n'
      '磁盘上界计三份接受保护掩码，按文件类型落实JSON/计数日志上限，并保留全部不同测量q/logq，未假设可复用。文件数上界已纠正为128000、目录17000，同一640MiB开销仍覆盖126522/16693最坏实际清单。整轮12,136,488,300 bytes，固定reserve2GiB，起始至少14,283,971,948 bytes；RAM6GiB、可用OS提交空间6.25GiB、GPU512MiB。保存快照磁盘/GPU满足而RAM不足；启动前重新核查。CPU合成资格代替了legacy跨角色保留oracle及官方frozen helper，仅证明后台接线；正式旧能力及训练收益须真实零步和完整原行验收。\n\n')
    body=PLAN.read_bytes()+text.encode('utf-8')
    for path in [research,findings,contract,resource,backend]:body+=(f'\n\n### 原始来源 {path.relative_to(ROOT).as_posix()}\n\nSHA256 `{sha(path)}`\n\n'.encode('utf-8')+path.read_bytes())
    body+=('\n\n## 完整依赖与资格清单摘要\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\n## 前一方向与全部历史限制完整保留\n\n').encode('utf-8')+previous.read_bytes();assert len(body)<=256*1024
    mutables=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutables[2]);old=load_catalog(ROOT,mutables[2],256*1024);history={r['id']:r for r in old['documents']};assert len(history)==457 and old['project']['authoritative_direction_id']=='v169-preparation'
    for row in history.values():assert sha(ROOT/row['path'])==row['sha256']
    catalog['documents'].insert(0,dict(id=ID,title='V169完整后台资格与执行前资源契约候选',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V169','后台资格','资源预算']))
    project=catalog['project'];project['authoritative_direction_id']=ID;project['current_summary']='V169调用图已独立核算，完整参数/实际CPU合成后台/206旧正确保护/同参数重放已资格；物理存储和RAM根审查、执行预算注册与封存尚未完成，0新官方调用/fit。历史19178头/768导数不重置，最新训练V164、完整交付V159，分类未掌握，三目标未完成。';project['current_direction']=['只检验B共享可学习先验系数，两臂相同原始人口、频次与逐行永久保护；v13完成后台资格和明确资源候选，待根独立资源/全依赖审查、注册封存后执行既定point0及配对训练。'];project['known_limits']=['CPU合成后台资格不证明官方旧能力或分类收益；原frozen helper及206接线已有独立gold审查，仍需真实零步。','调用图已独立核算，物理存储/RAM资格和执行预算尚未注册；物理RAM仍6GiB，增加有实测存活数组支持的6.25GiB可用提交空间；必须在CUDA初始化后实际满足门槛才能封存/启动。','所有旧限制和替代/失败源码完整保留；最新训练V164、完整交付V159，三个目标和完整质量未通过。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024;OUT.mkdir()
    for path in mutables:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body);save(OUT/'bundle_summary.json',summary)
    lead='当前方向：[V169完整后台资格与执行前资源候选](artifacts/v169_commit_ready_direction_20261002/plan.md)。调用图已独立核算，v13完整后台和206旧正确保护已资格；物理存储/RAM审查、执行预算注册与封存尚未完成。0新官方调用/fit，历史19178头/768导数保留，最新训练V164、完整交付V159，三目标未完成。\n\n'
    mutables[0].write_text(lead+mutables[0].read_text(encoding='utf-8'),encoding='utf-8');mutables[1].write_text(mutables[1].read_text(encoding='utf-8')+'\n\n## V169完整准备\n\n'+lead+'当前direction=v169-commit-ready，457旧记录全部保留；正式源/完整物理manifest见preseal bundle v5，合成资格v4，resource v7，execution candidate v3，seal v4。\n',encoding='utf-8');mutables[2].write_bytes(encoded);mutables[3].write_text(mutables[3].read_text(encoding='utf-8').replace('v169-preparation',ID),encoding='utf-8')
    now={r['id']:r for r in load_catalog(ROOT,mutables[2],256*1024)['documents']};assert len(now)==458 and all(now[k]==v for k,v in history.items())
    files=[Path(__file__).resolve(),BUNDLE,PLAN,previous,research,findings,contract,resource,backend,OUT/'plan.md',OUT/'bundle_summary.json',*mutables]+[ROOT/f'artifacts/{name}_20261002/qualification.json' for name in ['v169_memory_exact_solver_qualification','v169_cuda_host_buffer_qualification_v2','v169_physical_and_commit_policy_qualification']]+[ROOT/'artifacts/v169_saved_storage_filecount_review_20261002/review.json',ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json'];save(OUT/'publication.json',dict(status='V169_actual_preparation_published_without_new_execution_authority',historic_records_preserved=457,effective_documents=458,current_direction=ID,authoritative_delivery='v159-delivery',official_calls=0,new_fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files}));print(json.dumps(dict(status='V169_preparation_published',historic_records=457,documents=458,document_bytes=len(body))))

def verify():
    target=OUT/'validation.json';assert not target.exists();check_bindings(read(OUT/'publication.json')['source_sha256']);old={r['id']:r for r in load_catalog(ROOT,OUT/'previous/mcp_readonly/catalog.json',256*1024)['documents']};current=load_catalog(ROOT,ROOT/'mcp_readonly/catalog.json',256*1024);now={r['id']:r for r in current['documents']};assert len(old)==457 and len(now)==458 and all(now[k]==v for k,v in old.items())
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    assert current['project']['authoritative_direction_id']==ID and current['project']['authoritative_delivery_id']=='v159-delivery';text=LOG.read_text(encoding='utf-8');assert 'Ran 23 tests' in text and text.rstrip().endswith('OK')
    body=(OUT/'plan.md').read_bytes();assert body.startswith(PLAN.read_bytes()) and (ROOT/'artifacts/v169_budget_scope_direction_20261002/plan.md').read_bytes() in body
    files=[Path(__file__).resolve(),OUT/'publication.json',OUT/'plan.md',LOG,ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py'];save(target,dict(status='V169_preparation_current_direction_all457_historic_records_and23_local_MCP_tests_verified',MCP_tests_passed=23,effective_documents=458,historic_records_preserved=457,current_direction=ID,latest_actual_training='V164',latest_complete_delivery='V159',official_calls=0,fits=0,permanent_updates=0,execution_authority=False,quality_acceptance=False,three_goals_complete=False,live_MCP_not_verified=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files}));print(json.dumps(dict(status='V169_preparation_direction_verified',MCP_tests=23,official_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['publish','verify']);args=parser.parse_args();(publish if args.mode=='publish' else verify)()
