"""Make current status honest while independent actual review is still pending."""
import json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v168_actual_pending_independent_review_20261002'

def main():
    assert not OUT.exists();trial=ROOT/'artifacts/v168_decision_floor_diagnostic_20261002';diag=read(trial/'role1/diagnostic.json');quality=ROOT/'artifacts/v168_saved_decision_floor_quality_review_20261002/review.json';geometry=ROOT/'artifacts/v168_saved_actual_floor_geometry_review_20261002/review.json'
    for p in [quality,geometry]:check_bindings(read(p)['source_sha256'])
    check_bindings(read(trial/'run_seal.json')['source_sha256']);assert diag['exception'] is None and diag['actual_finite_accepted'] and diag['all_parameters_restored'] and diag['new_fits']==diag['permanent_updates']==0 and diag['counts']['head_completed']==98 and diag['counts']['margin_completed']==50 and diag['actual_QP_solves']==diag['finite_proposals']==1
    old_direction=ROOT/'artifacts/v167_direction_20261002/validation.json';check_bindings(read(old_direction)['source_sha256']);mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json']];before=load_catalog(ROOT,mutable[2],256*1024);historic={r['id']:r for r in before['documents']};assert len(historic)==451 and before['project']['authoritative_direction_id']=='v168-plan' and before['project']['authoritative_delivery_id']=='v159-delivery'
    OUT.mkdir()
    for p in mutable:
        dest=OUT/'previous'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    lead='当前实际进度：V168唯一角色1诊断已执行，入口基础与复合有限保护均记录通过；两条S平手恢复正确、四条M候选修复保留，纯错误1952→1948。98头/50导数/1QP、0fit/0永久更新，已恢复V164，累计19178头/768导数。根独立实际审查尚待完成，尚无追加训练权限；三个目标未完成，最新真实训练V164、完整质量交付V159。当前登记方向：[V168根计划](docs/V167_ROOT_RESULTS_AND_V168_DECISION_AWARE_FLOOR_PLAN_20261002.md)，完整实际记录位于 `artifacts/v168_decision_floor_diagnostic_20261002`。\n\n'
    text=mutable[0].read_text(encoding='utf-8');assert text.startswith('当前方向：');mutable[0].write_bytes((lead+text.split('\n\n',1)[1]).encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V168实际完成、根独审待定\n\n'+lead+'入口过程exit0，末尾require复核完成；基础probe.json与最终v168_complete_probe_review.json分列，不把有限诊断称为新训练。角色0/2冻结旧引用，新增官方调用仅角色1；无第二修正或fit权限。\n').encode('utf-8'))
    catalog=read(mutable[2]);catalog['project']['current_summary']='V168已执行，入口记录通过，根独审待定；98头/50导数/1QP，0fit/更新，全恢复V164；累计19178头/768导数。分类未掌握，三目标未完成，交付V159。';catalog['project']['current_direction']=['等待根独立实际验收并发布完整结果；本轮不追加候选或fit。'];encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024;mutable[2].write_bytes(encoded);after=load_catalog(ROOT,mutable[2],256*1024);assert {r['id']:r for r in after['documents']}==historic
    sources=[Path(__file__).resolve(),quality,geometry,trial/'role1/diagnostic.json',trial/'registration.json',trial/'run_seal.json',ROOT/'artifacts/v168_decision_floor_diagnostic_original_console_20261002.txt',old_direction,*mutable]
    (OUT/'record.json').write_bytes((json.dumps(dict(status='V168_actual_execution_restoration_and_worker_saved_reviews_complete_root_independent_actual_review_pending',actual_heads=98,actual_derivatives=50,actual_QP_solves=1,actual_candidates=1,cumulative_heads=19178,cumulative_all_complete_derivatives=768,new_fits=0,permanent_updates=0,all_existing_document_metadata_preserved=451,actual_entry_process_exit_code_observed=0,execution_completion_tool_chunk='9d0179',root_actual_review_completed=False,new_training_permission=False,quality_acceptance=False,latest_actual_training='V164',latest_complete_quality_delivery='V159',official_calls_by_this_record=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources}),ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status='V168_actual_pending_independent_review_recorded',preserved_records=451,official_calls=0)))

if __name__=='__main__':main()
