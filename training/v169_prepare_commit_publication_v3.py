"""Publish the new exact kernel and separate commit prerequisite after stable bundle."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_publish_preparation_direction_v2.py';target=ROOT/'training/v169_publish_preparation_direction_v3.py';assert not target.exists();s=source.read_text(encoding='utf-8')
    s=s.replace('v169_preparation_direction_20261002','v169_commit_ready_direction_20261002').replace('v169_full_preseal_bundle_v3_20261002','v169_full_preseal_bundle_v4_20261002').replace("ID='v169-preparation'","ID='v169-commit-ready'").replace('v169_preparation_direction_v2_MCP_tests_original_console_20261002.txt','v169_commit_ready_direction_v3_MCP_tests_original_console_20261002.txt').replace('v169_prior_pair_training_entry_v12.py','v169_prior_pair_training_entry_v13.py').replace('v169_actual_backend_synthetic_qualification_v3_20261002','v169_actual_backend_synthetic_qualification_v4_20261002')
    s=s.replace("len(history)==456 and old['project']['authoritative_direction_id']=='v169-plan-v2'","len(history)==457 and old['project']['authoritative_direction_id']=='v169-preparation'").replace('len(now)==457','len(now)==458').replace('len(old)==456','len(old)==457').replace('historic_records_preserved=456','historic_records_preserved=457').replace('effective_documents=457','effective_documents=458').replace('historic_records=456,documents=457','historic_records=457,documents=458').replace(".replace('v169-plan-v2',ID)",".replace('v169-preparation',ID)")
    s=s.replace('入口v12','入口v13').replace('最新入口v12','最新入口v13').replace('v12完整后台','v13完整后台').replace('v12完整','v13完整').replace('v12完成','v13完成').replace('direction=v169-preparation','direction=v169-commit-ready').replace('456旧记录','457旧记录').replace('全456_historic','全457_historic').replace('all456_historic','all457_historic').replace('preseal bundle v3，合成资格v3，resource v5，execution candidate v2','preseal bundle v4，合成资格v4，resource v7，execution candidate v3，seal v3')
    s=s.replace('RAM6GiB、GPU512MiB。','RAM6GiB、可用OS提交空间6.25GiB、GPU512MiB。')
    s=s.replace('没有新增模型方法。','没有新增模型方法。原QP去掉一份完整矩阵副本；四次A/B完整64函数对照证明SVD输入、完整修正/位移及原单位验证逐位相同。')
    s=s.replace('并保留全部不同测量q/logq，未假设可复用。','并保留全部不同测量q/logq，未假设可复用。文件数上界已纠正为128000、目录17000，同一640MiB开销仍覆盖126522/16693最坏实际清单。')
    s=s.replace('当前RAM快照低于6GiB，必须实际满足前提后启动。','物理RAM仍6GiB，增加有实测存活数组支持的6.25GiB可用提交空间；必须在CUDA初始化后实际满足门槛才能封存/启动。')
    # Bind new measured reports in the publication even where the compact body
    # describes their results instead of embedding another large raw ledger.
    old="files=[Path(__file__).resolve(),BUNDLE,PLAN,previous,research,findings,contract,resource,backend,OUT/'plan.md',OUT/'bundle_summary.json',*mutables]"
    new="files=[Path(__file__).resolve(),BUNDLE,PLAN,previous,research,findings,contract,resource,backend,OUT/'plan.md',OUT/'bundle_summary.json',*mutables]+[ROOT/f'artifacts/{name}_20261002/'+Path('qualification.json') if False else ROOT/f'artifacts/{name}_20261002/qualification.json' for name in ['v169_memory_exact_solver_qualification','v169_cuda_host_buffer_qualification_v2','v169_physical_and_commit_policy_qualification']]+[ROOT/'artifacts/v169_saved_storage_filecount_review_20261002/review.json',ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json']"
    assert old in s;s=s.replace(old,new)
    # Simpler explicit path expression, no dead conditional in the real source.
    s=s.replace("ROOT/f'artifacts/{name}_20261002/'+Path('qualification.json') if False else ROOT/f'artifacts/{name}_20261002/qualification.json'","ROOT/f'artifacts/{name}_20261002/qualification.json'")
    compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print('Prepared commit-ready publication v3')

if __name__=='__main__':main()
