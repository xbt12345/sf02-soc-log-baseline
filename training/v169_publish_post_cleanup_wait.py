"""Publish actual completed background cleanup and unchanged resource wait."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT = ROOT / 'artifacts/v169_post_cleanup_resource_wait_direction_20261002'
PREVIOUS = ROOT / 'artifacts/v169_resource_observation_stop_direction_20261002'
CLEANUP = ROOT / 'artifacts/v169_background_cleanup_20261002'
ID = 'v169-post-cleanup-resource-wait'
OLD_ID = 'v169-resource-observation-wait'
LOG = ROOT / 'artifacts/v169_post_cleanup_wait_MCP_tests_original_console_20261002.txt'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bindings(value):
    for key, expected in value.items():
        assert sha(ROOT / key) == expected, key


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def publish():
    assert not OUT.exists()
    bindings(read(PREVIOUS / 'validation.json')['source_sha256'])
    summary = read(CLEANUP / 'summary.json')
    resource = read(CLEANUP / 'post_cleanup_resources.json')
    assert summary['edge_preserved_by_explicit_reply'] is True
    assert summary['training_started'] is False and summary['training_model_calls'] == 0
    assert summary['physical_gate_passed'] is False
    assert summary['physical_required_bytes'] == resource['physical_required_bytes'] == 5637144576
    assert resource['commit_required_bytes'] == 6710886400
    assert resource['disk_required_bytes'] == 14283971948
    assert resource['commit_gate_passed'] and resource['disk_gate_passed']
    assert sorted(p.name for p in (ROOT / 'artifacts/v169_prior_pair_training').iterdir()) == ['registration.json', 'run_seal.json']
    mutables = [ROOT / p for p in ['README.md', 'HANDOFF.md', 'mcp_readonly/catalog.json', 'mcp_readonly/tests/test_readonly_mcp.py']]
    catalog = load_catalog(ROOT, mutables[2], 256 * 1024)
    old = {row['id']: row for row in catalog['documents']}
    assert len(old) == 461 and catalog['project']['authoritative_direction_id'] == OLD_ID
    for row in old.values():
        assert sha(ROOT / row['path']) == row['sha256']
    cleanup_sources = sorted(CLEANUP.glob('*.json'))
    assert len(cleanup_sources) == 6
    gap = summary['physical_required_bytes'] - summary['final_cim_free_physical_bytes']
    lead = ('当前实际：[V169后台清理后仍等待物理RAM](artifacts/v169_post_cleanup_resource_wait_direction_20261002/plan.md)。'
            '根已完成用户授权清理，Edge按明确回复保留；物理RAM仍不足，正式入口不重试。'
            'V169新增官方调用/fit/更新均0，真实零步未完成；最新训练V164、完整交付V159，三个目标未解决。\n\n')
    body = ('# V169后台清理完成，物理RAM仍不足，等待真实资源变化\n\n'
            '主聊天已完成用户授权的后台清理，并要求本线程保持原封存、预算和学习因子，不启动训练、不重复正式入口或资源诊断、不下调门槛；等待真实资源变化。用户明确要求保留Edge，已保留。\n\n'
            '根记录39次直接停止动作，父进程退出也结束部分子进程。对象为确认不用的MCP文件/内存/Playwright、Codex Security工具后台、Windows Widgets/Phone Link、Java更新、Intel DSA/图形设置托盘、SOLIDWORKS后台下载及NVIDIA浮层；活动Codex、CUA/网络代理、系统驱动和安全组件保留，启动配置未修改。三个可选系统服务Stop-Service因权限被拒，部分工具/浮层辅助进程被宿主拉起。这不是全部后台均已清除的证明。\n\n'
            '清理期间物理可用RAM约3.4–4.0GiB，仍不足固定5.25GiB。21:46:18的GlobalMemoryStatusEx实测物理3990192128、提交余量11996278784bytes，磁盘20173533184bytes：物理不通过，提交和磁盘通过。根同时报告GPU可用7956MiB；这些是OS/驱动快照，不是实际入口通过或CUDA进程内零步资格。\n\n'
            f"根最终CIM快照（{summary['timestamp']}）物理可用{summary['final_cim_free_physical_bytes']}bytes（{summary['final_cim_free_physical_bytes']/1024**3:.3f}GiB），距固定门槛缺{gap}bytes（{gap/1024**3:.3f}GiB）。快照不能预测未来就绪；真实变化后仍必须核完整源和正式入口全部即时门槛。\n\n"
            '恢复条件不变：物理可用RAM>=5637144576bytes、可用commit>=6710886400bytes、GPU>=536870912bytes、完整轮起始磁盘>=14283971948bytes，并保持稳定。只有真实恢复后按既有授权核源/即时门槛并继续同一sealed入口；本次等待不重新sealer、不改任何正式训练源或科学约束、不追加fit预算。起始0后台/0调用资源失败不算正式fit；进入过正式fit后的失败仍遵守原全局停止约束。\n\n'
            'V169目录仍仅registration.json和run_seal.json，没有role/backend；两次原始入口失败、纠正的同进程诊断和清理记录全部保留。新增官方heads/features/derivatives/fits/updates均0。历史19178头/768完整导数、V159以来9fits/170更新不变。最新实际训练V164、完整质量交付V159；三个总目标、完整三分类与独立来源泛化均未完成，无模型晋升。\n\n')
    for path in cleanup_sources:
        body += f'## 清理原始证据 {path.relative_to(ROOT).as_posix()}\n\nSHA256 `{sha(path)}`\n\n' + path.read_text(encoding='utf-8-sig') + '\n\n'
    body += '## Previous complete direction and all inherited scientific constraints\n\n' + (PREVIOUS / 'plan.md').read_text(encoding='utf-8')
    body_bytes = body.encode('utf-8')
    assert len(body_bytes) <= 256 * 1024
    raw = read(mutables[2])
    raw['documents'].insert(0, dict(id=ID, title='V169后台清理后物理RAM仍不足，0正式调用', path=(OUT / 'plan.md').relative_to(ROOT).as_posix(), sha256=hashlib.sha256(body_bytes).hexdigest(), category='review_evidence', keywords=['V169', '后台清理', 'RAM', 'Edge保留', '等待']))
    project = raw['project']
    project['authoritative_direction_id'] = ID
    project['current_summary'] = '用户授权后台清理已完成，根39次直接停止，Edge按明确回复保留；可用RAM约3.4–4.0GiB，最终CIM3.492GiB，仍不足固定5.25GiB；提交/磁盘实测达标，根GPU快照足够。部分工具重启、三个可选服务停止被权限拒绝。等待真实资源变化，不重复正式入口/资源诊断、不降门槛、不改已封存学习因子。V169无role目录，0新增官方调用/fit/更新，真实零步未完成；最新训练V164、完整交付V159，三目标未解决。'
    project['current_direction'] = ['保持原封存与全部预算/科学限制，等待物理RAM真实恢复并稳定；恢复后核完整源和正式入口即时RAM5.25GiB/commit6.25GiB/GPU512MiB/磁盘14283971948bytes。当前不启动训练，不重复入口或诊断，不降低门槛；Edge按用户明确回复保留。']
    project['known_limits'] = ['清理完成未证明资源恢复；物理RAM仍不足，OS/GPU快照不证明正式入口、真实零步或训练通过。', '三个可选服务停止被权限拒绝，部分工具由宿主重启；未改启动项或驱动/系统安全/活动连接。', 'V169仅注册封存，0新增官方调用/fit/更新；三个目标、完整三分类及独立来源质量均未解决，完整交付仍V159。']
    encoded = (json.dumps(raw, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    assert len(encoded) <= 256 * 1024
    OUT.mkdir()
    for path in mutables:
        target = OUT / 'previous' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (OUT / 'plan.md').write_bytes(body_bytes)
    mutables[0].write_text(lead + mutables[0].read_text(encoding='utf-8'), encoding='utf-8')
    mutables[1].write_text(mutables[1].read_text(encoding='utf-8') + '\n\n## V169后台清理完成但物理RAM仍不足\n\n' + lead, encoding='utf-8')
    mutables[2].write_bytes(encoded)
    mutables[3].write_text(mutables[3].read_text(encoding='utf-8').replace(OLD_ID, ID), encoding='utf-8')
    current = {row['id']: row for row in load_catalog(ROOT, mutables[2], 256 * 1024)['documents']}
    assert len(current) == 462 and all(current[key] == value for key, value in old.items())
    sources = [Path(__file__).resolve(), PREVIOUS / 'validation.json', PREVIOUS / 'plan.md', *cleanup_sources, OUT / 'plan.md', *mutables]
    save(OUT / 'publication.json', dict(status='post_cleanup_physical_RAM_wait_published', current_direction=ID, effective_documents=462, historic_records_preserved=461, official_calls=0, fits=0, permanent_updates=0, execution_authority=False, quality_acceptance=False, three_goals_complete=False, waiting_real_resource_change=True, source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in sources}))
    print(json.dumps(dict(status='post_cleanup_wait_published', documents=462, old_preserved=461, official_calls=0)))


def verify():
    bindings(read(OUT / 'publication.json')['source_sha256'])
    catalog = load_catalog(ROOT, ROOT / 'mcp_readonly/catalog.json', 256 * 1024)
    old = {row['id']: row for row in load_catalog(ROOT, OUT / 'previous/mcp_readonly/catalog.json', 256 * 1024)['documents']}
    current = {row['id']: row for row in catalog['documents']}
    assert len(old) == 461 and len(current) == 462 and all(current[key] == value for key, value in old.items())
    for row in current.values():
        assert sha(ROOT / row['path']) == row['sha256'] and (ROOT / row['path']).stat().st_size <= 256 * 1024
    assert catalog['project']['authoritative_direction_id'] == ID and catalog['project']['authoritative_delivery_id'] == 'v159-delivery'
    result = LOG.read_text(encoding='utf-8')
    assert 'Ran 23 tests' in result and result.rstrip().endswith('OK')
    sources = [Path(__file__).resolve(), OUT / 'publication.json', OUT / 'plan.md', LOG, *[ROOT / p for p in ['README.md', 'HANDOFF.md', 'mcp_readonly/catalog.json', 'mcp_readonly/tests/test_readonly_mcp.py']]]
    save(OUT / 'validation.json', dict(status='post_cleanup_wait_461_old_records_and23_local_tests_verified', current_direction=ID, effective_documents=462, historic_records_preserved=461, MCP_tests_passed=23, official_calls=0, fits=0, permanent_updates=0, three_goals_complete=False, quality_acceptance=False, execution_authority=False, live_MCP_not_verified=True, source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in sources}))
    print(json.dumps(dict(status='post_cleanup_wait_verified', MCP_tests=23, official_calls=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['publish', 'verify'])
    arguments = parser.parse_args()
    (publish if arguments.mode == 'publish' else verify)()
