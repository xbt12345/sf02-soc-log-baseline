"""Record root-owned lossless disk relief without changing training authority."""
import json
import shutil
import sys
from pathlib import Path

from experiment_review import ROOT, read, sha, check_bindings
sys.path.insert(0, str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT = ROOT / 'artifacts/v169_disk_headroom_current_status_20261002'
REPORT = ROOT / 'artifacts/v169_root_disk_headroom_compression_20261002/review.json'
RECEIPTS = REPORT.parent / 'file_receipts.jsonl'

def main():
    assert not OUT.exists()
    previous = ROOT / 'artifacts/v169_actual_resource_wait_direction_20261002'
    check_bindings(read(previous / 'validation.json')['source_sha256'])
    report = read(REPORT)
    check_bindings(report['source_sha256'])
    bundle_path = ROOT / report['source_bundle']
    assert sha(bundle_path) == report['source_bundle_sha256']
    bindings = read(bundle_path)['source_sha256']
    rows = [json.loads(line) for line in RECEIPTS.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == report['files'] == 123
    assert len({r['path'] for r in rows}) == 123
    for row in rows:
        bound = bindings.get(row['path'], bindings.get(str(ROOT / row['path'])))
        assert bound is not None
        assert row['sha256_before'] == row['sha256_after'] == bound
        assert row['logical_bytes_exact'] and row['NTFS_compressed_attribute']
        assert row['compact_exit_code'] == 0
    assert sum(r['allocation_before'] - r['allocation_after'] for r in rows) == report['allocation_reduction_bytes'] == 1022014080
    assert report['headroom_above_complete_run_start_bytes'] == report['free_disk_after'] - report['complete_run_start_minimum'] == 1075016340
    assert report['all_logical_SHA256_preserved'] and report['free_RAM_not_fixed_by_this_action']
    assert report['no_files_deleted'] and not report['execution_authority']
    for key in ['official_heads', 'official_features', 'official_derivatives', 'fits', 'permanent_updates']:
        assert report[key] == 0
    assert not (ROOT / 'artifacts/v169_prior_pair_training').exists()
    assert not (ROOT / 'training/review_policy/v169_prior_pair_execution_contract.json').exists()

    mutables = [ROOT / p for p in ['README.md', 'HANDOFF.md', 'mcp_readonly/catalog.json']]
    before = load_catalog(ROOT, mutables[2], 256 * 1024)
    history = {r['id']: r for r in before['documents']}
    assert len(history) == 459
    assert before['project']['authoritative_direction_id'] == 'v169-resource-wait'
    OUT.mkdir()
    for path in mutables:
        target = OUT / 'previous' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    note = ('# V169资源等待：磁盘余量实际增加\n\n'
            '根对123个已绑定历史gradient.npy执行可逆NTFS压缩，逐文件前后SHA均与v5整包绑定一致；未删除文件、未改逻辑字节、输入、模型参数或训练契约。释放1,022,014,080 bytes，压缩完成时磁盘可用15,358,988,288 bytes，比整轮启动门槛14,283,971,948多1,075,016,340 bytes。\n\n'
            '原始凭据：`artifacts/v169_root_disk_headroom_compression_20261002/review.json`和`file_receipts.jsonl`。本记录核对123条收据与整包期望SHA及实际分配差额，不重复执行根侧压缩或训练。\n\n'
            '磁盘变化没有解决物理RAM缺口。CUDA初始化后2026-10-02T07:10:32.033000+00:00只读复核：RAM5,649,248,256 bytes，须6,442,450,944，缺793,202,688 bytes；commit9,888,182,272、GPU7,451,181,056、磁盘15,355,957,248均达各自门槛。这些值只代表保存时点，启动前须实际复核全部门槛。\n\n'
            '正式OUT和PLAN仍不存在；不因磁盘变化重试封存，不降低RAM6GiB/commit6.25GiB/GPU512MiB/完整磁盘门槛，不关闭用户应用。待资源实际满足后，才可按既有根审查、同一封存器v4和v13入口推进。当前方向v169-resource-wait不变，459既有文档记录与全部旧证据保留。\n\n'
            '本次0官方头/特征/导数/fit/永久更新；最新训练V164、完整交付V159，三目标和实际训练验收均未完成。\n')
    (OUT / 'status.md').write_text(note, encoding='utf-8')
    link = '[V169磁盘余量实测补充](artifacts/v169_disk_headroom_current_status_20261002/status.md)'
    lead = f'资源补充：{link}。123份历史梯度无损NTFS压缩释放约0.95GiB，完成时整轮磁盘余量约1.00GiB；RAM仍未达6GiB，未重试封存，0新官方调用/fit。原设计与封存绑定有效，当前方向v169-resource-wait。\n\n'
    mutables[0].write_text(lead + mutables[0].read_text(encoding='utf-8'), encoding='utf-8')
    mutables[1].write_text(mutables[1].read_text(encoding='utf-8') + '\n\n## V169磁盘空间补充\n\n' + lead, encoding='utf-8')
    catalog = read(mutables[2])
    catalog['project']['current_summary'] = 'V169最终设计与根独立审查通过；实际封存因RAM不足拒绝，未写OUT/PLAN，0新官方调用/fit。根对123份历史梯度做NTFS无损压缩，SHA与v5绑定一致，释放1022014080bytes，完成时磁盘比整轮门槛多1075016340bytes；RAM仍待外部实际变化。保持全部资源门槛与v13/封存器v4，不盲重试。最新训练V164、完整交付V159，三目标未完成。'
    catalog['project']['current_direction'].append('实际磁盘空间补充见artifacts/v169_disk_headroom_current_status_20261002/status.md及根压缩原始收据；未改变训练契约、原始数据或正式执行权限。')
    mutables[2].write_text(json.dumps(catalog, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    after = load_catalog(ROOT, mutables[2], 256 * 1024)
    assert {r['id']: r for r in after['documents']} == history
    assert after['project']['authoritative_direction_id'] == 'v169-resource-wait'
    assert after['project']['authoritative_delivery_id'] == 'v159-delivery'
    paths = [Path(__file__).resolve(), REPORT, RECEIPTS, bundle_path, previous / 'validation.json', OUT / 'status.md', *mutables]
    receipt = dict(status='root_lossless_disk_relief_recorded_in_current_state',
                   historical_records_preserved=459, current_direction='v169-resource-wait',
                   root_receipts_checked_against_bundle=123, allocation_reduction_bytes=1022014080,
                   official_calls=0, fits=0, permanent_updates=0, execution_authority=False,
                   quality_acceptance=False, three_goals_complete=False,
                   source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    (OUT / 'record.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'}))

if __name__ == '__main__':
    main()
