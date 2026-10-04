"""Preserve publication, repair only its incomplete-goal status wording."""
import argparse
import json
import shutil
from pathlib import Path
import v169_publish_post_cleanup_wait as prior

ROOT = prior.ROOT
OUT = prior.OUT
REPAIR = OUT / 'summary_wording_repair'
CATALOG = ROOT / 'mcp_readonly/catalog.json'
LOG = ROOT / 'artifacts/v169_post_cleanup_wait_MCP_tests_retry1_original_console_20261002.txt'


def repair():
    publication = prior.read(OUT / 'publication.json')
    prior.bindings(publication['source_sha256'])
    assert not REPAIR.exists()
    before = prior.read(CATALOG)
    after = json.loads(json.dumps(before))
    assert after['project']['current_summary'].count('三目标未解决') == 1
    after['project']['current_summary'] = after['project']['current_summary'].replace('三目标未解决', '三目标未完成')
    REPAIR.mkdir()
    shutil.copyfile(CATALOG, REPAIR / 'original_catalog.json')
    CATALOG.write_text(json.dumps(after, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    sources = dict(publication['source_sha256'])
    sources['mcp_readonly/catalog.json'] = prior.sha(CATALOG)
    for path in [Path(__file__).resolve(), OUT / 'publication.json', REPAIR / 'original_catalog.json', ROOT / 'artifacts/v169_post_cleanup_wait_MCP_tests_original_console_20261002.txt']:
        sources[path.relative_to(ROOT).as_posix()] = prior.sha(path)
    prior.save(REPAIR / 'receipt.json', dict(status='only_current_summary_incomplete_goal_wording_repaired', changed_fields=['project.current_summary'], old='三目标未解决', new='三目标未完成', official_calls=0, fits=0, permanent_updates=0, source_sha256=sources))
    print(json.dumps(dict(status='summary_wording_repaired', official_calls=0)))


def verify():
    receipt = prior.read(REPAIR / 'receipt.json')
    prior.bindings(receipt['source_sha256'])
    before = prior.read(REPAIR / 'original_catalog.json')
    after = prior.read(CATALOG)
    after['project']['current_summary'] = after['project']['current_summary'].replace('三目标未完成', '三目标未解决')
    assert after == before
    assert prior.sha(REPAIR / 'original_catalog.json') == prior.read(OUT / 'publication.json')['source_sha256']['mcp_readonly/catalog.json']
    current = prior.load_catalog(ROOT, CATALOG, 256 * 1024)
    old = {row['id']: row for row in prior.load_catalog(ROOT, OUT / 'previous/mcp_readonly/catalog.json', 256 * 1024)['documents']}
    now = {row['id']: row for row in current['documents']}
    assert len(old) == 461 and len(now) == 462 and all(now[key] == value for key, value in old.items())
    for row in now.values():
        assert prior.sha(ROOT / row['path']) == row['sha256'] and (ROOT / row['path']).stat().st_size <= 256 * 1024
    assert current['project']['authoritative_direction_id'] == prior.ID and current['project']['authoritative_delivery_id'] == 'v159-delivery'
    text = LOG.read_text(encoding='utf-8')
    assert 'Ran 23 tests' in text and text.rstrip().endswith('OK')
    sources = dict(receipt['source_sha256'])
    for path in [REPAIR / 'receipt.json', LOG]:
        sources[path.relative_to(ROOT).as_posix()] = prior.sha(path)
    prior.save(OUT / 'validation.json', dict(status='post_cleanup_wait_461_old_records_and23_local_tests_verified_after_summary_wording_repair', current_direction=prior.ID, effective_documents=462, historic_records_preserved=461, MCP_tests_passed=23, original_failed_tests_preserved=True, official_calls=0, fits=0, permanent_updates=0, three_goals_complete=False, quality_acceptance=False, execution_authority=False, live_MCP_not_verified=True, source_sha256=sources))
    print(json.dumps(dict(status='post_cleanup_wait_verified', MCP_tests=23, official_calls=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['repair', 'verify'])
    args = parser.parse_args()
    (repair if args.mode == 'repair' else verify)()
