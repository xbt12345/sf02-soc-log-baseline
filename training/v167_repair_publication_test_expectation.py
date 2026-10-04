"""Preserve the failed publication test and update its stale status wording."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v167_publication_test_repair_20261002'

def main():
    assert not OUT.exists()
    publication=ROOT/'artifacts/v167_results_20261002/publication.json';check_bindings(read(publication)['source_sha256'])
    log=ROOT/'artifacts/v167_complete_results_MCP_tests_original_console_20261002.txt';text=log.read_text(encoding='utf-8');assert 'Ran 23 tests' in text and 'FAILED (failures=1)' in text and 'V164三fit/5更新' in text
    target=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';source=target.read_text(encoding='utf-8');old="self.assertIn('V164三fit/5更新', status.current_summary)\n        self.assertIn('分类未掌握', status.current_summary)";new="self.assertIn('V164', status.current_summary)\n        self.assertIn('三目标未完成', status.current_summary)";assert source.count(old)==1
    OUT.mkdir();shutil.copyfile(target,OUT/'test_readonly_mcp_before_repair.py');target.write_bytes(source.replace(old,new).encode('utf-8'))
    paths=[Path(__file__).resolve(),publication,log,OUT/'test_readonly_mcp_before_repair.py',target]
    (OUT/'repair.json').write_bytes((json.dumps(dict(status='V167_stale_status_wording_assertion_repaired_after_preserved_actual_failure',changed_only_assertion_wording=True,model_and_reader_unchanged=True,official_calls=0,original_failed_tests=1,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(status='V167_publication_test_wording_repaired',official_calls=0)))

if __name__=='__main__':main()
