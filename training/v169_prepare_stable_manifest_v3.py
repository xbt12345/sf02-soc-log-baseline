"""Exclude live self log from immutable dependency manifests."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_prepare_full_preseal_bundle_v2.py';target=ROOT/'training/v169_prepare_full_preseal_bundle_v3.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_full_preseal_bundle_v2_20261002','v169_full_preseal_bundle_v3_20261002')
    old="    forbidden=[ROOT/'README.md'"
    new="    files.discard(ROOT/'artifacts/v169_prepare_full_preseal_bundle_v3_original_console_20261002.txt')\n    forbidden=[ROOT/'README.md'"
    assert old in s;s=s.replace(old,new);target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_seal_prior_pair_training.py';target=ROOT/'training/v169_seal_prior_pair_training_v2.py';assert not target.exists();s=source.read_text(encoding='utf-8').replace('v169_full_preseal_bundle_v2_20261002','v169_full_preseal_bundle_v3_20261002');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_publish_preparation_direction.py';target=ROOT/'training/v169_publish_preparation_direction_v2.py';assert not target.exists();s=source.read_text(encoding='utf-8').replace('v169_full_preseal_bundle_v2_20261002','v169_full_preseal_bundle_v3_20261002').replace('v169_preparation_direction_MCP_tests_original_console_20261002.txt','v169_preparation_direction_v2_MCP_tests_original_console_20261002.txt').replace('preseal bundle v2','preseal bundle v3');target.write_text(s,encoding='utf-8');print('Prepared stable manifest v3, sealer v2 and publication v2')

if __name__=='__main__':main()
