"""Prepare a new bounded local solver without modifying historical executed code."""
from experiment_review import ROOT,sha

def main():
    source=ROOT/'training/v166_coverage_joint_restoration.py';target=ROOT/'training/v169_working_joint_restoration.py'
    if target.exists():raise FileExistsError(target)
    text=source.read_text(encoding='utf-8')
    if text.count('MAX_NORMALS=25')!=1:raise ValueError('Original solver source identity mismatch')
    revised=text.replace('MAX_NORMALS=25','MAX_NORMALS=64')
    target.write_bytes(revised.encode('utf-8'))
    print(dict(status='V169_new_solver_copy_prepared_no_execution',source_sha256=sha(source),target_sha256=sha(target),single_source_change='local simultaneous solver resource bound25_to64',old_source_unchanged=True,official_calls=0))

if __name__=='__main__':main()
