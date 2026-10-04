"""Build one small self-verifying Python update file; never bundles official data."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/"artifacts/v331_ready_20260912_r2"
FILES=["soc_v3_prepare.py","soc_v32_prepare.py","v32_finalize.py","v32_features.py","v32_split.py",
       "v32_model_diagnostics.py","run_v32_train.py","audit_v32_prepare.py",
       "test_soc_v3_prepare.py","test_soc_v32_prepare.py","test_v32_execution.py",
       "v331_prepare.py","run_v331_prepare.py","audit_v331_prepare.py",
       "v331_protocol.py","verify_v331_protocol.py","v331_common.py","run_v331_train.py",
       "review_v331_round.py","test_v331_input.py","test_v331_execution.py"]


def sha(data):return hashlib.sha256(data).hexdigest()


LOADER = r'''"""Verified v3.3.1 update. Official data remain in the existing platform folders."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import stat
import subprocess
import sys
import zipfile

ARCHIVE_SHA="__ARCHIVE_SHA__"
PAYLOAD=(
__PAYLOAD__
)


def install(destination):
    data=base64.b85decode(PAYLOAD.encode("ascii"))
    if hashlib.sha256(data).hexdigest()!=ARCHIVE_SHA:raise SystemExit("Update upload is incomplete or damaged")
    destination=Path(destination).resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos=z.infolist();names=[v.filename for v in infos]
        if len(names)!=len(set(names)) or z.testzip() is not None:raise SystemExit("Archive identity check failed")
        for entry in infos:
            name=entry.filename;relative=PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise SystemExit("Unsafe archive member")
            if stat.S_ISLNK(entry.external_attr>>16):raise SystemExit("Archive links are forbidden")
            target=(destination/name).resolve()
            if destination not in target.parents:raise SystemExit("Archive path escapes destination")
        destination.mkdir(parents=True,exist_ok=True)
        for entry in infos:
            target=destination/entry.filename;value=z.read(entry.filename)
            if target.exists():
                if not target.is_file() or target.read_bytes()!=value:
                    raise SystemExit("Existing runtime file differs; no files overwritten: "+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open("xb") as f:f.write(value)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument("--extract-only",action="store_true");p.add_argument("--install-dir")
    known,remaining=p.parse_known_args()
    dest=known.install_dir or str(Path(__file__).resolve().parent/("sf02_v331_"+ARCHIVE_SHA[:10]))
    root=install(dest)
    print("UPDATE_VERIFIED: "+str(root),flush=True)
    if known.extract_only:return
    for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[key]="16"
    os.environ["PYTHONDONTWRITEBYTECODE"]="1";os.environ["PYTHONUNBUFFERED"]="1"
    raise SystemExit(subprocess.call([sys.executable,"-u",str(root/"training/run_v331_pipeline.py")]+remaining))


if __name__=="__main__":main()
'''


def build():
    protocol=json.loads((PREP/"protocol.json").read_text(encoding="utf-8"))
    verification=json.loads((PREP/"protocol_verification.json").read_text(encoding="utf-8"))
    if not verification["all_checks_passed"] or not protocol["all_registered_tasks_supported"]:
        raise ValueError("Local complete prepared/protocol review required")
    content={}
    for name in FILES:
        path=ROOT/"training"/name
        data=path.read_bytes();ast.parse(data.decode("utf-8"),feature_version=(3,8))
        content["training/"+name]=data
    path=ROOT/"platform/run_v331_pipeline.py"
    ast.parse(path.read_text(encoding="utf-8"),feature_version=(3,8))
    content["training/run_v331_pipeline.py"]=path.read_bytes()
    fixture=ROOT/"evidence/2026-09-12/v32_actual_fixtures.json"
    content["evidence/2026-09-12/v32_actual_fixtures.json"]=fixture.read_bytes()
    expected={"content":verification["portable_content"],
              "roles":{t["name"]:t["role_content_sha256"] for t in protocol["tasks"]},
              "expected_tasks":9,"expected_classifier_fits":27,
              "scope":"Content reference verified locally; no trained SOC model in this update."}
    content["expected_content.json"]=(json.dumps(expected,indent=2)+"\n").encode()
    readme="""# v3.3.1 平台更新包

只上传外层 sf02_v331_update.py。复用 /root/work/sf02_data/train.parquet 和 /root/work/sf02_v32_round/prepared。
运行：conda run --no-capture-output -n cm-model python -u /root/work/sf02_v331_update.py

CPU 模式，最多 16 个库线程；不安装包、不使用昇腾卡、不修改官方标签。
程序自行校验、准备、分组、执行 9 个任务共 27 次拟合、复算结果并生成一个 ZIP。
再次运行同一命令会核验并复用完整任务；未完成任务重新执行，不覆盖旧尝试。
程序输出 RETURN_THIS_ZIP 后下载它；出错时优先带回 RETURN_FAILURE_ZIP。
自动化执行完成不等于迁移效果通过，需要审查返回结果。
"""
    content["README.md"]=readme.encode("utf-8")
    manifest={"version":"v331-update-1.0","official_train_sha256":"6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742",
       "parent_corpus_sha256":"95c79304fae8e60af997821b2ed279635e091ae74ffac3e694da29d3747d6c10",
       "files":[{"path":k,"bytes":len(v),"sha256":sha(v)} for k,v in sorted(content.items())],
       "model_trained":False,"external_data":False,"platform_execution_not_yet_performed":True}
    content["bundle_manifest.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(content.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,9,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;z.writestr(info,data)
    archive=buf.getvalue();encoded=base64.b85encode(archive).decode("ascii")
    literal="\n".join("    "+repr(encoded[i:i+100]) for i in range(0,len(encoded),100))
    loader=LOADER.replace("__ARCHIVE_SHA__",sha(archive)).replace("__PAYLOAD__",literal)
    ast.parse(loader,feature_version=(3,8))
    dest=ROOT/"platform/sf02_v331_update.py"
    dest.write_text(loader,encoding="utf-8")
    receipt={"path":str(dest),"bytes":dest.stat().st_size,"sha256":sha(dest.read_bytes()),
         "archive_sha256":sha(archive),"payload_files":len(content),"runtime_folder":"sf02_v331_"+sha(archive)[:10],
         "expected_tasks":9,"expected_classifier_fits":27,"data_included":False,"model_included":False,
         "local_prepared_content":verification["portable_content"],
         "package_source_sha256":sha(Path(__file__).read_bytes()),"platform_executed":False}
    (ROOT/"platform/v331_bundle_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":build()
