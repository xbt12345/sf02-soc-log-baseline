"""One entry: verify, prepare, audit, freeze roles, train, replay, package results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

# Set before importing NumPy/BLAS, including on manual direct invocation.
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[key]="16"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["PYTHONUNBUFFERED"]="1"


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8388608),b""):h.update(b)
    return h.hexdigest()
def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def checked_process(argv,work):
    print("STAGE: "+" ".join(str(a) for a in argv[2:]),flush=True)
    with (work/"execution.log").open("a",encoding="utf-8") as log:
        proc=subprocess.Popen([str(x) for x in argv],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding="utf-8",errors="replace",bufsize=1)
        for line in proc.stdout:
            print(line,end="",flush=True);log.write(line);log.flush()
        code=proc.wait()
    if code:raise RuntimeError("Stage failed (exit {}): {}".format(code," ".join(map(str,argv[2:]))))


def verify_package(package):
    manifest=load(package/"bundle_manifest.json")
    for entry in manifest["files"]:
        path=(package/entry["path"]).resolve()
        if package.resolve() not in path.parents:raise ValueError("Unsafe package path")
        if not path.is_file() or path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]:
            raise ValueError("Missing or damaged package file: "+entry["path"])
    return manifest


def make_zip(work,experiment,prepared,package,success):
    stamp=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    dest=work/(("v331_results_" if success else "v331_failure_")+stamp+".zip")
    while dest.exists():dest=dest.with_name(dest.stem+"_new.zip")
    with zipfile.ZipFile(dest,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.write(package/"bundle_manifest.json","runtime/bundle_manifest.json")
        for p in (package/"training").glob("*.py"):z.write(p,"runtime/training/"+p.name)
        if (work/"execution.log").exists():z.write(work/"execution.log","execution.log")
        for p in work.glob("failure_*.json"):z.write(p,p.name)
        if prepared.exists():
            for p in prepared.iterdir():
                if p.suffix==".json" or p.name=="protocol_manifest.parquet":
                    z.write(p,"prepared_evidence/"+p.name)
        if experiment.exists():
            for p in experiment.iterdir():
                if p.is_file():z.write(p,"experiment/"+p.name)
            for task in (experiment/"tasks").glob("*") if (experiment/"tasks").exists() else []:
                receipt=task/"complete.json"
                if receipt.exists():
                    d=load(receipt);z.write(receipt,"experiment/tasks/"+task.name+"/complete.json")
                    attempt=task/d["attempt"]
                    for name in d["files"]:
                        z.write(attempt/name,"experiment/tasks/"+task.name+"/"+d["attempt"]+"/"+name)
                elif not success:
                    for p in task.rglob("*.json"):z.write(p,"experiment/tasks/"+str(p.relative_to(task.parent)).replace("\\","/"))
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise ValueError("Output ZIP failed CRC")
    return dest


def main(args):
    package=Path(__file__).resolve().parents[1];training=package/"training"
    base=Path(args.base_dir).resolve();train=Path(args.train).resolve()
    work=Path(args.work_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    prepared=Path(args.prepared_dir).resolve() if args.prepared_dir else work/"prepared"
    experiment=work/"experiment"
    manifest=verify_package(package)
    expected=load(package/"expected_content.json")
    if not train.is_file() or sha(train)!=manifest["official_train_sha256"]:
        raise ValueError("Official train missing or damaged: "+str(train))
    parent=base/"prepared"
    if not (parent/"prepared_corpus.parquet").is_file() or sha(parent/"prepared_corpus.parquet")!=manifest["parent_corpus_sha256"]:
        raise ValueError("Existing v3.2 prepared cache missing or damaged: "+str(parent))
    print("Verified update and official data. CPU only; no packages installed; no labels changed.",flush=True)
    def stage(name,*argv):
        checked_process([sys.executable,"-u",training/name,*argv],work)
    try:
        print("SOFTWARE_CHECKS_ONLY: regression and synthetic execution tests; these are not SOC model scores.",flush=True)
        checked_process([sys.executable,"-m","unittest","discover","-s",training,"-p","test*32*.py"],work)
        checked_process([sys.executable,"-m","unittest","discover","-s",training,"-p","test_v331*.py"],work)
        if not (prepared/"result.json").exists():
            if prepared.exists():
                # Never overwrite a failed partial preparation. Use another attempt directory.
                i=2
                while (work/("prepared_attempt_"+str(i))).exists():i+=1
                prepared=work/("prepared_attempt_"+str(i))
            stage("run_v331_prepare.py","--train",train,"--parent-run",parent,"--output-dir",prepared)
        result=load(prepared/"result.json")
        for name,digest in result["source_sha256"].items():
            if sha(training/name)!=digest:raise ValueError("Existing preparation uses different code: "+name)
        if not (prepared/"preparation_audit.json").exists():
            stage("audit_v331_prepare.py","--train",train,"--parent-run",parent,"--run-dir",prepared)
        else:
            audit=load(prepared/"preparation_audit.json")
            if not audit["all_checks_passed"] or audit["script_sha256"]!=sha(training/"audit_v331_prepare.py"):
                raise ValueError("Existing preparation audit is obsolete")
        if not (prepared/"protocol.json").exists():
            stage("v331_protocol.py","--run-dir",prepared)
        else:
            p=load(prepared/"protocol.json")
            if p["protocol_source_sha256"]!=sha(training/"v331_protocol.py") or p["allocation_source_sha256"]!=sha(training/"v32_split.py"):
                raise ValueError("Existing protocol uses different allocation code")
        stage("verify_v331_protocol.py","--run-dir",prepared)
        actual=load(prepared/"portable_content.json");protocol=load(prepared/"protocol.json")
        for key in ["rows","corpus_content_sha256","group_content_sha256"]:
            if actual[key]!=expected["content"][key]:raise ValueError("Prepared portable content differs from local full audit: "+key)
        roles={t["name"]:t["role_content_sha256"] for t in protocol["tasks"]}
        if roles!=expected["roles"]:raise ValueError("Platform role allocation differs from reviewed local allocation")
        save(work/"active_prepared.json",{"path":str(prepared),"content":actual})
        if args.prepare_only:
            print("PREPARATION_AND_PROTOCOL_VERIFIED; model training not started.",flush=True);return
        stage("run_v331_train.py","--train",train,"--run-dir",prepared,"--output-dir",experiment)
        stage("review_v331_round.py","--run-dir",prepared,"--experiment-dir",experiment)
        review=load(experiment/"independent_review.json")
        if not review["all_execution_checks_passed"] or review["completed_tasks"]!=len(expected["roles"]):
            raise ValueError("Not all registered tasks completed and replayed")
        if (work/"delivery.json").exists():
            old=load(work/"delivery.json");dest=work/old["archive"]
            if dest.is_file() and sha(dest)==old["sha256"]:
                print("RETURN_THIS_ZIP: "+str(dest),flush=True);return
        dest=make_zip(work,experiment,prepared,package,True)
        save(work/"delivery.json",{"archive":dest.name,"sha256":sha(dest),"bytes":dest.stat().st_size,
            "all_registered_training_tasks_executed":True,"independent_replay_passed":True,
            "external_transfer_validated":False})
        print("BATCH_EXECUTION_COMPLETE; model quality still requires result review.",flush=True)
        print("RETURN_THIS_ZIP: "+str(dest),flush=True)
    except Exception as e:
        save(work/("failure_"+str(time.time_ns())+".json"),{"error":str(e),"prepared_dir":str(prepared),"model_quality_accepted":False})
        try:print("RETURN_FAILURE_ZIP: "+str(make_zip(work,experiment,prepared,package,False)),flush=True)
        except Exception as packaging_error:print("Failure packaging also failed: "+str(packaging_error),flush=True)
        raise


if __name__=="__main__":
    package=Path(__file__).resolve().parents[1];home=package.parent
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--base-dir",default=str(home/"sf02_v32_round"))
    a.add_argument("--train",default=str(home/"sf02_data/train.parquet"))
    a.add_argument("--work-dir",default=str(package/"work"))
    a.add_argument("--prepared-dir");a.add_argument("--prepare-only",action="store_true")
    args=a.parse_args()
    active=Path(args.work_dir)/"active_prepared.json"
    if not args.prepared_dir and active.exists():args.prepared_dir=load(active)["path"]
    main(args)
