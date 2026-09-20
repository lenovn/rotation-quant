import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def start_tmux(command, project, environment, log_path, session):
    socket = "rotation-quant-phase3"
    child = ["env", *[f"{key}={value}" for key, value in environment.items()], *command]
    shell_command = f"exec {shlex.join(child)} > {shlex.quote(str(log_path))} 2>&1"
    result = subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", session,
                             "-c", str(project), "-P", "-F", "#{pane_pid}", shell_command],
                            check=True, text=True, capture_output=True)
    return int(result.stdout.strip()), socket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(1, 3, 5, 6, 7), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", choices=("train", "postprocess", "sequential", "local-d", "distill"), default="train")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Run name must be one path component")
    if not shutil.which("tmux"):
        parser.error("tmux is required for new launches; no environment changes were made")
    source = Path(__file__).resolve().parents[2]
    project = source.parents[1]
    directory = project / "runs/phase3"
    output = directory / args.name
    launch_path = directory / (args.name + ".launch.json")
    log_path = directory / (args.name + ".log")
    if output.exists() or launch_path.exists() or log_path.exists():
        parser.error("This run name is already used; preserve existing artifacts")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True)
    snapshot = []
    for line in query.splitlines():
        index, uuid, used, total, utilization = [field.strip() for field in line.split(",")]
        snapshot.append(dict(index=int(index), uuid=uuid, memory_used_mib=int(used),
                             memory_total_mib=int(total), utilization=int(utilization)))
    selected = next(record for record in snapshot if record["index"] == args.gpu)
    environment = dict(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       HF_HOME=str(project / "cache/huggingface"), GIT_OPTIONAL_LOCKS="0",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    entrypoint = {"train": "run.py", "postprocess": "postprocess.py", "sequential": "sequential_postprocess.py",
                  "local-d": "local_d.py", "distill": "distill.py"}[args.task]
    command = ["/home/dongpeiyan/miniconda3/envs/rotation-quant-p0/bin/python", "-u",
               str(Path(__file__).parent / entrypoint),
               "--output", str(output), *arguments]
    session = "phase3-" + args.name.replace(".", "_").replace(":", "_")
    with log_path.open("x"):
        pass
    pid, socket = start_tmux(command, project, environment, log_path, session)
    record = dict(pid=pid, gpu=args.gpu, started=time.time(), command=command,
                  backend="tmux", tmux_socket=socket, tmux_session=session,
                  attach_command=shlex.join(["tmux", "-L", socket, "attach", "-t", session]),
                  child_environment=environment,
                  shell_command=shlex.join(command), selected_gpu=selected, gpu_snapshot=snapshot,
                  output=str(output), log=str(log_path), policy="latest user authorization: actual headroom, parallel independent jobs, no hard ratio/memory/time cap, never terminate others")
    with launch_path.open("x") as outfile:
        json.dump(record, outfile, indent=2)
        outfile.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
