"""Submit personal_al_meeting.py to SLURM, one job per (fold, repeat) shard.

Fills the repo's template.sh the way submit_batch.py does -- read the template, substitute, write
tmp/slurm_<name>.sh, sbatch it -- but this experiment takes CLI flags rather than a JSON kwargs
blob, so the template's final `python -u refactor_run.py ...` line is replaced wholesale.

Each shard writes its own results/personal_al_meeting/per_round_fold<f>_r<r>.csv, so concurrent
jobs never append to the same file, and a requeued job skips the (target, repeat) units already in
its part file. Collect everything afterwards with:

    python experiments/personal_al_meeting.py --summarize

    python submit_personal_al.py --dryrun            # print the scripts, submit nothing
    python submit_personal_al.py                     # 3 folds x 5 repeats = 15 jobs
    python submit_personal_al.py --folds 0 --repeats 0 --targets M17 M18
"""
import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(ROOT, "template.sh")
TMP_DIR = os.path.join(ROOT, "tmp")
LOGS = "meeting/results/personal_al_meeting/logs"


def build(template, name, cmd, hours, mem, cores, data_dir=None):
    s = template
    s = s.replace("#SBATCH -n 1 # Number of cores", f"#SBATCH -n {cores} # Number of cores")
    s = s.replace("#SBATCH -t 0-010:00", f"#SBATCH -t 0-{hours:03d}:00")
    s = s.replace("#SBATCH --mem=30GB", f"#SBATCH --mem={mem}")
    s = s.replace("ADARP/results/logs", LOGS)
    s = s.replace("#SBATCH -o %s/out_%%j.txt" % LOGS, f"#SBATCH -o {LOGS}/{name}_%j.txt")
    s = s.replace("#SBATCH -e %s/err_%%j.txt" % LOGS, f"#SBATCH -e {LOGS}/{name}_%j.err")
    body = s.split("# python -u run.py EXPDIR EXPNAME KWARGS")[0]
    # BLAS threads must match -n: the acquisition step is one big matmul per round
    env = (f"export OMP_NUM_THREADS={cores}\nexport OPENBLAS_NUM_THREADS={cores}\n"
           f"export MKL_NUM_THREADS={cores}\nexport VECLIB_MAXIMUM_THREADS={cores}\n")
    if data_dir:                       # walking/data.py reads ES_DATA_DIR before its default path
        env += f'export ES_DATA_DIR="{data_dir}"\n'
    return body + env + f"\ncd meeting\n{cmd}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--cores", type=int, default=4); ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--mem", default="16GB")
    ap.add_argument("--data_dir", default=os.environ.get("ES_DATA_DIR"),
                    help="ExtraSensory CSVs on the cluster; exported as ES_DATA_DIR in each job")
    ap.add_argument("--dryrun", action="store_true", help="write the scripts but do not sbatch")
    args = ap.parse_args()
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, LOGS), exist_ok=True)
    template = open(TEMPLATE).read()

    for f in args.folds:
        for r in args.repeats:
            name = f"pal_f{f}_r{r}"
            cmd = (f"python -u experiments/personal_al_meeting.py --fold {f} --repeat {r}"
                   + (" --targets " + " ".join(args.targets) if args.targets else ""))
            path = os.path.join(TMP_DIR, f"slurm_{name}.sh")
            open(path, "w").write(build(template, name, cmd, args.hours, args.mem, args.cores,
                                          args.data_dir))
            if args.dryrun:
                print(f"[dryrun] {path}: {cmd}")
            else:
                ret = subprocess.call(["sbatch", path])
                print(("submitted " if ret == 0 else f"FAILED ({ret}) ") + name)


if __name__ == "__main__":
    main()
