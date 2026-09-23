"""ADARP entrypoint. Mirrors run.py but takes prepare_data from preprocess_adarp_data.

Run directly:
    python ADARP/run_adarp.py --task adarp --participant_id 112 --pool global --classifier lr

Only participants with enough button presses to split can be targets; prepare_data
names the usable set if the one asked for is not among them. Everyone else still
feeds the global training pool.

No file in the repo root is modified by this copy.
"""

import json
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import sys
from pathlib import Path
from types import SimpleNamespace
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Anchor to the project root for the shared helpers, and to ADARP/ for the
# preprocessing module, which imports load_adarp flat.
_ADARP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ADARP_DIR.parent
for _path in (str(_REPO_ROOT), str(_ADARP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import utility
import preprocess  # noqa: F401 (kept for parity with run.py)
from preprocess_adarp_data import prepare_data
import new_helper  # noqa: F401
from new_helper import (
    parse_args,
    set_output_dir,
    compute_budget,
    build_hp_folder,
    write_summary,
    run_experiment,
    reset_seeds,
    set_classifier,
    aggregate_per_round_labeled_and_compute_auc,
)


# SSL encoder knobs, ADARP-only. They are read here, passed into
# preprocess_adarp_data.prepare_data, and end at ADARP's two encoders; nothing
# else in the repo sees them, so BP and WESAD keep src/compare_pipelines.py's
# 32/100. Defaults match those values, so setting neither changes nothing.
BATCH_SSL_ADARP = int(os.environ.get("ADARP_BATCH_SSL", "32"))
SSL_EPOCHS_ADARP = int(os.environ.get("ADARP_SSL_EPOCHS", "100"))

DRYRUN = True
_FINAL_COUNTS = {}
_PER_USER_AL_PROGRESS = {}
_PER_USER_ROUND_EVAL = {}
_PER_USER_FULL_DATA_EVAL = {}


args, _ = parse_args()

if args.task in ("adarp", "bp"):
    if args.participant_id is None:
        raise SystemExit("For ADARP, please provide --participant_id (e.g. 112).")
    args.user = str(args.participant_id)
    args.fruit = "ADARP"
    args.scenario = "stress"
    BP_MODE = True  # kept True so set_output_dir picks the same default tree shape
else:
    BP_MODE = False

set_classifier(args.classifier)

if getattr(args, "exclude_users", ""):
    os.environ["BAN_AL_EXCLUDE_USERS"] = args.exclude_users

OUTPUT_DIR = os.environ.get("BAN_AL_OUTPUT_DIR") or set_output_dir(args.pool, BP_MODE)


unlabeled_frac = [float(args.unlabeled_frac)]
dropout_rate = [float(args.dropout_rate)]
warm_start = [bool(int(args.warm_start))]
T = [50]
K = [3]   # matches run.py (BP spike)

# Smallest number of windows of EACH class the initial labeled seed must contain.
# The seed size is raised until a stratified split clears this for both classes,
# so a run never starts from one or two positives.
MIN_PER_CLASS = 5

Budget = [None]


QUEUE = [
    ("random", dict(
        user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario],
        task=[args.task], participant_id=[args.participant_id],
        K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate,
        warm_start=warm_start, input_df=[args.input_df], classifier=[args.classifier],
    )),
    ("coreset", dict(
        user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario],
        task=[args.task], participant_id=[args.participant_id],
        K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate,
        warm_start=warm_start, input_df=[args.input_df], classifier=[args.classifier],
    )),
]


def run(exp_dir, exp_name, exp_kwargs):
    print("Running experiment {}:".format(exp_name))
    print(f"SSL encoder: batch={BATCH_SSL_ADARP} epochs={SSL_EPOCHS_ADARP} "
          "(ADARP_BATCH_SSL / ADARP_SSL_EPOCHS)")
    print("Results are stored in:", exp_dir)
    print("with hyperparameters", exp_kwargs)
    print("\n")

    if not exp_kwargs:
        raise SystemExit("ADARP/run_adarp.py requires exp_kwargs from the driver.")

    classifier_kind = exp_kwargs.get("classifier", "mlp")
    if classifier_kind == "lr" and exp_name == "uncertainty":
        raise SystemExit(
            "aq=uncertainty is not supported with classifier=lr "
            "(MC-dropout/BALD require dropout layers). Pick aq=random or aq=coreset."
        )
    set_classifier(classifier_kind)

    exclude_users = exp_kwargs.get("exclude_users", "")
    if exclude_users:
        os.environ["BAN_AL_EXCLUDE_USERS"] = exclude_users

    args_ns = SimpleNamespace(
        user=exp_kwargs["user"],
        pool=exp_kwargs["pool"],
        fruit=exp_kwargs["fruit"],
        scenario=exp_kwargs["scenario"],
        task=exp_kwargs.get("task", "adarp"),
        participant_id=exp_kwargs.get("participant_id"),
        unlabeled_frac=exp_kwargs["unlabeled_frac"],
        dropout_rate=exp_kwargs["dropout_rate"],
        warm_start=exp_kwargs.get("warm_start"),
        results_subdir=exp_kwargs.get("results_subdir", "results"),
        input_df=exp_kwargs["input_df"],
        classifier=classifier_kind,
    )

    exp_dir_path = Path(exp_dir)
    top_out = Path(OUTPUT_DIR)

    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"

    split_seed = int(exp_kwargs.get("seed", 42))
    reset_seeds(split_seed)
    prep = prepare_data(
        args=args_ns,
        top_out=top_out,
        shared_enc_root=shared_enc_root,
        shared_cnn_root=shared_cnn_root,
        batch_ssl=BATCH_SSL_ADARP,
        ssl_epochs=SSL_EPOCHS_ADARP,
        pool=args_ns.pool,
        task=args_ns.task,
        input_df=args_ns.input_df,
        seed=split_seed,
    )

    if prep is None:
        print(f"Skipping user {args_ns.user}: prepare_data returned no data.")
        return

    (df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, *_,) = prep

    # Under LR, fold val into the training pool so AL can query from it.
    if classifier_kind == "lr" and df_val is not None and len(df_val) > 0:
        if df_all_tr is not None:
            df_all_tr = pd.concat([df_all_tr, df_val], ignore_index=False)
        if df_tr is not None:
            df_tr = pd.concat([df_tr, df_val], ignore_index=False)
        df_val = df_val.iloc[0:0].copy()

    if df_all_tr is not None:
        pre_hash, pre_meta = utility.split_fingerprint(df_all_tr)
        (exp_dir_path / "presplit_df_all_tr_fingerprint.txt").write_text(
            "\n".join([
                "source=ADARP/run_adarp.py",
                f"rows={pre_meta['rows']}",
                f"time_col={pre_meta['time_col']}",
                f"sha256={pre_hash}",
                "",
            ])
        )

    uf_val = float(exp_kwargs["unlabeled_frac"])
    dr_val = float(exp_kwargs["dropout_rate"])
    k_val = int(exp_kwargs["K"])
    t_val = exp_kwargs.get("T")
    if isinstance(t_val, (list, tuple)):
        t_val = t_val[0] if t_val else None

    budget = compute_budget(args_ns.pool, df_tr, df_all_tr, uf_val, k_val)
    exp_kwargs["Budget"] = budget

    hp_folder = build_hp_folder(uf_val, k_val, budget, t_val, dr_val)
    if exp_dir_path.name != hp_folder:
        exp_dir_path = exp_dir_path / hp_folder
    exp_dir_path.mkdir(parents=True, exist_ok=True)

    write_summary(
        str(exp_dir_path),
        args_ns.user,
        args_ns.pool,
        args_ns.fruit,
        args_ns.scenario,
        uf_val,
        dr_val,
        k_val,
        budget,
        t_val,
    )

    if args_ns.pool == "personal":
        split_source = df_tr.reset_index(drop=True)
    else:
        split_source = df_all_tr.reset_index(drop=True)

    reset_seeds(split_seed)

    n_classes = split_source["state_val"].nunique()
    n_total = len(split_source)
    n_pos = int(split_source["state_val"].sum())
    n_neg = n_total - n_pos

    # Guarantee MIN_PER_CLASS of each class survives the stratified split. For a class
    # holding c of N rows, stratify picks ~round(n_labeled * c / N); invert that to get
    # the smallest n_labeled that clears the threshold for the rarer class.
    class_floors = [
        int(np.ceil(n_total / c * MIN_PER_CLASS)) for c in (n_pos, n_neg) if c > 0
    ]
    min_labeled = max([2 * n_classes] + class_floors)
    min_labeled = min(min_labeled, n_total)

    n_labeled_requested = int(round(uf_val * n_total))
    n_labeled = max(min_labeled, n_labeled_requested)
    effective_uf = n_labeled / n_total
    if effective_uf != uf_val:
        print(
            f"[split] unlabeled_frac={uf_val} yields {n_labeled_requested} labeled rows; "
            f"flooring to {n_labeled} (effective uf={effective_uf:.4f}); "
            f"MIN_PER_CLASS={MIN_PER_CLASS}, n_pos={n_pos} n_neg={n_neg} of {n_total}."
        )

    df_tr_labeled, df_tr_unlabeled = train_test_split(
        split_source,
        test_size=(1 - effective_uf),
        stratify=split_source["state_val"],
        random_state=42,
    )

    Z_split = utility.encode_single_df(
        split_source,
        enc_hr,
        enc_st,
        args_ns.pool,
    )

    from embedding_spread import compute_user_class_spread
    compute_user_class_spread(
        Z=Z_split,
        df=split_source,
        user_id=args_ns.user,
        label_col="state_val",
        user_col="user_id",
        out_path=exp_dir_path / "embedding_spread.json",
    )

    run_out = run_experiment(
        str(exp_dir_path),
        exp_name,
        exp_kwargs,
        args_ns,
        prep,
        clf_epochs=200,
        clf_patience=15,
        df_tr_labeled=df_tr_labeled,
        df_tr_unlabeled=df_tr_unlabeled,
    )

    if run_out is None:
        print(f"Skipping collection for user {args_ns.user}: run_experiment returned None.")
        return

    labeled_len = int(run_out["labeled_len"])
    unlabeled_len = int(run_out["unlabeled_len"])

    user_key = str(args_ns.user)
    _PER_USER_AL_PROGRESS[user_key] = run_out.get("al_progress")
    _PER_USER_ROUND_EVAL[user_key] = run_out.get("round_eval_payloads")
    _PER_USER_FULL_DATA_EVAL[user_key] = run_out.get("full_data_eval_payload")

    aggregate_dir = Path(OUTPUT_DIR) / args_ns.pool / "aggregates" / exp_name / hp_folder
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    def _read_pickle_dict(path: Path):
        if not path.exists():
            return {}
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    existing_progress = _read_pickle_dict(aggregate_dir / "per_user_al_progress.pkl")
    merged_progress = {**existing_progress, **_PER_USER_AL_PROGRESS}
    existing_round_eval = _read_pickle_dict(aggregate_dir / "per_user_round_eval.pkl")
    merged_round_eval = {**existing_round_eval, **_PER_USER_ROUND_EVAL}
    existing_full_data_eval = _read_pickle_dict(aggregate_dir / "per_user_full_data_eval.pkl")
    merged_full_data_eval = {**existing_full_data_eval, **_PER_USER_FULL_DATA_EVAL}

    _PER_USER_AL_PROGRESS.clear()
    _PER_USER_AL_PROGRESS.update(merged_progress)
    _PER_USER_ROUND_EVAL.clear()
    _PER_USER_ROUND_EVAL.update(merged_round_eval)
    _PER_USER_FULL_DATA_EVAL.clear()
    _PER_USER_FULL_DATA_EVAL.update(merged_full_data_eval)

    with open(aggregate_dir / "per_user_al_progress.pkl", "wb") as f:
        pickle.dump(merged_progress, f)
    with open(aggregate_dir / "per_user_round_eval.pkl", "wb") as f:
        pickle.dump(merged_round_eval, f)
    with open(aggregate_dir / "per_user_full_data_eval.pkl", "wb") as f:
        pickle.dump(merged_full_data_eval, f)

    agg = aggregate_per_round_labeled_and_compute_auc(
        merged_progress,
        per_user_round_eval=merged_round_eval,
        per_user_full_data_eval=merged_full_data_eval,
    )
    agg_auc = agg.get("auc_per_round")
    if agg_auc is not None:
        agg_auc.to_csv(aggregate_dir / "auc_per_round_aggregated.csv", index=False)
    full_data_auc = agg.get("full_data_auc")
    if full_data_auc is not None:
        with open(aggregate_dir / "full_data_auc_aggregated.json", "w") as f:
            json.dump(full_data_auc, f, indent=2)

    base_root = Path(OUTPUT_DIR) / args_ns.pool / args_ns.user / f"{args_ns.fruit}_{args_ns.scenario}" / hp_folder
    key = (str(base_root), args_ns.task, args_ns.participant_id)
    _FINAL_COUNTS.setdefault(key, {})
    _FINAL_COUNTS[key][exp_name] = (labeled_len, unlabeled_len)


def main():
    if len(sys.argv) > 3:
        exp_dir = sys.argv[1]
        exp_name = sys.argv[2]
        exp_kwargs = json.loads(sys.argv[3])
        run(exp_dir, exp_name, exp_kwargs)
        return

    exp_dir = OUTPUT_DIR
    exp_name = "random"
    exp_kwargs = {
        "user": args.user,
        "pool": args.pool,
        "fruit": args.fruit,
        "scenario": args.scenario,
        "task": args.task,
        "participant_id": args.participant_id,
        "unlabeled_frac": float(args.unlabeled_frac),
        "dropout_rate": float(args.dropout_rate),
        "warm_start": int(args.warm_start),
        "K": int(K[0]),
        "T": int(T[0]),
        "Budget": None,
        "input_df": args.input_df,
        "classifier": args.classifier,
    }
    run(exp_dir, exp_name, exp_kwargs)


if __name__ == "__main__":
    main()
