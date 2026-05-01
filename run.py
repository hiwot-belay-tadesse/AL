import json
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import sys
from pathlib import Path
from types import SimpleNamespace
import pickle
from sklearn.model_selection import train_test_split
import utility
import preprocess
# from preprocess import prepare_data
from new_prep import prepare_data
import new_helper
from new_helper import (
    parse_args,
    set_output_dir,
    compute_budget,
    build_hp_folder,
    write_summary,
    run_experiment,
    reset_seeds,
    aggregate_per_round_labeled_and_compute_auc,
)


DRYRUN = True
_FINAL_COUNTS = {}
_PER_USER_AL_PROGRESS = {}
_PER_USER_ROUND_EVAL = {}
_PER_USER_FULL_DATA_EVAL = {}


args, _ = parse_args()

if args.task == "bp":
    if args.participant_id is None:
        raise SystemExit("For --task bp, please provide --participant_id.")
    args.user = str(args.participant_id)
    args.fruit = "BP"
    args.scenario = "spike"
    BP_MODE = True
else:
    BP_MODE = False

# This is the base directory where the results will be stored.
OUTPUT_DIR = os.environ.get("BANAL_OUTPUT_DIR") or set_output_dir(args.pool, BP_MODE)


unlabeled_frac = [float(args.unlabeled_frac)]
dropout_rate = [float(args.dropout_rate)]
warm_start = [bool(int(args.warm_start))]
T = [50]
K = [100]
# K = [100]

Budget = [None]



QUEUE = [
    # ("uncertainty", dict(
    #     user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario],
    #     task=[args.task], participant_id=[args.participant_id],
    #     T=T, K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate, warm_start=warm_start,
    #     input_df=[args.input_df]
    # )),

    ("random", dict(
        user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario],
        task=[args.task], participant_id=[args.participant_id],
        K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate, warm_start=warm_start, input_df=[args.input_df]
    )),
    (
        "coreset", dict(
            user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario],
            task=[args.task], participant_id=[args.participant_id],
            K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate, 
            warm_start=warm_start, input_df=[args.input_df]
        )
    ),  
    # (
    #     "kmeans", dict(
    #         user=[args.user], pool=[args.pool], fruit=[args.fruit], scenario=[args.scenario],
    #         task=[args.task], participant_id=[args.participant_id],
    #         K=K, Budget=Budget, unlabeled_frac=unlabeled_frac, dropout_rate=dropout_rate, 
    #         warm_start=warm_start, input_df=[args.input_df]
    #     )
    # ) 
]



def run(exp_dir, exp_name, exp_kwargs):
    """
    This is the function that will actually execute the job.
    """
    print("Running experiment {}:".format(exp_name))
    print("Results are stored in:", exp_dir)
    print("with hyperparameters", exp_kwargs)
    print("\n")

    if not exp_kwargs:
        raise SystemExit("refactor_run.py requires exp_kwargs from submit_batch.py.")

    ##Build args namespace from exp_kwargs 
    args_ns = SimpleNamespace(
        user=exp_kwargs["user"],
        pool=exp_kwargs["pool"],
        fruit=exp_kwargs["fruit"],
        scenario=exp_kwargs["scenario"],
        task=exp_kwargs.get("task", "fruit"),
        participant_id=exp_kwargs.get("participant_id"),
        unlabeled_frac=exp_kwargs["unlabeled_frac"],
        dropout_rate=exp_kwargs["dropout_rate"],
        warm_start=exp_kwargs.get("warm_start"),
        results_subdir=exp_kwargs.get("results_subdir", "results"),
        input_df = exp_kwargs["input_df"],
    )

    ## Use OUTPUT_DIR as the top-level output root
    exp_dir_path = Path(exp_dir)
    top_out = Path(OUTPUT_DIR)

    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"


    ## This is so that run_multi_seeds.py can call run() 
    # with different seeds without affecting the global random state 
    split_seed = int(exp_kwargs.get("seed", 42))
    reset_seeds(split_seed)
    prep = prepare_data(
        args=args_ns,
        top_out=top_out,
        shared_enc_root=shared_enc_root,
        shared_cnn_root=shared_cnn_root,
        batch_ssl=32,
        ssl_epochs=100,
        pool=args_ns.pool,
        task=args_ns.task,
        input_df=args_ns.input_df,
    )
    
    if prep is None:
        print(f"Skipping user {args_ns.user}: prepare_data returned no data.")
        return
    
    df_tr, df_all_tr, *_ = prep

    if df_all_tr is not None:
        pre_hash, pre_meta = utility.split_fingerprint(df_all_tr)
        # print(
        #     f"[presplit_fingerprint] hash={pre_hash} "
        #     f"rows={pre_meta['rows']} time_col={pre_meta['time_col']}"
        # )
        (exp_dir_path / "presplit_df_all_tr_fingerprint.txt").write_text(
            "\n".join(
                [
                    "source=run.py",
                    f"rows={pre_meta['rows']}",
                    f"time_col={pre_meta['time_col']}",
                    f"sha256={pre_hash}",
                    "",
                ]
            )
        )

    # Split will be done inside run_experiment to mirror run.py
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

    # Build labeled/unlabeled split outside run_experiment.
    if args_ns.pool == "personal":
        split_source = df_tr.reset_index(drop=True)
    else:
        split_source = df_all_tr.reset_index(drop=True)

    # Hard reset RNG state immediately before split to eliminate drift.
    reset_seeds(split_seed)
    # df_tr_labeled, df_tr_unlabeled = utility.make_labeled_unlabeled_with_target_quota(
    #     split_source,
    #     target_uid=args_ns.user,
    #     unlabeled_frac=(1-uf_val),
    #     seed=split_seed,
    # )

    # df_tr_labeled = df_tr_labeled.reset_index(drop=True)
    # split_hash, split_meta = utility.split_fingerprint(df_tr_labeled)
    # print(
    #     f"[split_fingerprint] seed={split_seed} hash={split_hash} "
    #     f"rows={split_meta['rows']} time_col={split_meta['time_col']}"
    # )
   
    # reset_seeds(42)  
    df_tr_labeled, df_tr_unlabeled = train_test_split(
        split_source,
        test_size=(1- uf_val),
        stratify=split_source["state_val"],
        random_state=split_seed,
    )

  
    # df_tr_labeled, df_tr_unlabeled = new_helper.select_balanced_centroid_seed(split_source, n_per_class=2, label_col="state_val")


    # df_tr_labeled, df_tr_unlabeled = new_helper.split_labeled_unlabeled_kmeans(
    #     split_source, B=10, n_clusters=k_val, random_state=split_seed
    # )
    

    run_out = run_experiment(
        str(exp_dir_path),
        exp_name,
        exp_kwargs,
        args_ns,
        prep,
        # clf_epochs=500,
        clf_epochs=200,
        # clf_patience=20,
        clf_patience=15,
        df_tr_labeled=df_tr_labeled,
        df_tr_unlabeled=df_tr_unlabeled,
        # input_df=input_df,
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

    aggregate_dir = Path(OUTPUT_DIR) / args_ns.pool / "aggregates" /exp_name/hp_folder
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
    if "uncertainty" in _FINAL_COUNTS[key] and "random" in _FINAL_COUNTS[key]:
        u_lab, u_unlab = _FINAL_COUNTS[key]["uncertainty"]
        r_lab, r_unlab = _FINAL_COUNTS[key]["random"]
        print("num of labeled in uncertainty ", u_lab, "num of unlabeled in uncertainty", u_unlab)
        print("num of labeled in random", r_lab,  "num of unlabeled in random", r_unlab)


def main():
    if len(sys.argv) > 3:
        exp_dir = sys.argv[1]
        exp_name = sys.argv[2]
        exp_kwargs = json.loads(sys.argv[3])
        run(exp_dir, exp_name, exp_kwargs)
        return

    # Fallback: allow direct CLI usage without JSON payload
    exp_dir = OUTPUT_DIR
    exp_name = "uncertainty"
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
        "K": 20,
        "T": 30,
        "Budget": None,
    }
    run(exp_dir, exp_name, exp_kwargs)


if __name__ == "__main__":
    main()
