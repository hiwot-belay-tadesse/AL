SHELL := /bin/bash

run:
	python submit_batch.py

debug: 
	python -m pdb main.py

# Almond_Use_users := ID11 ID13 ID19 ID25 ID28
# Almond_Crave_users := ID11 ID19 ID25
# Melon_Crave_users := ID5 ID9 ID12 ID19 ID20 ID21 ID27
# Melon_Use_users := ID12 ID19 ID20 ID27
# Carrot_Crave_users := ID10 ID11 ID14 ID15 ID18 ID25
# Carrot_Use_users := ID10 ID11 ID13 ID14 ID15 ID18 ID26
Nectarine_Crave_users := ID10 ID11 ID12 ID20 ID21 ID27
Nectarine_Use_users := ID10 ID11 ID12 ID13 ID20 ID21 ID27
Melon_Crave_users := ID5 ID9 ID12 ID19 ID20 ID21 ID27



## runs active learning for banware data 
.PHONY: run_%
run_%:
	$(eval fruit := $(word 1,$(subst _, ,$*)))
	$(eval scenario := $(word 2,$(subst _, ,$*)))
	@for user in $($(fruit)_$(scenario)_users); do \
	  python submit_batch.py \
	  	--task fruit \
	    --user $$user \
	    --pool global \
	    --fruit $(fruit) \
	    --scenario $(scenario)  \
		--warm_start 0;\
	done

.PHONY: run_all
# run_all: run_Melon_Crave run_Melon_Use run_Nectarine_Crave run_Nectarine_Use run_Carrot_Crave run_Carrot_Use run_Almond_Use run_Almond_Crave
run_all: run_Nectarine_Crave run_Nectarine_Use run_Melon_Crave run_Melon_Use

## runs compare_pipelines for banware data
PIPELINES_FRUIT ?= global_supervised
.PHONY: cp_%
cp_%:
	$(eval fruit := $(word 1,$(subst _, ,$*)))
	$(eval scenario := $(word 2,$(subst _, ,$*)))
	@for user in $($(fruit)_$(scenario)_users); do \
	  python -m src.compare_pipelines \
	    --task fruit \
	    --user $$user \
	    --fruit $(fruit) \
	    --scenario $(scenario) \
	    --output-dir OVERFIT_DEBUG_Ban \
	    --pipelines $(PIPELINES_FRUIT) \
		--force_retrain; \
	done

.PHONY: run_all_cp
# run_all_cp: cp_Melon_Crave cp_Melon_Use cp_Nectarine_Crave cp_Nectarine_Use cp_Carrot_Crave cp_Carrot_Use cp_Almond_Use cp_Almond_Crave
run_all_cp: cp_Nectarine_Crave cp_Nectarine_Use


## runs active learning for Cardiomate data
# BP_IDS ?= 10 15 16 17 18 20 22 23 24 25 26 30 31 32 33 34 35 36 39 40

BP_IDS ?=  15 20 22 23 24 25 26 30 31 32 33  35 39
# 15 20 22 23 24 25 26 30 31 32 33  35 39
## Users to report AL: 15, 20, 22, 24, 25, 26, 30, 31, 33, 39
.PHONY: run_bp_submit
run_bp_submit:
	@for pid in $(BP_IDS); do \
	  python submit_batch.py --task bp --participant_id $$pid --pool global --input_df raw; \
	done

# Run compare_pipelines.py for Cardiomate data.
USERS_CP ?= 10 15 16 17 18 20 22 23 24 25 26 30 31 32 33 34 35 36 39 40
TASK_CP ?= bp
FRUIT_CP ?= BP
SCENARIO_CP ?= spike
PIPELINES_CP ?= global_ssl
#personal_ssl global_ssl

.PHONY: run_cp
run_cp:
	@failed=""; \
	for user in $(USERS_CP); do \
	  echo "Running user $$user"; \
	  python -m src.compare_pipelines \
	    --task $(TASK_CP) \
	    --user $$user \
	    --fruit $(FRUIT_CP) \
	    --scenario $(SCENARIO_CP) \
	    --output-dir OVERFIT_DEBUG \
	    --pipelines $(PIPELINES_CP); \
	done
	echo "Failed users:$$failed"


.PHONY: clean_logs
clean_logs:
	find Cardiomate_AL -type f \( -name '*err*.txt' -o -name '*out*.txt' \) -exec rm -rf {} +



SEEDS_MS ?= 41,42,43,44,45,46,47,48,49,50
TARGET_USER_MS ?= 20
POOL_MS ?= global
FRUIT_MS ?= BP
SCENARIO_MS ?= spike
UNLABELED_FRAC_MS ?= 0.0018
DROPOUT_RATE_MS ?= 0.5
WARM_START_MS ?= 0
TASK_MS ?= bp
INPUT_DF_MS ?= raw
OUTDIR_MS ?= avg_auc_results_lr
LOCAL_MS ?= 1
LOCAL_FLAG_MS := $(if $(filter 1 true yes,$(LOCAL_MS)),--local,)

.PHONY: run_multi_seeds
run_multi_seeds:
	python run_multi_seeds.py \
	  --outdir $(OUTDIR_MS) \
	  --seeds $(SEEDS_MS) \
	  --user $(TARGET_USER_MS) \
	  --participant_id $(TARGET_USER_MS) \
	  --pool $(POOL_MS) \
	  --fruit $(FRUIT_MS) \
	  --scenario $(SCENARIO_MS) \
	  --unlabeled_frac $(UNLABELED_FRAC_MS) \
	  --dropout_rate $(DROPOUT_RATE_MS) \
	  --warm_start $(WARM_START_MS) \
	  --task $(TASK_MS) \
	  --input_df $(INPUT_DF_MS) $(LOCAL_FLAG_MS)


SEEDS_AA ?= 41,42,43,44
TARGET_USERS_AA ?=  15 20 22 24 25 26 30 31 33 35 39
METHODS_AA ?= random,coreset
POOL_AA ?= global
FRUIT_AA ?= BP
SCENARIO_AA ?= spike
UNLABELED_FRAC_AA ?= 0.0018
DROPOUT_RATE_AA ?= 0.5
WARM_START_AA ?= 0
TASK_AA ?= bp
INPUT_DF_AA ?= raw
OUTDIR_AA ?= avg_auc_results_lr_retry
CLASSIFIER_AA ?= lr
LOCAL_AA ?= 1
RUN_MODE_FLAG_AA := $(if $(filter 1 true yes,$(LOCAL_AA)),--local,--submit)
ANALYZE_ONLY_AA ?= 0
ANALYZE_FLAG_AA := $(if $(filter 1 true yes,$(ANALYZE_ONLY_AA)),--analyze_only,)

# Exclusion controls. Set AUTO_EXCLUDE_AA=1 to discover bad users automatically
# and derive the target cohort from the survivors (TARGET_USERS_AA is ignored in
# that mode; the cohort = candidates - invalid - manual). EXCLUDE_USERS_AA is the
# manual override and works with or without AUTO_EXCLUDE_AA.
AUTO_EXCLUDE_AA ?= 0
AUTO_EXCLUDE_FLAG_AA := $(if $(filter 1 true yes,$(AUTO_EXCLUDE_AA)),--auto_exclude,)
EXCLUDE_USERS_AA ?=
EXCLUDE_USERS_FLAG_AA := $(if $(EXCLUDE_USERS_AA),--exclude_users $(EXCLUDE_USERS_AA),)
BIN_SIZE_AA ?= 1

# Skip per-seed encoder warmup submission. Set SKIP_WARMUP_AA=1 on step 2 after
# the warmup slurm jobs have finished, so only the AL jobs get submitted.
SKIP_WARMUP_AA ?= 0
SKIP_WARMUP_FLAG_AA := $(if $(filter 1 true yes,$(SKIP_WARMUP_AA)),--skip_warmup,)

.PHONY: run_avg_auc
run_avg_auc:
	python avg_auc.py \
	  --outdir $(OUTDIR_AA) \
	  --seeds $(SEEDS_AA) \
	  --methods $(METHODS_AA) \
	  --users "$(TARGET_USERS_AA)" \
	  --pool $(POOL_AA) \
	  --fruit $(FRUIT_AA) \
	  --scenario $(SCENARIO_AA) \
	  --unlabeled_frac $(UNLABELED_FRAC_AA) \
	  --dropout_rate $(DROPOUT_RATE_AA) \
	  --warm_start $(WARM_START_AA) \
	  --classifier $(CLASSIFIER_AA) \
	  --task $(TASK_AA) \
	  --input_df $(INPUT_DF_AA) \
	  --bin_size $(BIN_SIZE_AA) \
	  $(AUTO_EXCLUDE_FLAG_AA) \
	  $(EXCLUDE_USERS_FLAG_AA) \
	  $(SKIP_WARMUP_FLAG_AA) \
	  $(RUN_MODE_FLAG_AA) \
	  $(ANALYZE_FLAG_AA)

.PHONY: plot_avg_auc_grid
plot_avg_auc_grid:
	python avg_auc.py \
	  --analyze_only \
	  --outdir $(OUTDIR_AA) \
	  --seeds $(SEEDS_AA) \
	  --methods $(METHODS_AA) \
	  --users "$(TARGET_USERS_AA)" \
	  --pool $(POOL_AA) \
	  --fruit $(FRUIT_AA) \
	  --scenario $(SCENARIO_AA) \
	  --unlabeled_frac $(UNLABELED_FRAC_AA) \
	  --dropout_rate $(DROPOUT_RATE_AA) \
	  --warm_start $(WARM_START_AA) \
	  --task $(TASK_AA) \
	  --input_df $(INPUT_DF_AA) \
	  --bin_size $(BIN_SIZE_AA) \
	  $(EXCLUDE_USERS_FLAG_AA)
