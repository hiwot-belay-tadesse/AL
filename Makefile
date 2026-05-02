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
PIPELINES ?= global_supervised
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
	    --pipelines $(PIPELINES) \
		--force_retrain; \
	done

.PHONY: run_all_cp
# run_all_cp: cp_Melon_Crave cp_Melon_Use cp_Nectarine_Crave cp_Nectarine_Use cp_Carrot_Crave cp_Carrot_Use cp_Almond_Use cp_Almond_Crave
run_all_cp: cp_Nectarine_Crave cp_Nectarine_Use


## runs active learning for Cardiomate data
# BP_IDS ?= 10 15 16 17 18 20 22 23 24 25 26 30 31 32 33 34 35 36 39 40

BP_IDS ?= 15
# 15 20 22 24 25 26 30 31 33 39
## Users to report AL: 15, 20, 22, 24, 25, 26, 30, 31, 33, 39
.PHONY: run_bp_submit
run_bp_submit:
	@for pid in $(BP_IDS); do \
	  python submit_batch.py --task bp --participant_id $$pid --pool global --input_df raw; \
	done

# Run compare_pipelines.py for Cardiomate data.
USERS ?= 10 15 16 17 18 20 22 23 24 25 26 30 31 32 33 34 35 36 39 40

TASK ?= bp
FRUIT ?= BP
SCENARIO ?= spike
PIPELINES_cp ?= global_ssl
#personal_ssl global_ssl

.PHONY: run_cp
run_cp:
	@failed=""; \
	for user in $(USERS); do \
	  echo "Running user $$user"; \
	  python -m src.compare_pipelines \
	    --task $(TASK) \
	    --user $$user \
	    --fruit $(FRUIT) \
	    --scenario $(SCENARIO) \
	    --output-dir OVERFIT_DEBUG \
	    --pipelines $(PIPELINES_cp); \
	done
	echo "Failed users:$$failed"


.PHONY: clean_logs
clean_logs:
	find Cardiomate_AL -type f \( -name '*err*.txt' -o -name '*out*.txt' \) -exec rm -rf {} +



SEEDS ?= 41,42,43
TARGET_USER ?= 20
POOL ?= global
FRUIT ?= BP
SCENARIO ?= spike
UNLABELED_FRAC ?= 0.22
DROPOUT_RATE ?= 0.5
WARM_START ?= 0
TASK ?= bp
INPUT_DF ?= raw
OUTDIR ?= multiseeds
LOCAL ?= 1
LOCAL_FLAG := $(if $(filter 1 true yes,$(LOCAL)),--local,)

.PHONY: run_multi_seeds
run_multi_seeds:
	python run_multi_seeds.py \
	  --outdir $(OUTDIR) \
	  --seeds $(SEEDS) \
	  --user $(TARGET_USER) \
	  --participant_id $(TARGET_USER) \
	  --pool $(POOL) \
	  --fruit $(FRUIT) \
	  --scenario $(SCENARIO) \
	  --unlabeled_frac $(UNLABELED_FRAC) \
	  --dropout_rate $(DROPOUT_RATE) \
	  --warm_start $(WARM_START) \
	  --task $(TASK) \
	  --input_df $(INPUT_DF) $(LOCAL_FLAG)
