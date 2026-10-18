
SHELL := /bin/bash

POOL ?= global
OUTDIR ?= outputs/$(POOL)/$*

PIPELINES ?= global_ssl
CARDIOMATE_OUTPUT_DIR ?= BP_SPIKE_PRED

## To run compare_pipelines for either Banaware or Cardiomate users, with specified fruit and scenario
.PHONY: cp_%
cp_%:
	$(eval a := $(word 1,$(subst _, ,$*)))
	$(eval b := $(word 2,$(subst _, ,$*)))
	$(eval c := $(word 3,$(subst _, ,$*)))

	@TASK=fruit; \
	USERS="$($(a)_$(b)_users)"; \
	if [ "$(a)" = "Cardiomate" ]; then \
	  TASK=bp; \
	  USERS="$(Cardiomate_$(b)_$(c)_users)"; \
	fi; \
	for user in $$USERS; do \
	  python -m src.compare_pipelines \
	    --task $$TASK \
	    --user $$user \
	    --fruit $(if $(filter Cardiomate,$(a)),$(b),$(a)) \
	    --scenario $(if $(filter Cardiomate,$(a)),$(c),$(b)) \
	    --output-dir $(OUTDIR) \
	    --pipelines $(PIPELINES); \
	done

## run AL for either Banaware or Cardiomate users, with specified pool, fruit, and scenario
.PHONY: run_%
run_%:
	$(eval a := $(word 1,$(subst _, ,$*)))
	$(eval b := $(word 2,$(subst _, ,$*)))
	$(eval c := $(word 3,$(subst _, ,$*)))

	@TASK=fruit; \
	FRUIT=$(a); \
	SCENARIO=$(b); \
	USERS="$($(a)_$(b)_users)"; \
	WARM=""; \
	if [ "$(POOL)" = "global" ]; then WARM="--warm_start 0"; fi; \
	if [ "$(a)" = "Cardiomate" ]; then \
	  TASK=bp; \
	  FRUIT=$(b); \
	  SCENARIO=$(c); \
	  USERS="$(Cardiomate_$(b)_$(c)_users)"; \
	fi; \
	for user in $$USERS; do \
	  python submit_batch.py \
	    --task $$TASK \
	    --user $$user \
	    --pool $(POOL) \
	    --fruit $$FRUIT \
	    --scenario $$SCENARIO \
	    --output-dir $(OUTDIR) \
	    $$WARM; \
	done

# ---------------------------------------------------------------- personalized AL (IN_A_MEETING)
# meeting/experiments/personal_al_meeting.py, sharded one SLURM job per (fold, repeat).
# Each shard writes its own part file, so jobs never collide and a requeued job resumes.
PAL_FOLDS   ?= 0 1 2
PAL_REPEATS ?= 0 1 2 3 4
PAL_TARGETS ?=
PAL_CORES   ?= 4
PAL_HOURS   ?= 6
PAL_MEM     ?= 16GB
PAL_PART    ?=            # SLURM partition; empty = template.sh's serial_requeue
PAL_ACCT    ?=            # SLURM account, if the partition needs one
ES_DATA_DIR ?=            # ExtraSensory CSVs on the cluster; empty = beside the checkout
LOCAL_PAL   ?= 0          # 1 = run the shards here instead of submitting them

## submit (or, with LOCAL_PAL=1, run) every fold x repeat shard
.PHONY: submit_personal_al
submit_personal_al:
	@if [ "$(LOCAL_PAL)" = "1" ]; then \
	  for f in $(PAL_FOLDS); do for r in $(PAL_REPEATS); do \
	    (cd meeting && python -u experiments/personal_al_meeting.py --fold $$f --repeat $$r \
	      $(if $(PAL_TARGETS),--targets $(PAL_TARGETS),)) ; \
	  done; done; \
	else \
	  python meeting/submit_personal_al.py --folds $(PAL_FOLDS) --repeats $(PAL_REPEATS) \
	    --cores $(PAL_CORES) --hours $(PAL_HOURS) --mem $(PAL_MEM) \
	    $(if $(ES_DATA_DIR),--data_dir $(ES_DATA_DIR),) \
	    $(if $(PAL_PART),--partition $(PAL_PART),) $(if $(PAL_ACCT),--account $(PAL_ACCT),) \
	    $(if $(PAL_TARGETS),--targets $(PAL_TARGETS),) $(if $(filter 1,$(DRYRUN_PAL)),--dryrun,); \
	fi

## print the generated sbatch scripts without submitting anything
.PHONY: dryrun_personal_al
dryrun_personal_al:
	python meeting/submit_personal_al.py --folds $(PAL_FOLDS) --repeats $(PAL_REPEATS) \
	  --cores $(PAL_CORES) --hours $(PAL_HOURS) --mem $(PAL_MEM) \
	  $(if $(ES_DATA_DIR),--data_dir $(ES_DATA_DIR),) \
	  $(if $(PAL_PART),--partition $(PAL_PART),) $(if $(PAL_ACCT),--account $(PAL_ACCT),) \
	  $(if $(PAL_TARGETS),--targets $(PAL_TARGETS),) --dryrun

## collect every shard's part file into per_round.csv, summary.md and the figures
.PHONY: analyze_personal_al
analyze_personal_al:
	cd meeting && python experiments/personal_al_meeting.py --summarize

## the closed-form / greedy check for the variance-reduction acquisition
.PHONY: test_personal_al
test_personal_al:
	cd meeting && python experiments/personal_al_meeting.py --unit_test
