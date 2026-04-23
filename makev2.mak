
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
