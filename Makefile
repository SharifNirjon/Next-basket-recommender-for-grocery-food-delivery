# Usage: make all            (full data)
#        make all SAMPLE=1   (~10% of users, fast iteration)
PY      ?= .venv/bin/python
FLAG    := $(if $(SAMPLE),--sample,)
ARTS    := $(if $(SAMPLE),artifacts/sample,artifacts)
PORT    ?= 8000

.PHONY: setup data features train evaluate artifacts serve test lint format bench docker all clean

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install --no-deps -e .

data:
	$(PY) -m recsys.data.download $(FLAG)
	$(PY) -m recsys.data.prepare $(FLAG)

features:
	$(PY) -m recsys.candidates.item2vec $(FLAG)
	$(PY) -m recsys.candidates.generate $(FLAG)
	$(PY) -m recsys.features.build $(FLAG)

train:
	$(PY) -m recsys.ranker.train $(FLAG)

evaluate:
	$(PY) -m recsys.evaluate.run $(FLAG)

artifacts:
	$(PY) -m recsys.serve.export $(FLAG)

serve:
	ARTIFACTS_DIR=$(ARTS) $(PY) -m uvicorn recsys.serve.app:app --host 0.0.0.0 --port $(PORT) --no-access-log

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts

format:
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m black src tests scripts

# Requires a running API (make serve / docker compose up).
bench:
	$(PY) scripts/benchmark_latency.py --url http://localhost:$(PORT) --n 1000 --artifacts $(ARTS) \
		--out $(if $(SAMPLE),reports/sample/latency.json,reports/latency.json)

docker:
	docker compose build

all: data features train evaluate artifacts

clean:
	rm -rf data/processed/* artifacts/* mlruns reports/sample
