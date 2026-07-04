.PHONY: test lint typecheck static demo stream eval benchmark profile quality docker-build docker-api docker-up docker-down docker-health docker-ready

test:
	PYTHONPATH=src python3 -m pytest tests -q

lint:
	PYTHONPATH=src python3 -m ruff format --check src tests scripts
	PYTHONPATH=src python3 -m ruff check src tests scripts

typecheck:
	PYTHONPATH=src python3 -m mypy src scripts tests

static:
	python3 -m compileall -q src scripts
	$(MAKE) lint
	$(MAKE) typecheck

demo:
	PYTHONPATH=src python3 -m voice_rag_agent.cli query "How does the agent keep latency low?"

stream:
	PYTHONPATH=src python3 -m voice_rag_agent.cli query "What do MCP and A2A expose?" --stream

eval:
	PYTHONPATH=src python3 -m voice_rag_agent.cli eval --cases tests/fixtures/eval_cases.json --output outputs/eval_results.json --report outputs/eval_report.md

benchmark:
	PYTHONPATH=src python3 -m voice_rag_agent.cli benchmark --mode retrieve --queries tests/fixtures/benchmark_queries.json --concurrency 8 --repeat 5 --max-p95-ms 50 --max-error-rate 0 --min-qps 100 --output outputs/benchmark_retrieve_report.json

profile:
	PYTHONPATH=src python3 -m voice_rag_agent.cli profile "voice latency streaming" --mode retrieve --repeat 100 --warmup 5 --output outputs/profile_retrieve.json

quality:
	PYTHONPATH=src python3 scripts/quality_gate.py

# Docker API deployment

docker-build:
	docker build -t voice-rag-agent-api:local .

docker-api:
	docker run --rm -p 8000:8000 voice-rag-agent-api:local

docker-up:
	docker compose up --build api

docker-down:
	docker compose down

docker-health:
	curl http://127.0.0.1:8000/health

docker-ready:
	curl http://127.0.0.1:8000/ready
