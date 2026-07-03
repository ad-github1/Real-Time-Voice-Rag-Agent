.PHONY: test demo stream eval

test:
	PYTHONPATH=src python3 -m unittest discover -s tests

demo:
	PYTHONPATH=src python3 -m voice_rag_agent.cli query "How does the agent keep latency low?"

stream:
	PYTHONPATH=src python3 -m voice_rag_agent.cli query "What do MCP and A2A expose?" --stream

eval:
	PYTHONPATH=src python3 -m voice_rag_agent.cli eval
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

