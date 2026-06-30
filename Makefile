.PHONY: test demo stream eval

test:
	PYTHONPATH=src python3 -m unittest discover -s tests

demo:
	PYTHONPATH=src python3 -m voice_rag_agent.cli query "How does the agent keep latency low?"

stream:
	PYTHONPATH=src python3 -m voice_rag_agent.cli query "What do MCP and A2A expose?" --stream

eval:
	PYTHONPATH=src python3 -m voice_rag_agent.cli eval
