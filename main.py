from rag_core import OrchestratorConfig, RAGOrchestrator

orc = RAGOrchestrator(config=OrchestratorConfig(top_k=20))  # tweak top_k/model if you like
resp = orc.run(input("Your answer: "), session_id="demo-session")
print("Answer:\n", resp.answer)
print("\nRewritten query:", resp.rewritten_query)
