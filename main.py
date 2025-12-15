from rag_core import RAGOrchestrator

orc = RAGOrchestrator(top_k=20)  # tweak top_k/model if you like
resp = orc.run(input("Your answer: "), session_id="demo-session")
print("Answer:\n", resp["answer"])
print("\nRewritten query:", resp["rewritten_query"])
print("\nChunks:")
for r in resp["retrieved_chunks"]:
    print("-", r.get("chunk_id"), r.get("score"), r.get("source"), r.get("page_number"))
