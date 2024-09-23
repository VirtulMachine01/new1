from llm_chains import load_vectorstore

vectorstore = load_vectorstore("faiss_db/faiss_index.index")
docs = vectorstore.similarity_search("components of fusecap", k=5)
print(f"Document: {docs.page_content}")