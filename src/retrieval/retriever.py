def retrieve(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    return retriever.get_relevant_documents(query)