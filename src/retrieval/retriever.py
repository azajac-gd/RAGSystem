def retrieve(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    return retriever.get_relevant_documents(query)