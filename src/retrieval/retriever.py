def retrieve(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)
    return retriever.get_relevant_documents(query)