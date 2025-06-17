from qdrant_client.models import Filter, FieldCondition, MatchValue


def retrieve(vectorstore, query, page_start=None, page_end=None, section=None, type_=None):
    must_conditions = []

    if page_start is not None or page_end is not None:
        range_filter = {}
        if page_start is not None:
            range_filter["gte"] = page_start
        if page_end is not None:
            range_filter["lte"] = page_end
        must_conditions.append({
            "key": "metadata.page", 
            "range": range_filter
        })
        
    if section is not None:
        must_conditions.append({
            "key": "metadata.section", 
            "match": {"value": section}
        })

    if type_ is not None:
        must_conditions.append({
            "key": "metadata.type", 
            "match": {"value": type_}
        })

    filter_ = {"must": must_conditions} if must_conditions else None
    print(filter_)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "filter": filter_}
    )

    return retriever.get_relevant_documents(query)
