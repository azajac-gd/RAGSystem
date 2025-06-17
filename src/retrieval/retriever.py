from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def retrieve(vectorstore, query, page_start=None, page_end=None, section=None, type_=None):
    conditions = []

    if page_start is not None or page_end is not None:
        range_args = {}
        if page_start is not None:
            range_args["gte"] = page_start
        if page_end is not None:
            range_args["lte"] = page_end
        conditions.append(
            FieldCondition(
                key="metadata.page",
                range=Range(**range_args)
            )
        )

    if section is not None:
        conditions.append(
            FieldCondition(
                key="metadata.section",
                match=MatchValue(value=section)
            )
        )

    if type_ is not None:
        conditions.append(
            FieldCondition(
                key="metadata.type",
                match=MatchValue(value=type_)
            )
        )

    filter_ = Filter(must=conditions) if conditions else None

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "filter": filter_
        }
    )

    return retriever.get_relevant_documents(query)
