#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[31]:


def load_all_pdfs():
    folder_path = "data/raw"

    num_docs = 0
    all_docs = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(folder_path, filename)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["document_name"] = filename

            all_docs.extend(docs)
            num_docs += 1

            print(f"Loaded: {filename} → {len(docs)} pages")

    print("\nTotal PDFs:", num_docs)
    print("Total pages:", len(all_docs))

    return all_docs


# In[32]:


docs = load_all_pdfs()


# In[33]:


print(type(docs))
print(type(docs[0]))


# In[35]:


print(docs[0].page_content[:500])


# In[36]:


def combine_document_pages(docs):

    grouped_documents = {}

    for doc in docs:

        document_name = doc.metadata["document_name"]

        if document_name not in grouped_documents:
            grouped_documents[document_name] = []

        grouped_documents[document_name].append(doc)

    combined_documents = []

    for document_name, pages in grouped_documents.items():

        # Keep pages in original order
        pages = sorted(
            pages,
            key=lambda x: x.metadata.get("page", 0)
        )

        full_text = "\n".join(
            page.page_content for page in pages
        )

        metadata = pages[0].metadata.copy()

        metadata["document_name"] = document_name
        metadata["total_pages"] = len(pages)

        combined_documents.append(
            Document(
                page_content=full_text,
                metadata=metadata
            )
        )

    return combined_documents


# In[37]:


combined_docs = combine_document_pages(docs)

print("Combined documents:", len(combined_docs))


# In[38]:


for doc in combined_docs:

    print("\nDocument:", doc.metadata["document_name"])
    print("Characters:", len(doc.page_content))


# In[40]:


print(combined_docs[1].page_content[:2000])


# In[41]:


def get_state(filename):

    filename = filename.lower()

    if "haryana" in filename:
        return "Haryana"

    if filename.startswith("up") or "uttar_pradesh" in filename:
        return "Uttar Pradesh"

    return "India"


# In[42]:


print(get_state("HARYANA.pdf"))
print(get_state("UP.pdf"))


# In[43]:


def get_document_type(filename):

    filename = filename.lower()

    if "haryana" in filename:
        return "State Motor Vehicle Rules"

    if filename.startswith("up") or "uttar_pradesh" in filename:
        return "State Motor Vehicle Rules"

    if "cmvr" in filename or "central_motor" in filename:
        return "Central Motor Vehicle Rules"

    if "motor_vehicles_act" in filename or "motor_vehicle_act" in filename:
        return "Central Motor Vehicle Act"

    return "Motor Vehicle Law"


# In[44]:


def split_into_rules(text):

    pattern = r'(?=\b\d+\.\s+[A-Z])'

    parts = re.split(pattern, text)

    return parts


# In[45]:


haryana_text = combined_docs[0].page_content

rules = split_into_rules(haryana_text)

print("Detected parts:", len(rules))


# In[46]:


for rule in rules[:10]:

    print("=" * 80)
    print(rule[:500])


# In[47]:


def get_rule_number(text):

    match = re.match(
        r'^\s*(\d+)\.\s+',
        text
    )

    if match:
        return match.group(1)

    return None


# In[48]:


print(get_rule_number("1. Short title and commencement"))
print(get_rule_number("9. Authority for making appointment"))


# ## Chunks

# In[49]:


def create_legal_rule_chunks(combined_docs):

    legal_chunks = []

    for doc in combined_docs:

        text = doc.page_content
        filename = doc.metadata["document_name"]

        state = get_state(filename)
        document_type = get_document_type(filename)

        rule_parts = split_into_rules(text)

        for part in rule_parts:

            part = part.strip()

            if not part:
                continue

            rule_number = get_rule_number(part)

            metadata = {
                "document_name": filename,
                "state": state,
                "document_type": document_type,
                "rule": rule_number
            }

            legal_chunks.append(
                Document(
                    page_content=part,
                    metadata=metadata
                )
            )

    return legal_chunks


# In[50]:


legal_chunks = create_legal_rule_chunks(combined_docs)

print("Total legal chunks:", len(legal_chunks))


# In[52]:


print(legal_chunks[1].page_content)
print(legal_chunks[1].metadata)


# In[53]:


for chunk in legal_chunks:

    if chunk.metadata["document_name"] == "UP.pdf":

        print(chunk.metadata)
        print(chunk.page_content[:500])

        break


# In[54]:


sub_chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)


# In[56]:


def create_final_chunks(legal_chunks):

    final_chunks = []

    for chunk in legal_chunks:

        text = chunk.page_content

        # Small enough → keep the complete legal rule
        if len(text) <= 1200:

            final_chunks.append(chunk)

        # Very large rule → split into smaller pieces
        else:

            smaller_chunks = sub_chunk_splitter.split_documents(
                [chunk]
            )

            for i, sub_chunk in enumerate(smaller_chunks):

                sub_chunk.metadata["sub_chunk"] = i + 1

                final_chunks.append(sub_chunk)

    return final_chunks


# In[57]:


final_chunks = create_final_chunks(legal_chunks)

print("Final chunks:", len(final_chunks))


# In[58]:


print(final_chunks[0].page_content)
print(final_chunks[0].metadata)


# In[59]:


for chunk in final_chunks:

    if chunk.metadata.get("rule") == "138":

        print(chunk.metadata)
        print(chunk.page_content[:500])
        print("-" * 80)


# In[60]:


def combine_document_pages_with_page_map(docs):

    grouped_documents = {}

    for doc in docs:

        document_name = doc.metadata["document_name"]

        if document_name not in grouped_documents:
            grouped_documents[document_name] = []

        grouped_documents[document_name].append(doc)

    combined_documents = []

    for document_name, pages in grouped_documents.items():

        pages = sorted(
            pages,
            key=lambda x: x.metadata.get("page", 0)
        )

        full_text = ""
        page_map = []

        for page in pages:

            page_number = page.metadata.get("page", 0)
            page_label = page.metadata.get(
                "page_label",
                page_number + 1
            )

            start = len(full_text)

            full_text += page.page_content + "\n"

            end = len(full_text)

            page_map.append({
                "start": start,
                "end": end,
                "page": page_number,
                "page_label": page_label
            })

        metadata = pages[0].metadata.copy()

        metadata["document_name"] = document_name
        metadata["total_pages"] = len(pages)

        combined_documents.append(
            Document(
                page_content=full_text,
                metadata=metadata
            )
        )

        # page map alag attach karenge
        combined_documents[-1].metadata["page_map"] = page_map

    return combined_documents


# In[61]:


combined_docs = combine_document_pages_with_page_map(docs)

print("Combined documents:", len(combined_docs))


# In[62]:


print(combined_docs[0].metadata["page_map"][:3])


# In[63]:


def get_chapter_at_position(text, position):

    chapter_pattern = r'CHAPTER\s+[IVXLCDM]+'

    chapters = list(
        re.finditer(
            chapter_pattern,
            text,
            re.IGNORECASE
        )
    )

    current_chapter = None

    for match in chapters:

        if match.start() <= position:
            current_chapter = match.group(0).strip()
        else:
            break

    return current_chapter


# In[64]:


def get_pages_for_position(page_map, start_position, end_position):

    pages = []

    for page in page_map:

        if page["end"] > start_position and page["start"] < end_position:
            pages.append(page)

    if not pages:
        return None, None

    page_start = pages[0]["page_label"]
    page_end = pages[-1]["page_label"]

    return page_start, page_end


# In[65]:


def create_legal_rule_chunks(combined_docs):

    legal_chunks = []

    for doc in combined_docs:

        text = doc.page_content
        filename = doc.metadata["document_name"]

        state = get_state(filename)
        document_type = get_document_type(filename)

        page_map = doc.metadata["page_map"]

        # Detect numbered rules
        pattern = r'(?=\b\d+\.\s+[A-Z])'

        parts = list(
            re.finditer(pattern, text)
        )

        for i, match in enumerate(parts):

            start = match.start()

            if i + 1 < len(parts):
                end = parts[i + 1].start()
            else:
                end = len(text)

            part = text[start:end].strip()

            if not part:
                continue

            rule_number = get_rule_number(part)

            chapter = get_chapter_at_position(
                text,
                start
            )

            page_start, page_end = get_pages_for_position(
                page_map,
                start,
                end
            )

            metadata = {
                "document_name": filename,
                "state": state,
                "document_type": document_type,
                "chapter": chapter,
                "rule": rule_number,
                "page_start": page_start,
                "page_end": page_end
            }

            legal_chunks.append(
                Document(
                    page_content=part,
                    metadata=metadata
                )
            )

    return legal_chunks


# In[66]:


legal_chunks = create_legal_rule_chunks(combined_docs)

print("Total legal chunks:", len(legal_chunks))


# In[67]:


for chunk in legal_chunks:

    if chunk.metadata.get("rule") == "138":

        print(chunk.metadata)
        print()
        print(chunk.page_content[:500])
        break


# In[68]:


len(chunk.page_content)


# In[69]:


for chunk in legal_chunks:

    if chunk.metadata.get("rule") == "138":
        print("Characters:", len(chunk.page_content))
        break


# In[70]:


sub_chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)


# In[71]:


def create_final_chunks(legal_chunks):

    final_chunks = []

    for chunk in legal_chunks:

        text = chunk.page_content

        # If legal rule is already small,
        # keep the complete rule.
        if len(text) <= 1200:

            chunk.metadata["sub_chunk"] = 1
            final_chunks.append(chunk)

        else:

            smaller_chunks = sub_chunk_splitter.split_documents(
                [chunk]
            )

            for i, sub_chunk in enumerate(smaller_chunks):

                # Preserve original legal metadata
                sub_chunk.metadata.update(
                    chunk.metadata
                )

                sub_chunk.metadata["sub_chunk"] = i + 1

                final_chunks.append(sub_chunk)

    return final_chunks


# In[72]:


final_chunks = create_final_chunks(legal_chunks)

print("Final chunks:", len(final_chunks))


# In[73]:


for chunk in final_chunks:

    if chunk.metadata.get("rule") == "138":

        print(chunk.metadata)
        print("Characters:", len(chunk.page_content))
        print(chunk.page_content[:300])
        print("-" * 80)


# In[74]:


get_ipython().system('pip install -U sentence-transformers faiss-cpu')


# In[75]:


from sentence_transformers import SentenceTransformer


# In[76]:


embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# In[77]:


texts = [
    chunk.page_content
    for chunk in final_chunks
]

print("Total texts:", len(texts))


# In[78]:


embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)


# In[79]:


print(type(embeddings))
print(embeddings.shape)


# In[80]:


import faiss
import numpy as np


# In[81]:


embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# In[82]:


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# In[83]:


index.add(embeddings)

print("Vectors stored:", index.ntotal)


# In[84]:


len(final_chunks)


# In[85]:


query = "Haryana mein seating capacity ka rule kya hai?"

query_embedding = embedding_model.encode(
    [query]
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)


# In[86]:


k = 5

distances, indices = index.search(
    query_embedding,
    k
)


# In[87]:


for i, idx in enumerate(indices[0]):

    print("=" * 80)
    print("Rank:", i + 1)
    print("Distance:", distances[0][i])
    print("Metadata:", final_chunks[idx].metadata)
    print()
    print(final_chunks[idx].page_content[:500])


# In[88]:


haryana_indices = [
    i
    for i, chunk in enumerate(final_chunks)
    if chunk.metadata.get("state") == "Haryana"
]

print("Haryana chunks:", len(haryana_indices))


# In[89]:


up_indices = [
    i
    for i, chunk in enumerate(final_chunks)
    if chunk.metadata.get("state") == "Uttar Pradesh"
]

print("UP chunks:", len(up_indices))


# In[90]:


haryana_embeddings = np.asarray(
    [
        embeddings[i]
        for i in haryana_indices
    ],
    dtype="float32"
)

haryana_index = faiss.IndexFlatL2(
    haryana_embeddings.shape[1]
)

haryana_index.add(haryana_embeddings)

print("Haryana vectors:", haryana_index.ntotal)


# In[91]:


query = "Haryana mein seating capacity ka rule kya hai?"

query_embedding = embedding_model.encode(
    [query]
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)

k = 5

distances, local_indices = haryana_index.search(
    query_embedding,
    k
)


# In[92]:


for rank, local_idx in enumerate(local_indices[0]):

    original_idx = haryana_indices[local_idx]

    chunk = final_chunks[original_idx]

    print("=" * 80)
    print("Rank:", rank + 1)
    print("Distance:", distances[0][rank])
    print("Metadata:", chunk.metadata)
    print()
    print(chunk.page_content[:500])


# In[93]:


original_idx = haryana_indices[local_idx]


# In[94]:


get_ipython().system('pip install rank-bm25')


# In[95]:


from rank_bm25 import BM25Okapi


# In[96]:


haryana_texts = [
    final_chunks[i].page_content
    for i in haryana_indices
]


# In[97]:


haryana_tokens = [
    text.lower().split()
    for text in haryana_texts
]


# In[98]:


bm25 = BM25Okapi(haryana_tokens)


# In[99]:


query = "Haryana mein seating capacity ka rule kya hai?"

query_tokens = query.lower().split()

bm25_scores = bm25.get_scores(query_tokens)


# In[100]:


top_bm25 = np.argsort(bm25_scores)[::-1][:5]


# In[101]:


for rank, local_idx in enumerate(top_bm25):

    original_idx = haryana_indices[local_idx]

    chunk = final_chunks[original_idx]

    print("=" * 80)
    print("Rank:", rank + 1)
    print("BM25 Score:", bm25_scores[local_idx])
    print("Metadata:", chunk.metadata)
    print()
    print(chunk.page_content[:500])


# In[102]:


k = 20

vector_distances, vector_local_indices = haryana_index.search(
    query_embedding,
    k
)


# In[103]:


bm25_top_k = 20

bm25_top_indices = np.argsort(
    bm25_scores
)[::-1][:bm25_top_k]


# In[104]:


def reciprocal_rank_fusion(
    vector_indices,
    bm25_indices,
    k=60
):

    scores = {}

    # Vector rankings
    for rank, idx in enumerate(vector_indices, start=1):

        scores[idx] = scores.get(idx, 0) + (
            1 / (k + rank)
        )

    # BM25 rankings
    for rank, idx in enumerate(bm25_indices, start=1):

        scores[idx] = scores.get(idx, 0) + (
            1 / (k + rank)
        )

    # Sort by combined score
    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results


# In[105]:


hybrid_results = reciprocal_rank_fusion(
    vector_local_indices[0],
    bm25_top_indices
)


# In[106]:


for rank, (local_idx, score) in enumerate(
    hybrid_results[:10],
    start=1
):

    original_idx = haryana_indices[local_idx]

    chunk = final_chunks[original_idx]

    print("=" * 80)
    print("Rank:", rank)
    print("RRF Score:", score)
    print("Metadata:", chunk.metadata)
    print()
    print(chunk.page_content[:400])


# In[107]:


vector_scores = 1 / (1 + vector_distances[0])

vector_scores = (
    vector_scores - vector_scores.min()
) / (
    vector_scores.max() - vector_scores.min() + 1e-8
)


# In[108]:


bm25_candidate_scores = bm25_scores[bm25_top_indices]

bm25_normalized = (
    bm25_candidate_scores - bm25_candidate_scores.min()
) / (
    bm25_candidate_scores.max() -
    bm25_candidate_scores.min() +
    1e-8
)


# In[110]:


hybrid_scores = {}

for i, local_idx in enumerate(vector_local_indices[0]):

    hybrid_scores[local_idx] = (
        hybrid_scores.get(local_idx, 0)
        + 0.7 * vector_scores[i]
    )


# In[111]:


for i, local_idx in enumerate(bm25_top_indices):

    hybrid_scores[local_idx] = (
        hybrid_scores.get(local_idx, 0)
        + 0.3 * bm25_normalized[i]
    )


# In[112]:


hybrid_results = sorted(
    hybrid_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


# In[113]:


for rank, (local_idx, score) in enumerate(
    hybrid_results[:10],
    start=1
):

    original_idx = haryana_indices[local_idx]

    chunk = final_chunks[original_idx]

    print("=" * 80)
    print("Rank:", rank)
    print("Hybrid Score:", round(score, 4))
    print("Metadata:", chunk.metadata)
    print()
    print(chunk.page_content[:400])


# In[114]:


def hybrid_search(
    query,
    k_vector=20,
    k_bm25=20,
    top_k=5,
    vector_weight=0.7,
    bm25_weight=0.3
):

    # -------------------------
    # 1. Query embedding
    # -------------------------

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # -------------------------
    # 2. Vector search
    # -------------------------

    vector_distances, vector_local_indices = (
        haryana_index.search(
            query_embedding,
            k_vector
        )
    )

    vector_distances = vector_distances[0]
    vector_local_indices = vector_local_indices[0]

    # Convert distance → similarity
    vector_scores = 1 / (1 + vector_distances)

    # Normalize
    vector_scores = (
        vector_scores - vector_scores.min()
    ) / (
        vector_scores.max()
        - vector_scores.min()
        + 1e-8
    )

    # -------------------------
    # 3. BM25 search
    # -------------------------

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_top_indices = np.argsort(
        bm25_scores
    )[::-1][:k_bm25]

    bm25_candidate_scores = (
        bm25_scores[bm25_top_indices]
    )

    # Normalize BM25
    bm25_normalized = (
        bm25_candidate_scores
        - bm25_candidate_scores.min()
    ) / (
        bm25_candidate_scores.max()
        - bm25_candidate_scores.min()
        + 1e-8
    )

    # -------------------------
    # 4. Hybrid scoring
    # -------------------------

    hybrid_scores = {}

    # Vector contribution
    for i, local_idx in enumerate(
        vector_local_indices
    ):

        hybrid_scores[local_idx] = (
            hybrid_scores.get(local_idx, 0)
            + vector_weight * vector_scores[i]
        )

    # BM25 contribution
    for i, local_idx in enumerate(
        bm25_top_indices
    ):

        hybrid_scores[local_idx] = (
            hybrid_scores.get(local_idx, 0)
            + bm25_weight * bm25_normalized[i]
        )

    # -------------------------
    # 5. Sort
    # -------------------------

    ranked_results = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------
    # 6. Return top K
    # -------------------------

    results = []

    for local_idx, score in ranked_results[:top_k]:

        original_idx = haryana_indices[
            local_idx
        ]

        chunk = final_chunks[
            original_idx
        ]

        results.append({
            "score": float(score),
            "document": chunk
        })

    return results


# In[115]:


results = hybrid_search(
    "Haryana mein seating capacity ka rule kya hai?"
)


# In[116]:


for i, result in enumerate(results, start=1):

    print("=" * 80)
    print("Rank:", i)
    print("Score:", result["score"])
    print("Metadata:", result["document"].metadata)
    print()
    print(result["document"].page_content[:500])


# In[117]:


def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        doc = result["document"]

        metadata = doc.metadata

        source = (
            f"{metadata['document_name']}, "
            f"{metadata['state']}, "
            f"Rule {metadata['rule']}, "
            f"Pages {metadata['page_start']}-"
            f"{metadata['page_end']}"
        )

        context_parts.append(
            f"""
SOURCE {i}
{source}

{doc.page_content}
"""
        )

    return "\n".join(context_parts)


# In[118]:


context = build_context(results)

print(context)


# In[119]:


get_ipython().system('pip install ollama')


# In[120]:


import ollama


# In[121]:


def create_prompt(question, context):

    prompt = f"""
You are NyayaDrive, an Indian Motor Vehicle Law Assistant.

Answer the user's question using ONLY the legal sources
provided below.

Rules:

1. Do not invent or assume legal provisions.
2. Do not use outside legal knowledge.
3. If the sources are insufficient, clearly say so.
4. Explain the answer in simple language.
5. Mention the relevant Rule/Section.
6. Always provide the source citation.
7. If multiple provisions are relevant, distinguish them clearly.

LEGAL SOURCES:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return prompt


# In[122]:


question = "Haryana mein seating capacity ka rule kya hai?"

prompt = create_prompt(
    question,
    context
)

print(prompt)


# In[123]:


def format_citation(metadata):
    return (
        f"{metadata['document_name']} | "
        f"{metadata['state']} | "
        f"Rule {metadata['rule']} | "
        f"Pages {metadata['page_start']}-{metadata['page_end']}"
    )


# In[124]:


metadata = results[0]["document"].metadata

print(format_citation(metadata))


# In[125]:


def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        doc = result["document"]
        metadata = doc.metadata

        citation = format_citation(metadata)

        context_parts.append(
            f"""
SOURCE [{i}]
Citation: {citation}

Legal Text:
{doc.page_content}
"""
        )

    return "\n".join(context_parts)


# In[126]:


def create_prompt(question, context):

    return f"""
You are NyayaDrive, an Indian Motor Vehicle Law Assistant.

Answer the user's question ONLY using the legal sources
provided below.

Rules:

1. Do not use outside legal knowledge.
2. Do not invent legal provisions.
3. If the sources are insufficient, say so clearly.
4. Explain the answer in simple language.
5. Mention the relevant Rule or Section.
6. Every important legal claim must have a source citation.
7. Use the SOURCE number provided in the context.
8. Never create or modify a citation.
9. If multiple sources are relevant, distinguish them.
10. If a source is only partially relevant, do not use it.

Citation format:

[Source 1]
[Source 2]

LEGAL SOURCES:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# In[127]:


final_response = {
    "answer": llm_answer,
    "sources": [
        {
            "source_id": i + 1,
            "document": result["document"].metadata["document_name"],
            "state": result["document"].metadata["state"],
            "rule": result["document"].metadata.get("rule"),
            "chapter": result["document"].metadata.get("chapter"),
            "page_start": result["document"].metadata.get("page_start"),
            "page_end": result["document"].metadata.get("page_end")
        }
        for i, result in enumerate(results)
    ]
}

