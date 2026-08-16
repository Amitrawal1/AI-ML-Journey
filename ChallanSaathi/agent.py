import os
from langchain_community.document_loaders import PyPDFLoader


def load_all_pdfs():
    folder_path = "data/raw"

    num_docs = 0
    all_docs = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            # Complete file path
            pdf_path = os.path.join(folder_path, filename)

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Add pages to main list
            all_docs.extend(docs)

            num_docs += 1

            print(f"Loaded: {filename}")
            print(f"Pages: {len(docs)}")

    print("\nTotal PDFs:", num_docs)
    print("Total pages:", len(all_docs))

    return all_docs

docs = load_all_pdfs()