from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_chains import create_embeddings2
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import yaml
import os
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def add_documents_to_db(uploaded_pdfs, output_folder=config["vectordb_path"], temp_pdf_folder=config["temp_pdf_path"]):
    os.makedirs(temp_pdf_folder, exist_ok=True)  # Create temp folder if it doesn't exist

    all_documents = []

    # Step 1: Save each uploaded PDF temporarily
    for i, pdf_bytes in enumerate(uploaded_pdfs):
        temp_pdf_path = os.path.join(temp_pdf_folder, f"temp_{i}.pdf")
        with open(temp_pdf_path, "wb") as temp_file:
            temp_file.write(pdf_bytes.read())  # Save the uploaded PDF bytes

        # Load the PDF using PyPDFDirectoryLoader
        loader = PyPDFDirectoryLoader(temp_pdf_folder)  # Load all PDFs from the temporary folder
        documents = loader.load()
        all_documents.extend(documents)  # Add documents from this PDF to the list

    # Step 2: Split all documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_documents)

    # Step 3: Create the FAISS vector store from all documents
    vectorstore = FAISS.from_documents(final_documents, create_embeddings2())

    # Step 4: Ensure the output folder exists and save the FAISS index
    os.makedirs(output_folder, exist_ok=True)
    index_file_path = os.path.join(output_folder, "faiss_index.index")
    vectorstore.save_local(index_file_path)

    # Step 5: Clean up the temporary PDFs
    for pdf_file in os.listdir(temp_pdf_folder):
        os.remove(os.path.join(temp_pdf_folder, pdf_file))  # Delete each temporary PDF

    return vectorstore




# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from llm_chains import load_vectordb, create_embeddings
# import pypdfium2

# def get_pdf_texts(pdfs_bytes):
#     return [extract_text_from_pdf(pdfs_bytes) for pdf_bytes in pdfs_bytes]

# def extract_text_from_pdf(pdf_bytes):
#     pdf_file = pypdfium2.PdfDocument(pdf_bytes)
#     return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))
    
# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap=50, separators=["\n", "\n\n"])
#     return splitter.split_text(text)

# def get_documents_chunks(text_list):
#     documents = []
#     for text in text_list:
#         for chunk in get_text_chunks(text):
#             documents.append(Document(page_content=chunk))
#     return documents

# def add_documents_to_db(pdfs_bytes):
#     texts = get_pdf_texts(pdfs_bytes)
#     documents = get_documents_chunks(texts)
#     vector_db = load_vectordb(create_embeddings())
#     vector_db.add_documents(documents)