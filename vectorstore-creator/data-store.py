import torch
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import os


def knowleadgebase_create(folder_path):
    file_reader = DirectoryLoader(folder_path, glob="**/*.txt")
    documents = file_reader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, chunk_overlap=1600, separators=["\n\n", "\n", " ", "", "<page number"])
    docs = text_splitter.split_documents(documents)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})

    knowledgeBase = FAISS.from_documents(docs, embeddings)

    knowledgeBase.save_local(DB_FAISS_PATH)

    return knowledgeBase


data_source = "../Extracted-text-data-v3"
leaf_folders = os.listdir(data_source)
print(leaf_folders)

count = 0
for foldername in leaf_folders:
    DB_FAISS_PATH = os.path.join("all-data-vectorstore-8000", foldername)

    data_folder_path = f"../Extracted-text-data-v3/{foldername}"

    knowledgeBase = knowleadgebase_create(data_folder_path)

    count += 1
    print(f'{count} - vectorstores created')

print('Finished - data vectorstore')
