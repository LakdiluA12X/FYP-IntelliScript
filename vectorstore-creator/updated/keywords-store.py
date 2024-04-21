import torch
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader


DB_FAISS_PATH = "keywords-vectorstore-new"


def knowleadgebase_create(folder_path):
    file_reader = DirectoryLoader(folder_path, glob="**/*.txt")
    documents = file_reader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})

    knowledgeBase = FAISS.from_documents(docs, embeddings)

    knowledgeBase.save_local(DB_FAISS_PATH)

    return knowledgeBase


data_folder_path = 'keywords-files-v2'  # folder path to the text files folder
knowledgeBase = knowleadgebase_create(data_folder_path)

print('Finished - keywords vectorstore')
