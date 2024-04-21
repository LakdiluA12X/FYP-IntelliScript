import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

individual_vectorstores = 'all-data-vectorstore-4000'
merged_vectorstores = '../final-codes/data-combined-4000'

embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cpu'})

found = 0
count = 0
for folder in os.listdir(individual_vectorstores):
    full_path = os.path.join(individual_vectorstores, folder)
    if found == 0:
        knowledgeBase_1 = FAISS.load_local(
            full_path, embeddings, allow_dangerous_deserialization=True)
        found = 1

    else:
        knowledgeBase_2 = FAISS.load_local(
            full_path, embeddings, allow_dangerous_deserialization=True)
        knowledgeBase_1.merge_from(knowledgeBase_2)

        count += 1

    print(count)

DB_FAISS_PATH = merged_vectorstores
knowledgeBase_1.save_local(DB_FAISS_PATH)

print(f'Merged vectorstores')
