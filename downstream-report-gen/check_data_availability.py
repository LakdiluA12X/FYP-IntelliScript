from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
import nltk
import json
import os

keyword_vectorstore = '../filter-documents/filtered-docs-keywords-vectorstore'
data_vectorstore = '../filter-documents/keyword-wise-filtered-docs-vectorstore-s-4000'

embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device': 'cpu'})

# Download NLTK resources (run this only once)
nltk.download('punkt')
nltk.download('stopwords')

# List of stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add('page')
stop_words.add('number')
stop_words.add('http')

kw_kbase = FAISS.load_local(
    keyword_vectorstore, embeddings, allow_dangerous_deserialization=True)
data_kbases = {}
for folder in os.listdir(data_vectorstore):
    data_kbases[folder] = FAISS.load_local(os.path.join(
        data_vectorstore, folder), embeddings, allow_dangerous_deserialization=True)

device = "cuda"  # the device to load the model onto
model_name = "Qwen/Qwen1.5-1.8B-Chat"
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
)
qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)


def qwen_response(system_prompt, user_content, temp, tokens):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = qwen_model.generate(
        model_inputs.input_ids,
        max_new_tokens=tokens,
        temperature=temp,
        do_sample=True,
        top_k=10,
        top_p=0.9,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = qwen_tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

    return response


def retriever(knowledgeBase, query, k):
    retriever_ = knowledgeBase.as_retriever(search_kwargs={"k": k})

    docs = retriever_.invoke(query)
    return docs


def get_keywords(text):
    # Tokenize text into words
    words = nltk.word_tokenize(text.lower())

    # Filter out stop words and non-significant words
    keywords = [word for word in words if word not in stop_words]

    return ' '.join(keywords)


def get_num_tokens(text_chunk):
    tokenized_text = qwen_tokenizer(text_chunk, return_tensors='pt')
    num_tokens = tokenized_text['input_ids'].shape[1]
    return num_tokens


def separate_search_filter(system_prompt, query_tuple, kbase, temp, tokens):
    kk = 2
    kb = 100
    matched_count = 0
    query_ins, query, year = query_tuple
    year = str(year)

    keyword_matches = retriever(kw_kbase, get_keywords(query), kk)
    match_metadata = {"keywords": [], "kbase": []}
    used_chunks = []
    output_list = []

    for domain in keyword_matches:
        data_folder = domain.metadata['source'].split('/')[-1].split('.')[0]
        match_metadata['keywords'].append(data_folder)

        doc_matches = retriever(kbase[data_folder], query, kb)
        for n in range(2):
            full_context = ""
            for dm in doc_matches:
                metadata_ = dm.metadata['source'].split('/')
                if year in dm.metadata['source'] or year in dm.page_content:
                    if dm.page_content not in used_chunks:
                        match_metadata['kbase'].append(dm.metadata['source'])

                        metadata_write = '/'.join(metadata_[-2:])
                        full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'
                        used_chunks.append(dm.page_content)
                        # print(dm.page_content)
                        # print('------------------------------------------------------\n\n\n')
                        matched_count += 1
                        if matched_count == 3:
                            matched_count = 0
                            break

            user_content = f'CONTEXT: {full_context}\n\n\n\nQUERY: {query_ins+query}'
            output = qwen_response(system_prompt, user_content, temp, tokens)
            output_list.append(output)

    return output_list, match_metadata


system_prompt_extract_data = """INSTRUCTIONS: You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. 
Filter the data relavant to the specified topic from the Context and output in the requested format.

ADDITINAL_DATA: The current month and year are April 2024.
"""

query_instruct = """You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. 

IMPORTANT INSTRUCTIONS:
1. Do not consider the data that is not present in the Context for Output generation.
2. Do not add any extra text like Explanation in the output. Return only the requested JSON.

Include the extracted value in a JSON like below,
{
    "interest rate": 
}
"""

for y in range(2020, 2021):
    user_query = f"interest rate value in year {y}"
    output, metadata = separate_search_filter(
        system_prompt_extract_data, (query_instruct, user_query, y), data_kbases, temp=0.8, tokens=2000)

    print(y)
    print(output)
print(metadata)
