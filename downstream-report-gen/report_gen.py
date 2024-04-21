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


def split_and_combine_text(text, seperator, chunk_size):
    chunks = []
    splitted_texts = text.split(seperator)
    temp = ''
    for st in splitted_texts:
        temp += st
        if get_num_tokens(temp) > chunk_size:
            chunks.append(temp)
            temp = ''

    if temp != '':
        chunks.append(temp)

    return chunks


def separate_search_filter(system_prompt, query_tuple, kbase, temp, tokens):
    kk = 2
    kb = 50
    matched_count = 0
    query_ins, query, year = query_tuple
    year = str(year)

    keyword_matches = retriever(kw_kbase, get_keywords(query), kk)
    full_context = ""
    match_metadata = {"keywords": [], "kbase": []}

    for domain in keyword_matches:
        data_folder = domain.metadata['source'].split('/')[-1].split('.')[0]
        match_metadata['keywords'].append(data_folder)

        doc_matches = retriever(kbase[data_folder], query, kb)
        for dm in doc_matches:
            metadata_ = dm.metadata['source'].split('/')
            if year in dm.metadata['source'] or year in dm.page_content:
                match_metadata['kbase'].append(dm.metadata['source'])

                metadata_write = '/'.join(metadata_[-2:])
                full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'
                # print(dm.page_content)
                # print('------------------------------------------------------\n\n\n')
                matched_count += 1
                if matched_count == 3:
                    matched_count = 0
                    break

    splitted_chunks = split_and_combine_text(full_context, '<chunk sep>', 4000)
    collected_data = ''

    for sch in splitted_chunks:
        user_content = f'CONTEXT: {sch}\n\n\n\nQUERY: {query}'
        output = qwen_response(
            system_prompt, f'{query_ins}\n\n\n{user_content}', temp, tokens)
        collected_data += output
        collected_data += '\n\n'

    return collected_data, match_metadata


Financial_System_Stability = ['capital adequacy ratio',
                              'net interest margin',
                              'return on assets',
                              'return on equity', 'banking sector in Sri Lanka',
                              'Household and Corporate Sector Analysis',
                              'Sustainable Financing',
                              'Financial Stability Indicators and Maps',
                              'Solvency Stress Testing',
                              'Liquidity Stress Testing',
                              'Systematic Risk Survey',
                              'Periodical risk assessments',
                              'Microfinance',
                              'Money Broking',
                              'Licensed Finance Companies',
                              'Specialised Leasing Companies',
                              'Financial Market Infrastructure']

system_prompt_extract_data = """INSTRUCTIONS: You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. 
Filter the data relavant to the specified topic from the Context and output short summaries of them.

ADDITINAL_DATA: The current month and year are April 2024.
"""

query_instruct_extract_data = """You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. 
Filter the data relavant to the specified topic from the Context and output short summaries of them. The Output needs to be short. Do not include unwanted explanatoins.

IMPORTANT INSTRUCTIONS:
1. Do not consider the data that is not present in the Context for Output generation.
2. The given Context is a combination of multiple text chunks from different sources. These text chunks are seperated by using '<chunk sep>' characters.

"""

all_matched_data = {}

for yr in range(2020, 2022):
    for topic in Financial_System_Stability[:1]:
        user_query = f"{topic} in year {yr}"
        output, metadata = separate_search_filter(
            system_prompt_extract_data, (query_instruct_extract_data, user_query, yr), data_kbases, temp=0.8, tokens=4000)
        all_matched_data[topic] = [
            f'Data relevant to the year {yr}\n\n\n{output}', metadata]
        print(yr, topic)


# system_prompt_report_gen = """INSTRUCTIONS: You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. Organize the given data to design a report.
# The output needs to be short and summarized.
# """

# query_instruct_report_gen = """You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. Organize the given data to design a report.

# IMPORTANT INSTRUCTIONS:
# 1. Do not consider the data that is not present in the Context for Output generation.
# 2. When a year is specified in the Query, only consider data that is matched to the specified year in the Context.

# Get relavant data from the context and build the report.

# """


# # report_data = []

# combined_text = '\n\n\n'.join([all_matched_data[i][0]
#                               for i in all_matched_data])
# # for k in all_matched_data:
# user_query = query_instruct_report_gen + \
#     f"CONTEXT: {combined_text}"
# output = qwen_response(system_prompt_report_gen,
#                        user_query, temp=0.7, tokens=6500)
# # report_data.append(output)

# with open('test-report.txt', 'w', encoding='utf-8') as outfile:
#     outfile.write(output)


# query_instruct_table_gen = """You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. Extract and rganize the given data to design a table if possible only.
# If you can create a table, output it in tabulate format.

# If there is no enough data to generate a table, output 'no table'.

# IMPORTANT INSTRUCTIONS:
# 1. Do not consider the data that is not present in the Context for Output generation.
# 2. When a year is specified in the Query, only consider data that is matched to the specified year in the Context.
# 3. Extract the data with correct labels to generate the table.

# Get relavant data from the context and build the report.

# """

# combined_text = '\n\n\n'.join([all_matched_data[i][0]
#                               for i in all_matched_data])
# # for k in all_matched_data:
# user_query = query_instruct_table_gen + \
#     f"CONTEXT: {combined_text}"
# output = qwen_response(system_prompt_report_gen,
#                        user_query, temp=0.7, tokens=3500)
# # report_data.append(output)

# print(output)


system_prompt_report_gen = """INSTRUCTIONS: You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. Organize the given data to generate future business insights.
The output needs to be short and summarized.
"""

query_instruct_report_gen = """You need to consider the data about the Central Bank of Sri Lanka (CBSL) only. Organize the given data to generate future business insights.

IMPORTANT INSTRUCTIONS:
1. Do not consider the data that is not present in the Context for Output generation.
2. When a year is specified in the Query, only consider data that is matched to the specified year in the Context.

Get relavant data from the context and generate the insights for future based on the context.

"""


# report_data = []

combined_text = '\n\n\n'.join([all_matched_data[i][0]
                              for i in all_matched_data])
# for k in all_matched_data:
user_query = query_instruct_report_gen + \
    f"CONTEXT: {combined_text}"
output = qwen_response(system_prompt_report_gen,
                       user_query, temp=0.7, tokens=6500)
# report_data.append(output)

with open('test-insights.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(output)
