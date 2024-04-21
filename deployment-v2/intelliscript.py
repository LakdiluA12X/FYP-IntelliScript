from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import pandas as pd
import nltk
import json
import os
import random
from report_type_variables import *

keyword_vectorstore = 'vectorstore/keywords-store/keywords-vectorstore'
data_seperate_1000_vectorstore = 'vectorstore/data-stores/all-data-vectorstore'
data_combined_4000_vectorstore = 'vectorstore/data-stores/data-combined-4000'

filtered_keywords_store = 'filtered-vectorstore/keywords-store/filtered-docs-keywords-vectorstore'
filtered_data_store = 'filtered-vectorstore/data-stores/keyword-wise-filtered-docs-vectorstore-s-4000'


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


def load_seperate_vectorstores(folder_path):
    temp = {}
    for folder in os.listdir(folder_path):
        temp[folder] = FAISS.load_local(os.path.join(
            folder_path, folder), embeddings, allow_dangerous_deserialization=True)

    return temp


kw_kbase = FAISS.load_local(
    keyword_vectorstore, embeddings, allow_dangerous_deserialization=True)
data_seperate_1000_kbases = {"size": 1000, "store": load_seperate_vectorstores(
    data_seperate_1000_vectorstore)}
data_combine_4000_kbase = {"size": 4000, "store": FAISS.load_local(
    data_combined_4000_vectorstore, embeddings, allow_dangerous_deserialization=True)}

filtered_kw_kbase = FAISS.load_local(
    filtered_keywords_store, embeddings, allow_dangerous_deserialization=True)
filtered_data_seperate_4000_kbase = load_seperate_vectorstores(
    filtered_data_store)


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


def qwen_response(system_prompt, user_promt, temp, context_len):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_promt}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = qwen_model.generate(
        model_inputs.input_ids,
        max_new_tokens=context_len,
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


def qna_response_generator(query, temp, context_len):
    kk = 5
    kb1 = 3
    kb2 = 4

    keyword_matches = retriever(kw_kbase, get_keywords(query), kk)
    full_context = ""
    match_metadata = {"keywords": [], "kbase_1": [], "kbase_2": []}
    for domain in keyword_matches:
        data_folder = domain.metadata['source'].split('/')[-1].split('.')[0]
        match_metadata['keywords'].append(data_folder)

        doc_matches_1 = retriever(
            data_seperate_1000_kbases['store'][data_folder], query, kb1)
        for dm in doc_matches_1:
            metadata_ = dm.metadata['source'].split('/')
            match_metadata['kbase_1'].append(dm.metadata['source'])

            metadata_write = '/'.join(metadata_[-2:])
            full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    doc_matches_2 = retriever(
        data_combine_4000_kbase['store'], query, kb2)
    for dm in doc_matches_2:
        metadata_ = dm.metadata['source'].split('/')
        match_metadata['kbase_2'].append(dm.metadata['source'])

        metadata_write = '/'.join(metadata_[-2:])
        full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    user_promt = f'{query_prompt_qna}\n\n\nCONTEXT: {full_context}\n\n\nQUERY: {query}'
    return qwen_response(system_prompt_qna, user_promt, temp, context_len), match_metadata


def report_generator(years_list, topics_list, temp, context_len):
    kk = 1
    kb = 50
    matched_count = 0
    full_context = ""
    match_metadata = {"keywords": [], "kbase": []}

    for year in years_list:
        for topic in topics_list:
            year = str(year)
            query = f'{topic} in {year}'

            keyword_matches = retriever(
                filtered_kw_kbase, get_keywords(query), kk)

            for domain in keyword_matches:
                data_folder = domain.metadata['source'].split(
                    '/')[-1].split('.')[0]
                match_metadata['keywords'].append(data_folder)

                doc_matches = retriever(
                    filtered_data_seperate_4000_kbase[data_folder], query, kb)
                for dm in doc_matches:
                    metadata_ = dm.metadata['source'].split('/')
                    if year in dm.metadata['source'] or year in dm.page_content:
                        match_metadata['kbase'].append(dm.metadata['source'])

                        metadata_write = '/'.join(metadata_[-2:])
                        full_context += f'\n<chunk sep>\nsource: {metadata_write}       year:{year}\n\n{dm.page_content}\n\n\n'
                        # print(dm.page_content)
                        # print('------------------------------------------------------\n\n\n')
                        matched_count += 1
                        if matched_count == 2:
                            matched_count = 0
                            break

    splitted_chunks = split_and_combine_text(full_context, '<chunk sep>', 4000)
    collected_data = ''

    for sch in splitted_chunks:
        user_content = f'CONTEXT: {sch}\n\n\n\nQUERY: {query}'
        extracted_data_output = qwen_response(
            system_prompt_extract_data, f'{query_prompt_extract_data}\n\n\n{user_content}', temp, context_len)
        collected_data += extracted_data_output
        collected_data += '\n\n\n'

    report_gen_query = query_prompt_report_gen + \
        f"CONTEXT: {collected_data}"
    report_output = qwen_response(system_prompt_report_gen,
                                  report_gen_query, temp=0.8, context_len=2000)

    return report_output, match_metadata


system_prompt_qna = """INSTRUCTION: Please consider the data about the Central Bank of Sri Lanka (CBSL) only.
    Generate the Output for the given Query only considering the given Context. 
    
    First search and filter out the requierd data from the Context to generate the most accurate answer for the Query. 
    Then combine the collected data in a meaningfull way to get the short and summarized Output response.

    Do not consider the data that is not present in the Context for Output generation.

    If you can not find required data in the Context to generate the Output, truthfully say 'I don't know' as the Output.

    The given Context is a combination of multiple text chunks from different sources. These text chunks are seperated by using '<chunk sep>' characters and the sources are indicated in the Context.
    You required to produce the most accurate response based on the given Context. Don't add unwanted details in the Output. Make the Output short."""

query_prompt_qna = """Generate an accurate Output for the given Query only considering the given Context. Make the prompt shorter and summarized."""


system_prompt_extract_data = """INSTRUCTIONS: Please consider the data about the Central Bank of Sri Lanka (CBSL) only.
Study well and filter the data relavant to the specified topic from the Context. Do not paraphrase the filtered text.

If you can't find any relevant information in the Context, truthfully say 'I don't know'.

ADDITINAL_DATA: The current month and year are April 2024."""

query_prompt_extract_data = """Do not include unwanted explanatoins. Study the given context well before filtering data relevant to the topic.

Please do not consider the data not present in the Context while filtering relevant data to the topic.
Filter the data relavant to the specified topic from the Context. Do not paraphrase the filtered text but label them.

If you can't find any relevant information in the Context, truthfully say 'I don't know'.
If there is a 'source' in the context, keep it in the Output.

"""

system_prompt_report_gen = """INSTRUCTIONS: Please consider the data about the Central Bank of Sri Lanka (CBSL) only.
Organize the given data to design a report. Add a suitable title to the report at the beginning. Do not chaneg the data in the Context.
"""

query_prompt_report_gen = """Get the relevant information from the context and generate the report. Use the data in the given context. Add a suitable title to the report at the beginning.

Pleasse do not consider the data that is not present in the Context for Output generation.
Do not use roman numbers to indicate topics in the report output.

If there are tabular data in the context, keep them as it is in the output report.
Add 'Sources' section at the end of the Output indicating the sources refered from the context.

"""

# system_prompt_table = """"""
# query_prompt_table = """"""

# system_prompt_graph = """"""
# query_prompt_graph = """"""

system_prompt_insight_extract = """INSTRUCTIONS: Please consider the data about the Central Bank of Sri Lanka (CBSL) only.
Filter the data relevenat to the specified topic from the context to generate insights. Do predictions.

"""
query_prompt_insight_extract = """You need to filter the data required to produce insights including predictions for the given topic.

Pleasse do not consider the data that is not present in the Context for Output generation.

"""

system_prompt_insight_gen = """INSTRUCTIONS: Please consider the data about the Central Bank of Sri Lanka (CBSL) only.
Consider the given context and produce business insights according to the topic. Do predictions.

"""
query_prompt_insight_gen = """Consider the given context and produce business insights according to the topic. Do predictions.
Make the output accurate, short, and summarized.

Pay more attention to clarity, relevance, validity, impact, insightfulness, actionability while making the insights.

Pleasse do not consider the data that is not present in the Context for Output generation.

"""

with open('indicator-data.csv', 'r') as infile:
    data_csv = infile.read()


def convert_to_table_json(data):
    content = {"type": "table",
               "content": {
                   "title": "",
                   "header": [],
                   "data": []
               }}
    lines = data.strip().split('\n')
    header = [header.strip() for header in lines[0].split(',')]
    content["content"]["title"] = header

    headers = [header.strip() for header in lines[1].split(',')]
    content["content"]["header"] = headers

    for line in lines[2:]:
        values = [value.strip() for value in line.split(',')]
        content["content"]["data"].append(values)

    return json.dumps(content, indent=4)


def convert_to_graph_json(data):
    graph_dicts = []

    lines = data.strip().split('\n')
    headers = [header.strip() for header in lines[1].split(',')]

    for line in lines[2:]:
        content = {
            "type": "graph",
            "content": {
                "type": "",
                "title": "",
                "x_axis_values": [],
                "y_axis_values": [],
                "x_label": "",
                "y_label": ""
            }
        }
        content["content"]["x_axis_values"] = headers[1:]
        values = [value.strip() for value in line.split(',')]
        content["content"]["y_axis_values"] = values[1:]
        content["content"]["x_label"] = headers[0]
        content["content"]["y_label"] = values[0]
        content["content"]["title"] = f'{values[0]} vs {headers[0]}'
        content["content"]["type"] = random.choice(['line', 'bar'])

        graph_dicts.append(json.dumps(content, indent=4))

    return graph_dicts


fss_tables = convert_to_table_json(data_csv)
fss_graphs = convert_to_graph_json(data_csv)


def convert_report_to_json(report_text):
    report_json = {
        "type": "report",
        "title": "",
        "sections": []
    }
    current_section = None
    current_content = None

    start_iter = 0
    lines = report_text.strip().split('\n')
    if 'title' in lines[0].split(':')[0].lower():
        report_json["title"] = lines[0].split(':')[1].strip()
        start_iter = 1

    for line in lines[start_iter:]:
        if line.strip() == "":
            continue
        elif line.endswith(":"):
            if current_section:
                report_json["sections"].append(current_section)
            current_section = {"topic": line.strip(), "contents": []}
        else:
            current_content = {"type": "paragraph", "content": line.strip()}
            current_section["contents"].append(current_content)

    if current_section:
        report_json["sections"].append(current_section)

    return json.dumps(report_json, indent=4)


def generate_final_report(query, temp, context_len):
    years_list = [2021, 2022]
    category, topics_list = get_financial_report_topics(query)
    if topics_list != []:
        report_data, metadata = report_generator(
            years_list, topics_list, temp, context_len)

        report_json = json.loads(convert_report_to_json(report_data))
        try:
            if category == "Financial_System_Stability":
                table_data = json.loads(fss_tables)
                graph_data = json.loads(random.choice(fss_graphs))

                report_json['sections'][(
                    len(report_json['sections'])//2)-1]['contents'].append(table_data)
                report_json['sections'][(
                    len(report_json['sections'])//2)]['contents'].append(graph_data)

            return report_json

        except Exception as e:
            print(e)

    else:
        return json.loads(empty_report)


def insight_generator(user_query, temp, context_len):
    kk = 2
    kb = 50
    matched_count = 0
    full_context = ""
    match_metadata = {"keywords": [], "kbase": []}
    years_list = [2020, 2021, 2022]

    for year in years_list:
        year = str(year)
        query = f'{user_query} in {year}'

        keyword_matches = retriever(
            filtered_kw_kbase, get_keywords(query), kk)

        for domain in keyword_matches:
            data_folder = domain.metadata['source'].split(
                '/')[-1].split('.')[0]
            match_metadata['keywords'].append(data_folder)

            doc_matches = retriever(
                filtered_data_seperate_4000_kbase[data_folder], query, kb)
            for dm in doc_matches:
                metadata_ = dm.metadata['source'].split('/')
                if year in dm.metadata['source'] or year in dm.page_content:
                    match_metadata['kbase'].append(dm.metadata['source'])

                    metadata_write = '/'.join(metadata_[-2:])
                    full_context += f'\n<chunk sep>\nsource: {metadata_write}       year:{year}\n\n{dm.page_content}\n\n\n'
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
        extracted_data_output = qwen_response(
            system_prompt_insight_extract, f'{query_prompt_insight_extract}\n\n\n{user_content}', temp, context_len)
        collected_data += extracted_data_output
        collected_data += '\n\n\n'

    insight_gen_query = query_prompt_insight_gen + \
        f"CONTEXT: {collected_data}\n\n\nTOPIC: {user_query}"
    insight_output = qwen_response(system_prompt_insight_gen,
                                   insight_gen_query, temp=0.8, context_len=1000)

    output_json = {
        "type": "paragraph",
        "content": insight_output}

    return json.dumps(output_json), match_metadata
