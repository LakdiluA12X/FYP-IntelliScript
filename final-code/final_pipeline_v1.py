from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from transformers import pipeline
import nltk
import json
import os

keyword_vectorstore = '../final-vectorstores/keywords-vectorstore'
data_seperate_1000_vectorstore = '../final-vectorstores/all-data-vectorstore'
data_seperate_4000_vectorstore = '../final-vectorstores/all-data-vectorstore-4000'

data_combined_1000_vectorstore = '../final-vectorstores/data-combined-1000'
data_combined_4000_vectorstore = '../final-vectorstores/data-combined-4000'

test_qr = 'generated-qrs.json'
response_outputs = 'response-outputs'

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
data_seperate_4000_kbases = {"size": 4000, "store": load_seperate_vectorstores(
    data_seperate_4000_vectorstore)}

data_combine_1000_kbase = {"size": 1000, "store": FAISS.load_local(
    data_combined_1000_vectorstore, embeddings, allow_dangerous_deserialization=True)}
data_combine_4000_kbase = {"size": 4000, "store": FAISS.load_local(
    data_combined_4000_vectorstore, embeddings, allow_dangerous_deserialization=True)}


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


def get_num_tokens(tokenizer, text_chunk):
    tokenized_text = tokenizer(text_chunk, return_tensors='pt')
    num_tokens = tokenized_text['input_ids'].shape[1]
    return num_tokens


'''
-----------------------------------------------------------------------------------------------------------------
'''


# def load_llama2_quantized(model_name):
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True,
#         quantization_config=BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_quant_type="nf4",
#         ),
#         # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
#     )

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     return model, tokenizer


# llama2, llama2_tokenizer = load_llama2_quantized(
#     'NousResearch/Llama-2-7b-chat-hf')

# llama2_pipeline = pipeline(
#     "text-generation",
#     model=llama2,
#     tokenizer=llama2_tokenizer,
#     max_new_tokens=1000,
#     temperature=0.75,
#     # "num_return_sequences": 1,
#     top_k=10,
#     top_p=0.9,
#     batch_size=1,
#     do_sample=True
#     # "no_repeat_ngram_size":4,
#     # "repetition_penalty" : 1.5
# )


'''
-----------------------------------------------------------------------------------------------------------------
'''

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


def qwen_response(query, context):
    system_prompt = """INSTRUCTION: Please consider the data about the Central Bank of Sri Lanka (CBSL) only.
    Generate the Output for the given Query only considering the given Context. 
    
    First search and filter out the requierd data from the Context to generate the most accurate answer for the Query. 
    Then combine the collected data in a meaningfull way to get the short and summarized Output response.

    Do not consider the data that is not present in the Context for Output generation.

    If you can not find required data in the Context to generate the Output, truthfully say 'I don't know' as the Output.

    The given Context is a combination of multiple text chunks from different sources. These text chunks are seperated by using '<chunk sep>' characters and the sources are indicated in the Context.
    You required to produce the most accurate response based on the given Context. Don't add unwanted details in the Output. Make the Output short.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Generate an accurate Output for the given Query only considering the given Context. Make the prompt shorter and summarized.\n\n\nCONTEXT: {context}\n\n\nQUERY: {query}'}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = qwen_model.generate(
        model_inputs.input_ids,
        max_new_tokens=1000,
        temperature=0.85,
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


'''
-----------------------------------------------------------------------------------------------------------------
'''


# def generate_response_prompt(query, context):
#     text = """INSTRUCTION: Generate the Output for the given Query considering the given Context. First search and extract the requierd data from the Context
#     to generate the most accurate answer for the Query. Then combine the collected data in a meaningfull way to get the short and summarized Output response.

#     Do not consider the data that is not present in the Context for Output generation.

#     If you can not find required data in the Context to generate the Output, give 'No data match' as the Output.

#     The given Context is a combination of multiple text chunks from different sources. These text chunks are seperated by using '<chunk sep>' characters.

#     You required to produce the most accurate response based on the given Context. Don't add unwanted details in the Output. Make the Output short.
#     """
#     text += f'QUERY: \n{query}\n\n'
#     text += f'CONTEXT: \n{context}\n\n'
#     text += 'OUTPUT: '
#     return {'text': text}


# def llama2_response(query, context):
#     output_response = llama2_pipeline(generate_response_prompt(query, context)['text'])[
#         0]['generated_text']
#     output_text = output_response.split('\nOUTPUT:')[1]
#     return output_text
#     pass


def separate_search_filter(model, query, kbase_1, kbase_2):
    kk = 0
    kb1 = 0
    kb2 = 0
    if model == 'llama2':
        if kbase_1['size'] == 1000 and kbase_2 == '':
            kk, kb1, kb2 = 3, 3, 0
        elif kbase_1['size'] == 4000 and kbase_2 == '':
            kk, kb1, kb2 = 3, 1, 0
        elif kbase_1['size'] == 1000 and kbase_2['size'] == 4000:
            kk, kb1, kb2 = 2, 3, 1

    else:
        if kbase_1['size'] == 1000 and kbase_2 == '':
            kk, kb1, kb2 = 5, 3, 0
        elif kbase_1['size'] == 4000 and kbase_2 == '':
            kk, kb1, kb2 = 4, 2, 0
        elif kbase_1['size'] == 1000 and kbase_2['size'] == 4000:
            kk, kb1, kb2 = 5, 3, 2

    keyword_matches = retriever(kw_kbase, get_keywords(query), kk)
    full_context = ""
    match_metadata = {"keywords": [], "kbase_1": [], "kbase_2": []}
    for domain in keyword_matches:
        data_folder = domain.metadata['source'].split('/')[-1].split('.')[0]
        match_metadata['keywords'].append(data_folder)

        doc_matches_1 = retriever(kbase_1['store'][data_folder], query, kb1)
        for dm in doc_matches_1:
            metadata_ = dm.metadata['source'].split('/')
            match_metadata['kbase_1'].append(dm.metadata['source'])

            metadata_write = '/'.join(metadata_[-2:])
            full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

        if kbase_2 != '':
            doc_matches_2 = retriever(
                kbase_2['store'][data_folder], query, kb2)
            for dm in doc_matches_2:
                metadata_ = dm.metadata['source'].split('/')
                match_metadata['kbase_2'].append(dm.metadata['source'])

                metadata_write = '/'.join(metadata_[-2:])
                full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    # if model == 'llama2':
    #     return llama2_response(query, full_context), match_metadata
    # else:
    return qwen_response(query, full_context), match_metadata


def combine_search_filter(model, query, kbase_1, kbase_2):
    kk = 0
    kb1 = 0
    kb2 = 0
    if model == 'llama2':
        if kbase_1['size'] == 1000 and kbase_2 == '':
            kk, kb1, kb2 = 0, 8, 0
        elif kbase_1['size'] == 4000 and kbase_2 == '':
            kk, kb1, kb2 = 0, 3, 0
        elif kbase_1['size'] == 1000 and kbase_2['size'] == 4000:
            kk, kb1, kb2 = 0, 4, 2

    else:
        if kbase_1['size'] == 1000 and kbase_2 == '':
            kk, kb1, kb2 = 0, 15, 0
        elif kbase_1['size'] == 4000 and kbase_2 == '':
            kk, kb1, kb2 = 0, 8, 0
        elif kbase_1['size'] == 1000 and kbase_2['size'] == 4000:
            kk, kb1, kb2 = 0, 15, 5

    full_context = ""
    match_metadata = {"kbase_1": [], "kbase_2": []}

    doc_matches_1 = retriever(kbase_1['store'], query, kb1)
    for dm in doc_matches_1:
        metadata_ = dm.metadata['source'].split('/')
        match_metadata['kbase_1'].append(dm.metadata['source'])

        metadata_write = '/'.join(metadata_[-2:])
        full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    if kbase_2 != '':
        doc_matches_2 = retriever(
            kbase_2['store'], query, kb2)
        for dm in doc_matches_2:
            metadata_ = dm.metadata['source'].split('/')
            match_metadata['kbase_2'].append(dm.metadata['source'])

            metadata_write = '/'.join(metadata_[-2:])
            full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    # if model == 'llama2':
    #     return llama2_response(query, full_context), match_metadata
    # else:
    return qwen_response(query, full_context), match_metadata


def mix_search_filter(model, query, kbase_1, kbase_2):
    kk = 0
    kb1 = 0
    kb2 = 0
    if model == 'llama2':
        if kbase_1['size'] == 1000 and kbase_2['size'] == 1000:
            kk, kb1, kb2 = 3, 2, 3

    else:
        if kbase_1['size'] == 1000 and kbase_2['size'] == 1000:
            kk, kb1, kb2 = 5, 3, 10
        elif kbase_1['size'] == 1000 and kbase_2['size'] == 4000:
            kk, kb1, kb2 = 5, 3, 4
        elif kbase_1['size'] == 4000 and kbase_2['size'] == 1000:
            kk, kb1, kb2 = 5, 2, 8
        elif kbase_1['size'] == 4000 and kbase_2['size'] == 4000:
            kk, kb1, kb2 = 4, 2, 3

    keyword_matches = retriever(kw_kbase, get_keywords(query), kk)
    full_context = ""
    match_metadata = {"keywords": [], "kbase_1": [], "kbase_2": []}
    for domain in keyword_matches:
        data_folder = domain.metadata['source'].split('/')[-1].split('.')[0]
        match_metadata['keywords'].append(data_folder)

        doc_matches_1 = retriever(kbase_1['store'][data_folder], query, kb1)
        for dm in doc_matches_1:
            metadata_ = dm.metadata['source'].split('/')
            match_metadata['kbase_1'].append(dm.metadata['source'])

            metadata_write = '/'.join(metadata_[-2:])
            full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    if kbase_2 != '':
        doc_matches_2 = retriever(
            kbase_2['store'], query, kb2)
        for dm in doc_matches_2:
            metadata_ = dm.metadata['source'].split('/')
            match_metadata['kbase_2'].append(dm.metadata['source'])

            metadata_write = '/'.join(metadata_[-2:])
            full_context += f'\n<chunk sep>\nsource: {metadata_write}\n\n{dm.page_content}\n\n\n'

    # if model == 'llama2':
    #     return llama2_response(query, full_context), match_metadata
    # else:
    return qwen_response(query, full_context), match_metadata


with open(test_qr, 'r', encoding='utf-8') as jsonfile:
    qr_data = json.load(jsonfile)


combinations_separate_search = [('qwen', data_seperate_1000_kbases,
                                 '', 'qwen-responses-s-1000.json'),
                                ('qwen', data_seperate_4000_kbases,
                                 '', 'qwen-responses-s-4000.json')]


combinations_combine_search = [('qwen', data_combine_1000_kbase,
                                '', 'qwen-responses-c-1000.json'),
                               ('qwen', data_combine_4000_kbase,
                                '', 'qwen-responses-c-4000.json')]


combinations_mixed_search = [('qwen', data_seperate_1000_kbases, data_combine_1000_kbase,
                              'qwen-responses-s-1000-c-1000.json'),
                             ('qwen', data_seperate_1000_kbases, data_combine_4000_kbase,
                              'qwen-responses-s-1000-c-4000.json')]


for cm in combinations_separate_search:
    if cm[0] == 'qwen':
        responses = {}
        for cat in qr_data:
            if cat not in responses:
                responses[cat] = []
            for qr in qr_data[cat]:
                temp = qr.copy()
                response, mdata = separate_search_filter(
                    cm[0], qr['query'], cm[1], cm[2])
                temp["generated-response"] = response
                temp["generated-metadata"] = mdata
                responses[cat].append(temp)

                print(f'finished - {qr["id"]}')
                # print(qr['response'])
                # print('------------------------------------------------------------')
                # print(response)
                # print('\n\n')

        with open(os.path.join(response_outputs, cm[3]), 'w', encoding='utf-8') as outfile:
            json.dump(responses, outfile)

        print(f'completed - {cm[3]}')


for cm in combinations_combine_search:
    if cm[0] == 'qwen':
        responses = {}
        for cat in qr_data:
            if cat not in responses:
                responses[cat] = []
            for qr in qr_data[cat]:
                temp = qr.copy()
                response, mdata = combine_search_filter(
                    cm[0], qr['query'], cm[1], cm[2])
                temp["generated-response"] = response
                temp["generated-metadata"] = mdata
                responses[cat].append(temp)

                print(f'finished - {qr["id"]}')

        with open(os.path.join(response_outputs, cm[3]), 'w', encoding='utf-8') as outfile:
            json.dump(responses, outfile)

        print(f'completed - {cm[3]}')

for cm in combinations_mixed_search:
    if cm[0] == 'qwen':
        responses = {}
        for cat in qr_data:
            if cat not in responses:
                responses[cat] = []
            for qr in qr_data[cat]:
                temp = qr.copy()
                response, mdata = mix_search_filter(
                    cm[0], qr['query'], cm[1], cm[2])
                temp["generated-response"] = response
                temp["generated-metadata"] = mdata
                responses[cat].append(temp)

                print(f'finished - {qr["id"]}')

        with open(os.path.join(response_outputs, cm[3]), 'w', encoding='utf-8') as outfile:
            json.dump(responses, outfile)

        print(f'completed - {cm[3]}')
