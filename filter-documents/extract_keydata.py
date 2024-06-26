import os
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_llama2_quantized(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        # revision='834565c23f9b28b96ccbeabe614dd906b6db551a'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_document_chunks(file_path):
    file_reader = TextLoader(file_path)
    documents = file_reader.load()

    node_parser = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=750, separators=["\n\n", "\n", " ", "", "<page sep>"])
    nodes = node_parser.split_documents(documents)

    return nodes


def generate_summary_prompt(topic, query):
    text = """INSTRUCTION: Extract the data about the given TOPIC from the given CONTEXT. 
    Extract data that only matches to the TOPIC. 
    Do not miss any important data about the TOPIC.

        1. Extract special dates relevant to the topic
        2. Extract values relevant to the topic
        3. Extract facts relevant to the topic
        4. Extract any other data relevant to the topic

    Do not add NOTE: or EXPLANATION: at the end of the generation.

    """
    text += f'TOPIC: \n{topic}\n\n'
    text += f'CONTEXT: \n{query}\n\n'
    text += 'OUTPUT: '
    return {'text': text}


def get_summarized(topic, query):
    prompt = generate_summary_prompt(topic, query)['text']
    output_response = llama2_pipeline(prompt)[0]['generated_text']
    output_text = output_response.split('\nOUTPUT:')[1]
    return output_text


def count_starting_spaces(line):
    """Count the number of starting spaces in a line."""
    return len(line) - len(line.lstrip())


def remove_initial_spaces(chunk):
    space_count = []
    datalines = chunk.split('\n')
    for line in datalines:
        if line != '' and line[0] != '<':
            space_count.append(count_starting_spaces(line))

    cleaned_data = []
    remove_spaces = min(space_count)
    for line_ in datalines:
        if line_ != '' and line_[0] != '<':
            cleaned_data.append(line_[remove_spaces:].rstrip())

        else:
            cleaned_data.append(line_.rstrip())

    return '\n'.join(cleaned_data)


llama2, llama2_tokenizer = load_llama2_quantized(
    'NousResearch/Llama-2-7b-chat-hf')

llama2_kwargs = {
    "max_new_tokens": 2000,
    "temperature": 0.75,
    # "num_return_sequences": 1,
    "top_k": 10,
    "top_p": 0.9,
    "batch_size": 1,
    "do_sample": True,
    # "no_repeat_ngram_size":4,
    # "repetition_penalty" : 1.5
}

llama2_pipeline = pipeline(
    "text-generation",
    model=llama2,
    tokenizer=llama2_tokenizer,
    **llama2_kwargs
)

# 'interest rate',

keywords = ['inflation rate', 'bank rate', 'unemployment']
for kw in keywords:
    source_dir = kw
    output_dir = os.path.join('llm-outputs', f'extracted-{kw}')
    count = 0
    error_count = 0

    for root, directories, files in os.walk(source_dir):
        for directory in directories:
            # Create corresponding directories in the output directory
            output_subdir = os.path.join(
                output_dir, os.path.relpath(root, source_dir), directory)
            os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            try:
                input_file_path = os.path.join(root, file)
                # Determine the output file path
                output_file_path = os.path.join(
                    output_dir, os.path.relpath(root, source_dir), file)

                if not os.path.exists(output_file_path):
                    with open(input_file_path, 'r', encoding='utf-8') as infile:
                        data = infile.read()

                    chunks = create_document_chunks(input_file_path)
                    chunk_count = 0
                    full_text = ''
                    for chunk in chunks:
                        summary = get_summarized(kw,
                                                 remove_initial_spaces(chunk.page_content))
                        full_text += f'{summary}\n'
                        chunk_count += 1

                        print(f'    {chunk_count} of {len(chunks)} Chunks')

                    with open(output_file_path, 'w', encoding='utf-8') as outfile:
                        outfile.write(
                            f'meta-data: {os.path.join(os.path.relpath(root, source_dir), file)}\ntopic: {kw}\n\n{full_text}')

                    count += 1

                    print(f'File {count} - {input_file_path}')

            except Exception as e:
                print(e)
                error_count += 1

    for root, dirs, files in os.walk(output_dir, topdown=False):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            if not os.listdir(folder_path):  # Check if folder is empty
                os.rmdir(folder_path)
                print(f"Deleted empty folder: {folder_path}")

    print(f"Success: {count} - Errors: {error_count}, {kw}")
