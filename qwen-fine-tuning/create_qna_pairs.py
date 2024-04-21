import json
import os

all_data = []
for root, directories, files in os.walk('qwen-fine-tuning\qa-dataset'):
    for file in files:
        filepath = os.path.join(root, file)
        with open(filepath, 'r', encoding='utf-8') as datafile:
            data = datafile.read().split('\n\n')

        for dp in data:
            data_temp = {}
            data_lines = dp.split('\n')
            if 'Question: ' in data_lines[0]:
                data_temp["query"] = data_lines[0].split('Question: ')[1]
                data_temp["response"] = data_lines[1].split('Answer: ')[1]

                all_data.append(data_temp)


with open('qna.json', 'w', encoding='utf-8') as outfile:
    json.dump(all_data, outfile)
