import json

with open('deployment-v2/test-queries.json', 'r', encoding='utf-8') as infile:
    data = json.load(infile)

no_id_data = {}
for k in data.keys():
    no_id_data[k] = []
    for it in data[k]:
        if 'id' in it:
            del it['id']
            no_id_data[k].append(it)

        else:
            no_id_data[k].append(it)

idx = 0
for k in no_id_data.keys():
    for it in no_id_data[k]:
        it["id"] = idx
        idx += 1
print(no_id_data)

with open('deployment-v2/test-queries.json', 'w', encoding='utf-8') as outfile:
    json.dump(no_id_data, outfile)
