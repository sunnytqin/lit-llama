import json
import re
import time
import os

from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

from wikidata.templates import TEMPLATES


# # WIKIDATA_DUMP_PATH = "wikidata.json"
PATH_PREFIX = "/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/"
WIKIDATA_DUMP_PATH =  os.path.join(PATH_PREFIX, "wikidata-20231009-all.json.bz2")

labels = {}
data = {}

wjd = WikidataJsonDump(WIKIDATA_DUMP_PATH)
chunk_file_names = wjd.create_chunks(num_lines_per_chunk=10_000_000)
print(chunk_file_names)
quit()
for i, entity_dict in enumerate(wjd):
    if(entity_dict["type"] != "item"):
        continue
    if i % 1_000_000 == 0:
        print(i)
    pass
quit()

print(i)
for i, entity_dict in enumerate(wjd):
    if(i > 0 and i % 100 == 0):
        print(f"{i} entries processed...", flush=True)
        print(data)
        break
        if i > 20_000_000:
            break

    if(entity_dict["type"] != "item"):
        continue

    entity = WikidataItem(entity_dict)
    print(i, entity.entity_id, entity.get_label(), end=' ')

    # We'll use these to decode IDs later
    labels[entity.entity_id] = entity.get_label()
    
    for property in TEMPLATES:
        dict = data.setdefault(property, {})

        claim_group = entity.get_truthy_claim_group(property)
        for claim in claim_group:
            if(claim.mainsnak.snaktype != "value"):
                continue
            print(property, end= " ")

            val = claim.mainsnak.datavalue.value["id"]

            l = dict.setdefault(entity.entity_id, [])
            l.append(val)
    print("\n")

# label_path = os.path.join(PATH_PREFIX, "labels.json")
# with open(label_path, "w") as fp:
#     json.dump(labels, fp)

# data_path = os.path.join(PATH_PREFIX, "data.json")
# with open(data_path, "w") as fp:
#     json.dump(data, fp)

# label_path = os.path.join(PATH_PREFIX, "labels.json")
# with open(label_path, "r") as fp:
#     labels = json.load(fp)

# data_path = os.path.join(PATH_PREFIX, "data.json")
# with open(data_path, "r") as fp:
#     data = json.load(fp)

# Avoid obscure duplicate labels by choosing entries with the smallest IDs
min_ids = {}
print("labels", labels)
for id, label in labels.items():
    # IDs are of the form Q#
    id_number = int(id[1:])
    cur_min = int(min_ids.get(label, id)[1:])
    print(id, label, id_number, cur_min)
    if(id_number <= cur_min):
        min_ids[label] = id
print("labels", labels)

labels = {k:v for k,v in labels.items() if min_ids[v] == k}
for property in data:
    dict = data[property]
    for k in list(dict.keys()):
        for v in dict[k]:
            if(
                k not in labels or 
                v not in labels
            ):
                dict.pop(k)
                break
print("avoid duplicates")
for k, d in data.items():
    print(len(d), end=' ')
# Remove confusing pairs where the key is the same as the value ("Germany is in Germany")
for property in data:
    dict = data[property]
    for k in list(dict.keys()):
        for v in dict[k]:
            if(labels[k] == labels[v]):
                dict.pop(k)
                break

# Remove keys containing the value ("The culture of the Maldives originated in the Maldives")
for property in data:
    dict = data[property]
    for k in list(dict.keys()):
        for v in dict[k]:
            if(labels[v] in labels[k]):
                dict.pop(k)
                break
print("avoid easy")
for k, d in data.items():
    print(len(d), end=' ')
# Remove entries containing dates ("The 2003 Winter Olympics")
for property in data:
    dict = data[property]
    year_regex = lambda s: re.match(r"^[0-9]{4}$", s)
    for k in list(dict.keys()):
        if(year_regex(labels[k])):
            dict.pop(k)
            break
        for v in dict[k]:
            if(year_regex(labels[v])):
                dict.pop(k)
                break

for property in data:
    dict = data[property]
    temp = {}
    for k in dict:
        temp[labels[k]] = []
        for v in dict[k]:
            temp[labels[k]].append(labels[v])

    data[property] = temp

print("final")
for k, d in data.items():
    print(len(d), end=' ')



data_plaintext_path = os.path.join(PATH_PREFIX, "data_plaintext.json")
with open(data_plaintext_path, "w") as fp:
    json.dump(data, fp)


