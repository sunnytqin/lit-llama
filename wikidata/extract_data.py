import json
import re
import time

from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.utils import dump_entities_to_json

from templates import TEMPLATES


WIKIDATA_DUMP_PATH = "wikidata.json"

# labels = {}
# data = {}

# wjd = WikidataJsonDump(WIKIDATA_DUMP_PATH)
# for i, entity_dict in enumerate(wjd):
#     if(i > 0 and i % 10000 == 0):
#         print(f"{i} entries processed...")

#     if(entity_dict["type"] != "item"):
#         continue

#     entity = WikidataItem(entity_dict)

#     # We'll use these to decode IDs later
#     labels[entity.entity_id] = entity.get_label()
    
#     for property in TEMPLATES:
#         dict = data.setdefault(property, {})

#         claim_group = entity.get_truthy_claim_group(property)
#         for claim in claim_group:
#             if(claim.mainsnak.snaktype != "value"):
#                 continue

#             val = claim.mainsnak.datavalue.value["id"]

#             l = dict.setdefault(entity.entity_id, [])
#             l.append(val)

# with open("labels.json", "w") as fp:
#     json.dump(labels, fp)

# with open("data.json", "w") as fp:
#     json.dump(data, fp)

with open("labels.json", "r") as fp:
    labels = json.load(fp)

with open("data.json", "r") as fp:
    data = json.load(fp)

# Avoid obscure duplicate labels by choosing entries with the smallest IDs
min_ids = {}
for id, label in labels.items():
    # IDs are of the form Q#
    id_number = int(id[1:])
    cur_min = int(min_ids.get(label, id)[1:])
    if(id_number <= cur_min):
        min_ids[label] = id

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
    print(property)
    dict = data[property]
    temp = {}
    for k in dict:
        temp[labels[k]] = []
        for v in dict[k]:
            temp[labels[k]].append(labels[v])

    data[property] = temp

with open("data_plaintext.json", "w") as fp:
    json.dump(data, fp)


