import json
from wikidata.templates import TEMPLATES
import os

PATH_PREFIX = "/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/"
data_pairs_path = "data_plaintext.json"
COLLISION_THRESHOLD = 5

with open(data_pairs_path, "r") as fp:
    d = json.load(fp)

new = {}
for wikidata_category, pairs in d.items():
    if(not TEMPLATES[wikidata_category]["type"] == "many_to_one"):
        continue

    collisions = {}
    for _,bl in pairs.items():
        for b in bl:
            collisions[b] = collisions.get(b, 0) + 1

    filtered = {}
    for k, bl in pairs.items():
        new_bl = []
        for b in bl:
            if(collisions[b] >= COLLISION_THRESHOLD):
                new_bl.append(b)

        if(len(new_bl) > 0):
            filtered[k] = new_bl

    new[wikidata_category] = filtered

filter_data_path = os.path.join(PATH_PREFIX, "filtered_data_plaintext.json")
with open(filter_data_path, "w") as fp:
    json.dump(new, fp)