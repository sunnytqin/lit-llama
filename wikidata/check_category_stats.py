import os
import json

from wikidata.templates import TEMPLATES

PATH_PREFIX = "/n/holyscratch01/barak_lab/Lab/sqin/hallucination/wikidata/"
data_pairs_path = os.path.join(PATH_PREFIX, "filtered_data_plaintext.json")

with open(data_pairs_path, "r") as fp:
    d = json.load(fp)


for wikidata_category, pairs in d.items():
    print(wikidata_category, len(pairs))
    collisions = {}
    for _,bl in pairs.items():
        for b in bl:
            collisions[b] = collisions.get(b, 0) + 1


    print(f"Category: {wikidata_category} ({TEMPLATES[wikidata_category]['name']})")
    print(f"Collision mean: {sum(collisions.values()) / max(1, len(collisions))}")
    # print(f"Collision median: {sorted(collisions.values())[len(collisions) // 2]}")
