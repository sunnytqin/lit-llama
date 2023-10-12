import json

from templates import TEMPLATES

data_pairs_path = "filtered_data_plaintext.json"

with open(data_pairs_path, "r") as fp:
    d = json.load(fp)


for wikidata_category, pairs in d.items():
    collisions = {}
    for _,bl in pairs.items():
        for b in bl:
            collisions[b] = collisions.get(b, 0) + 1

    print(f"Category: {wikidata_category} ({TEMPLATES[wikidata_category]['name']})")
    print(f"Collision mean: {sum(collisions.values()) / len(collisions)}")
    print(f"Collision median: {sorted(collisions.values())[len(collisions) // 2]}")
