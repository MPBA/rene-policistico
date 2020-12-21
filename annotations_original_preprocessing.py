#!/usr/bin/env python
# coding: utf-8

# # Annotation files preprocessing

import os
import json
from pathlib import Path

ROOT = Path(os.getcwd())
DATE = "settembre_2020"

images_path = ROOT / "images_original" / DATE
annotations_path = ROOT / "annotations_original" / DATE


for json_ann in os.listdir(annotations_path):
    if json_ann.endswith("json"):
        print(json_ann)
        with open(os.path.join(annotations_path, json_ann), "r") as f:
            s = f.read()
            s = s.replace("\t", "")
            s = s.replace("\n", "")
            s = s.replace(",}", "}")
            s = s.replace(",]", "]")
            data = json.loads(s)
            shapes = data["shapes"]
            name = os.path.splitext(data["imagePath"].split("\\")[-1])[0]

            PROCESSED_PATH = ROOT / "annotations" / DATE
            os.makedirs(PROCESSED_PATH, exist_ok=True)

            with open(PROCESSED_PATH / f"{name}.json", "w") as outfile:
                json.dump(data["shapes"], outfile)
                print(f"copied in {PROCESSED_PATH}/{name}.json!")
                print("\n")




