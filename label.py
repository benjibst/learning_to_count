import json
import os

try:
    with open("images/labels.json","r") as f:
        labels = json.load(f)
except FileNotFoundError:
    labels = {}
files = os.listdir("images")
for file in files: 
    if file not in labels:
        labels[file] = ""