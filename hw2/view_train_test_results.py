import pickle
from pathlib import Path

for file in Path(".").glob("*.p"):
    with open(file, "rb") as f:
        data = pickle.load(f)
        print(f"{file.name}: {data}")
