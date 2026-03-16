import argparse
import json
from src.clens import ClauseLens

parser = argparse.ArgumentParser()

#parser for cli arguments
parser.add_argument("-f", "--file", help="document path", required=True)
parser.add_argument("-o", "--output", help="JSON output path", required=True)
args = parser.parse_args()

#loads config file
with open('config/config.json') as f:
    config = json.load(f)

clens = ClauseLens(classifier_path=config["classifier_path"], model_name=config["llm"], policy=config["policy"], chunk_size=config["chunk_size"])

# run full pipeline
with open(args.file, 'rb') as f:
    doctype = args.file.split('.')[-1]
    output = clens.run(f, doctype)

#save output
with open(args.output, 'w') as f:
    json.dump(output, f)