import os
import json
import argparse
from transformers import AutoTokenizer
import transformers
import torch

# List of tasks for query expansion
TASK_LIST = [
    "arguana",
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress",
    "fiqa",
    "nfcorpus",
    "nq",
    "scidocs",
    "scifact",
    "webis-touche2020",
    "trec-covid"
]

# Different prompt types for query expansion
PROMPTS = [
    ["Write a passage that answers the following query: {query}", "Q2D"],
    ["Write a list of keywords for the following query: {query}", "Q2E"],
    ["Answer the following query:\n{query}\nGive the rationale before answering", "CoT"]
]

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # define if only want to run for specific tasks
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)

    # Define model engine, prompt type, and path where original query file present and new query file to be saved
    parser.add_argument("--engine", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompttype", type=str, default="Q2D")
    parser.add_argument("--datapath", type=str, default="Datasets")

    args = parser.parse_args()
    return args

def main(args):
    # Initialize the model and tokenizer
    model = args.engine
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline("text-generation", model=model, torch_dtype=torch.float16, device_map=0)

    for task in TASK_LIST[args.startid:args.endid]:
        input_file = f"{args.datapath}/{task}/queries.jsonl"
        output_dir = f"{args.datapath}/{task}"

        # Find the index of the selected prompt type
        index = [i for i, x in enumerate(PROMPTS) if x[1] == args.prompttype]

        # Check if given prompt type exists
        try:
            prompt_type = PROMPTS[index[0]]
        except:
            print('Error: Prompt Type Given Does Not Exist!')
            return
        
        print("Start generating", task, flush=True)

        # Load data from the input file
        with open(input_file, encoding='utf8') as fIn:
            loaded_data = [json.loads(line) for line in fIn]
        
        output_file = output_dir + f"/queries_{prompt_type[1]}.jsonl"
        all_responses = []

        # Check if the output file already exists, and load existing responses
        if os.path.exists(output_file):
            with open(output_file, encoding='utf8') as file:
                try:
                    all_responses = [json.loads(line) for line in file]
                except json.JSONDecodeError:
                    pass

        # Generate responses and write to the output file
        with open(output_file, "a", encoding='utf8') as outfile:
            num_line = len(all_responses)
            for line in loaded_data[len(all_responses):]:
                # Construct the prompt based on the selected prompt type
                prompt = prompt_type[0].format(query=line.get("text"))

                # Generate response using the model
                response = pipeline(
                    prompt,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=200,
                )
                
                # Modify the original query with the generated response
                line["text"] = line.get("text") + " " + response[0]['generated_text']

                # Write the modified query to the output file
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
                print('line: ', num_line)
                num_line = num_line + 1

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
