import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

os.environ['TRANSFORMERS_CACHE'] = '/w/284/mahamm/.cache'
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline("text-generation",model=model,torch_dtype=torch.float16,device_map='auto')

TASK_LIST = ["nq"]

PROMPTS = [
    ["Write a passage that answers the following query: {query}", "Q2D"]]

for task in TASK_LIST:
    input_file = f"Datasets/Expanded_Llama/{task}/queries.jsonl"
    output_dir = f"Datasets/Expanded_Llama/{task}"

    print("Start generating", task, flush=True)
    with open(input_file, encoding='utf8') as fIn:
        loaded_data = [json.loads(line) for line in fIn]

    for prompts in PROMPTS:
        output_file = output_dir + f"/queries_{prompts[1]}.jsonl"
        all_responses = []
        if os.path.exists(output_file):
            with open(output_file, encoding='utf8') as file:
                try:
                    all_responses = [json.loads(line) for line in file]
                except json.JSONDecodeError:
                    pass

        with open(output_file, "a", encoding='utf8') as outfile:
            num_line=len(all_responses)
            for line in loaded_data[len(all_responses):]:
                prompt=prompts[0].format(query=line.get("text"))
                response = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=200,
                )
                response=response[0]['generated_text'].split(":",1)
                line["text"] = line.get("text") * 4 + response[1]
                all_responses.append(line)
                json.dump(line, outfile, ensure_ascii=False)
                outfile.write('\n')
                print('line: ',num_line)
                num_line=num_line+1

        # fw = open(os.path.join(output_dir,f""), 'w')
        # cw = csv.writer(fw, delimiter='\t')
        # data = open(input_file, 'r').readlines()
        # for i in tqdm.trange(len(data)):
        #     q = data[i]
        #     inputs = tokenizer.encode(q.strip()+" ? To answer this question, we need to know", return_tensors="pt")
        #     outputs = model.generate(inputs.cuda(), max_new_tokens=100, do_sample=False, top_k=50)
        #     result = [tokenizer.decode(outputs[0], skip_special_tokens=True)]
        #     outputs = model.generate(inputs.cuda(), max_new_tokens=100, do_sample=True, top_k=50, num_return_sequences=num_per_q-1)
        #     result += [tokenizer.decode(outputs[j], skip_special_tokens=True) for j in range(num_per_q-1)]
        #     result = [str(i), q.strip()] + result
        #     cw.writerow(result)
        #     fw.flush()
        # fw.close()