from transformers import AutoTokenizer, LlamaConfig, AutoConfig
import torch
from bevllm.model.language_model.bevllm_llama import BevLLMLlamaForCausalLM
from dataset import NuScenesCaptionDataset, NuScenesViewCaptionDataset, NuScenesLidarLLmCaptionDataset
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import json
from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
import random
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv()

precision_scores = []
recall_scores = []
fmeasure_scores = []
bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

def collate_fn(batch):
    bev = [sample["bev"] for sample in batch]
    view = [sample["view"] for sample in batch]
    input_ids = [sample["input_ids"] for sample in batch]

    return {"input_ids": torch.stack(input_ids, dim=0), "bev": torch.cat(bev, dim=0), "view": view}


def calculate_bleu(reference, candidate):
    # Tokenize the reference and candidate sentences
    reference = [reference.split()]
    candidate = candidate.split()
    
    # Define the smoothing function
    smoothie = SmoothingFunction().method4
    
    # Calculate BLEU scores
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)
    
    return bleu1, bleu2, bleu3, bleu4

def get_view_string(view):    
    string_conversion = {3: "back view", 
                        2:" back left view",
                        4: "back right view",
                        0: "front view",
                        5: "front right view",
                        1: "front left view"}
    return string_conversion[view]

def calculate_rouge(ref, cand):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ref, cand)

    precision_scores.append(scores['rougeL'].precision)
    recall_scores.append(scores['rougeL'].recall)
    fmeasure_scores.append(scores['rougeL'].fmeasure)


    return scores


def collate_fn(batch):
    bev = [sample["bev"] for sample in batch]
    view = [sample["view"] for sample in batch]
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    token = [sample["token"] for sample in batch]
    text = [sample["text"] for sample in batch]
    return {"input_ids": torch.stack(input_ids, dim=0), "attention_mask":torch.stack(attention_mask, dim=0),  "bev": torch.cat(bev, dim=0), "view": view, "token":token, "text":text}


def collate_fn_lidarLLM(batch):

    #return{"question":question, "answer": answer, "bev": bev, "view":view, "token": self.cap_data[index]["sample_token"] }

    bev = [sample["bev"] for sample in batch]
    view = [sample["view"] for sample in batch]
    question = [sample["question"] for sample in batch]
    answer = [sample["answer"] for sample in batch]
    token = [sample["token"] for sample in batch]
    

    return {"question":question, "answer": answer, "bev": torch.cat(bev, dim=0).to("cuda"), "view": view, "token": token}


def eval_model(model, tokenizer, batch_size, start_step, stop_step):
    
    val_set = NuScenesViewCaptionDataset("/anvme/workspace/v103fe15-bev_features/data/sets/nuscenes/", 
                                  "val", 
                                  None, 
                                  tokenizer, 
                                  load_from_file=True, 
                                  tensor_root="/anvme/workspace/v103fe15-bev_features/bev_features/val/")
    val_dl = DataLoader(val_set, batch_size, False, collate_fn=collate_fn)
    
    samples = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation

        for step, data in enumerate(val_dl):
            
            if step < start_step:
                print(step)
                continue  # Skip already processed steps
            if stop_step != None:
                if step == stop_step:
                    return

            prompts = []
            for i in range(len(data["text"])):
                orientation_string = "This is the car's " + get_view_string(data["view"][i])
                message =[{"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                          {"role": "user", "content": " <image> "+ orientation_string + " Please describe the current scene."} ]
                prompts.append(tokenizer.apply_chat_template(message, tokenize=False, return_tensors="pt"))
        
            inputs = tokenizer(prompts, padding=True , return_tensors="pt", truncation=True).to("cuda")

            start_time = time.time()
            output = model.generate(inputs["input_ids"].to("cuda"), attention_mask = inputs["attention_mask"], bevs = data["bev"].to("cuda"), view = data["view"], max_new_tokens = 250)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'completed in {elapsed_time:.2f} seconds')
            text = tokenizer.batch_decode(output, skip_special_tokens=True)

            start_time = time.time()
            for i in range(len(text)):

                print(f'generated Text: {text[i]}')
                text[i] = text[i].lstrip("assistant\n\n")
                

                rouge_score = calculate_rouge( data["text"][i],text[i])
                bleu1, bleu2, bleu3, bleu4 = calculate_bleu(data["text"][i], text[i])
                sample = {"reference" : data["text"][i],
                            "prediction": text[i],
                            "ROUGE-L" : rouge_score,
                            "bleu1" : bleu1,
                            "bleu2" : bleu2,
                            "bleu3" : bleu3,
                            "bleu4" : bleu4,
                            "view" : data["view"][i],
                            "token" : data["token"][i] }
                print(sample)

                samples.append(sample)
                
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Print the elapsed time for the epoch
            print(f'completed in {elapsed_time:.2f} seconds')
            print(f'Step{step}/{len(val_dl)}')

    with open("evaluation_bevllm_llama3_1B_nuView.json", "w") as f:
        json.dump(samples, f, indent = 4)
           
                

    #with open("data_samples_instruction.json", "w") as f:
    #    json.dump({"samples": samples}, f, indent=4)

def chatting(model, tokenizer):
        
    questions = [
        "What can you see in this scene?",
        "What details are visible in the current view?",
    ]

    questions = [{"question": "What can you see in this scene?", "view":0 },
                 {"question": "What details are visible in the current view?", "view": 3}
                ]


    indicies = [1000]

    tokens = [
        "135bf33890ba4ca2984a931444923eda",
    ]


    prediction_list = []
    bev = torch.load("/anvme/workspace/v103fe15-bev_features/bev_features/val/tensor_" + tokens[0] + ".pt").unsqueeze(0)
    bev = bev.to("cpu")
    bev = bev.detach().clone() 
    for q in questions:
        orientation_string = "This is the car's" + get_view_string(q["view"]) + "."
        messages = [{"role": "user", "content": " <image>" + orientation_string + " " + q["question"] }]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        #print(inputs)   
        #print(y)
        output = model.generate(inputs = inputs["input_ids"],attention_mask=inputs["attention_mask"], bevs = bev, view = [q["view"]] ,max_new_tokens=250)

        text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(text)

        #print(f'Index_{index}: {text}')

       

def extract_samples(model, tokenizer):

    questions = [
        "How would you describe the overall road conditions?",
        "Are there any potential hazards orobstructions on the road that a driver should be aware of?",
        "Can you provide a description for the road scene such that it sets it apart from others?",
        "Describe the traffic conditions.",
        "Please describe the current scene."
    ]

    #questions = [ "Can you provide a description for the road scene such that it sets it apart from others?",
    #              "Describe the traffic conditions.",
    #              "Please describe the current scene.",
    #              
    #              ]


    indicies = [0,50,150,200,500,750,1000,2500,4000,4500]

    tokens = [
        "fd8420396768425eabec9bdddf7e64b6",
        "d34f2eec80f54b608f9b9d0a86f103bd",
        "2c0c09db419e4ae692666d7dd235594d",
        "fbd3aa2098234fb1869d34d920e16eda",
        "38c0c7b30e1142e4bc0833c2d489c493",
        "50d86a5cd45440c89e895a5d20ea5267",
        "135bf33890ba4ca2984a931444923eda",
        "b6244e44a9624fce85bfe11d67a6c730",
        "4e04d32a9949430287c14d49851f6f55",
        "b1303058fa624645b753d7283d70de45"
    ]
    #tokens = [
    #    "fd8420396768425eabec9bdddf7e64b6",
    #]
    #indicies = [0]


    for token, index in zip(tokens, indicies):
        prediction_list = []
        bev = torch.load("/anvme/workspace/v103fe15-bev_features/bev_features/val/tensor_" + token + ".pt").unsqueeze(0)
        bev = bev.to("cpu")
        bev = bev.detach().clone() 
        for y in range(6):
            q = questions[random.randint(0,len(questions)-1)]
            #q = "How are the road condition? Focus on the wether in the scene."
            orientation_string = "This is the car's " + get_view_string(y) + "."

            messages = [{"role": "system", "content": "You are an AI assistant in an autonomous vehicle describing the relevant scene details surrounding the vehicle."},
                        {"role": "user", "content": " <image>" + orientation_string + " " + q }]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt")
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            #print(inputs)   
            #print(y)
            output = model.generate(inputs = inputs["input_ids"],attention_mask=inputs["attention_mask"], bevs = bev.to("cuda"), view = [0],max_new_tokens=250)

            text = tokenizer.batch_decode(output, skip_special_tokens=True)
            
            text = text[0].lstrip("assistant\n\n")
            print(text)
            prediction_list.append({"question":orientation_string + " " + q, "text":text})

            #print(f'Index_{index}: {text}')
        
        
        with open("felix_brandstaetter_thesis/main/eval_results_paper/"+ "sample" + str(index) +".txt", "w" ) as file:
            for pred in prediction_list:
                file.write(f'{pred["question"]}\n')
                file.write(f'{pred["text"]}\n\n')


def main(args):
    #tokenizer = AutoTokenizer.from_pretrained("felix_brandstaetter_thesis/main/model/bevllm/base_model")

    #model = BevLLMLlamaForCausalLM

    #model.model = PeftModel.from_pretrained(model.model,"felix_brandstaetter_thesis/main/model/bevllm_pos/cap_adapter")

    #print(model)

    access_token = os.getenv("ACCESS_TOKEN")
    model_id = os.getenv("MODEL_ID")

    config = LlamaConfig.from_pretrained(model_id, token=access_token, cache_dir="felix_brandstaetter_thesis/main/model")
    config.cache_dir = "felix_brandstaetter_thesis/main/model"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token,  cache_dir="felix_brandstaetter_thesis/main/model")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})


    tokenizer.pad_token = tokenizer.eos_token

    #train_set = NuScenesQADataset(cache_dir + "/nuscenes/", "train", model.get_bev_config(),None,False)
    model = BevLLMLlamaForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    model.model = get_peft_model(model.model, peft_config)
    
    model.load_state_dict(torch.load("snapshot_bevllm_llama_1B_nuView.pt")["MODEL_STATE"])
    model.to("cuda")



    #chatting(model, tokenizer)
    #return 0
    #extract_samples(model, tokenizer)

    #return 0
    eval_model(model, tokenizer, 1, start_step=args.start_step, stop_step=args.stop_step)


    #torch.save(torch.tensor(precision_scores),"validation_results_mt2/precision_scores.pt")
    #torch.save(torch.tensor(recall_scores),"validation_results_mt2/recall_scores.pt")
    #torch.save(torch.tensor(fmeasure_scores),"validation_results_mt2/fmeasure_scores.pt")
    #torch.save(torch.tensor(bleu1_scores),"validation_results_mt2/bleu1_scores.pt")
    #torch.save(torch.tensor(bleu2_scores),"validation_results_mt2/bleu2_scores.pt")
    #torch.save(torch.tensor(bleu3_scores),"validation_results_mt2/bleu3_scores.pt")
    #torch.save(torch.tensor(bleu4_scores),"validation_results_mt2/bleu4_scores.pt")

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_fmeasure = np.mean(fmeasure_scores)
    avg_bleu1 = np.mean(bleu1_scores)
    avg_bleu2 = np.mean(bleu2_scores)
    avg_bleu3 = np.mean(bleu3_scores)
    avg_bleu4 = np.mean(bleu4_scores)

    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F-measure: {avg_fmeasure}")
    print(f"Average BLEU-1: {avg_bleu1}")
    print(f"Average BLEU-2: {avg_bleu2}")
    print(f"Average BLEU-3: {avg_bleu3}")
    print(f"Average BLEU-4: {avg_bleu4}")


if __name__ == "__main__":
    args = Argparser()
    main(args)

