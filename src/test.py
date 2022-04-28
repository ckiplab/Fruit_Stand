#---#
#usage: test a single fine-tuned model 
#---#

import logging
from argparse import ArgumentParser
from itertools import chain
from collections import defaultdict
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import get_dataset
from train_buyer import build_input_from_segments_buyer, add_special_tokens_buyer#, SPECIAL_TOKENS_BUYER
from train import build_input_from_segments, add_special_tokens_

MODEL_INPUTS_TEST = ["input_ids", 'labels']

def get_pretrained_model_and_tokenizer(args, is_fine_tuned =True):
    print("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (T5Tokenizer, T5ForConditionalGeneration) 
    model = model_class.from_pretrained(args.model)
    tokenizer = tokenizer_class.from_pretrained(args.model)
    if is_fine_tuned:
        if args.aspect_buyer:
            add_special_tokens_buyer(model, tokenizer)
        else:
            add_special_tokens_(model, tokenizer)
        model_weights = torch.load(args.model_checkpoint)['model']
        model.load_state_dict(model_weights)
    
    model.to(args.device)
    
    return model, tokenizer

def get_output_sentence(input_snt, model, tokenizer ):
    input_ids = tokenizer(input_snt, return_tensors="pt").input_ids  # Batch size 1
    outputs_ids = list(model.generate(input_ids))[0]
    outputs = tokenizer.decode(outputs_ids)
    return outputs

def get_data_dicts(args, tokenizer, chat):
    """ Prepare the dataset for training and evaluation """
    
    print("Build inputs and labels")
    if 'test' in args.dataset_path:
        datasets = {'test' :defaultdict(list)}
    else:
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    if args.aspect_buyer:
        
    else:
        
    for dataset_name, dataset in chat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        for dialog in dataset:
            if args.aspect_buyer:
                scenario = dialog["preference"] if args.value_only else dialog["scenario_buyer"]
            else:
                scenario = dialog['quality_profit'] if args.value_only else dialog['scenario']
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                reply = utterance["candidates"][-num_candidates]
                
                instance = build_input_from_segments(scenario, history, reply, tokenizer)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
    return datasets



def test():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/dataset_test423_reward.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='../data/dataset_test423_reward_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="t5-base", help="Model type ")  
    parser.add_argument("--model_checkpoint", type=str, default="~/model_folder/model.pt", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--show_only_pretrained", action="store_true")
    parser.add_argument("--value_only", action='store_true', help="If assigned get rid of description quality... profit...")
    parser.add_argument("--aspect_buyer", action='store_true', help="If assigned, test the buyer.")
    parser.add_argument("--max_generate_length", type=int, default=45, help="Number of max_generate_length")
    parser.add_argument("--min_generate_length", type=int, default=5, help="Number of min_generate_length")
    parser.add_argument("--train_valid", action='store_true', help="If assigned, test on training set and validation set.")
    
    args = parser.parse_args()

    if args.aspect_buyer:        
        args.dataset_path = args.dataset_path.replace('reward','reward_buyer_reply',1)
        args.dataset_cache = args.dataset_cache.replace('reward','reward_buyer_reply',1)

    if args.train_valid:
        assert '423' in args.dataset_path
        args.dataset_path = args.dataset_path.replace('test423','train3386_valid423',1)
        args.dataset_cache = args.dataset_cache.replace('test423','train3386_valid423',1)


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(pformat(args))

    model, tokenizer = get_pretrained_model_and_tokenizer(args)
    model.eval()

    chat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    datasets = get_data_dicts(args, tokenizer, chat)
    
    for dataset_name, items in datasets.items():
        with open (f'{args.model_checkpoint}.OUTPUTS_from_{args.dataset_path[8:-5]}.txt', 'w') as f:
            output_list = []
            ppls = []
            n = len(items['input_ids'])
            for i in range(n):
                input_ids = items['input_ids'][i]
                reply = items['labels'][i]
                input_tensor = torch.tensor([input_ids]).to(args.device)
                reply_tensor = torch.tensor([reply]).to(args.device)
                with torch.no_grad():
                    loss = model(input_ids=input_tensor, labels=reply_tensor).loss
                    outputs_tensor = model.generate(input_tensor, max_length=args.max_generate_length, min_length=args.min_generate_length)
                outputs = list(outputs_tensor)[0]
                ppls.append(loss)
                output_list.append({'Input': tokenizer.decode(input_ids), 
                                    'Reply': tokenizer.decode(reply),
                                    'Generate': tokenizer.decode(outputs)})
            ppl = torch.exp(torch.stack(ppls).sum()/n)
            print(f'ppl of {args.model_checkpoint}: {ppl}')
            value_only = 'value_only' if args.value_only else 'with description'
            f.write(f'{dataset_name} {args.model_checkpoint[8:]} dataset\nppl: {ppl}\nInput: {value_only}\n')
            for index, row in enumerate(output_list):
                f.write(f'=========================\n#{index}\nInput:\n{row["Input"]}\nReply:\n{row["Reply"]}\nGenerate:\n{row["Generate"]}\n')



    

if __name__ == '__main__':
    test()
