#----#
#usage: let two models affer RL talk to each other.
#----#
import logging
from argparse import ArgumentParser
from itertools import chain
from collections import defaultdict
from pprint import pformat
import warnings
import re 

import torch
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import get_dataset
from train_buyer import build_input_from_segments_buyer, add_special_tokens_buyer#, SPECIAL_TOKENS_BUYER
from train import build_input_from_segments, add_special_tokens_

MODEL_INPUTS_TEST = ["input_ids", 'labels']

class Agent:
    def __init__(self, args, aspect='Buyer', use_HF_form=True):
        print(f'Initializing {aspect}...')
        self.model_type = args.model
        self.aspect = aspect #Note: To be modified on newly trained models.
        self.model_path = args.model_checkpoint_buyer if aspect=='Buyer' else args.model_checkpoint_seller
        self.dataset_path =  args.dataset_buyer_path if aspect=='Buyer' else args.dataset_path
        self.dataset_cache = args.dataset_buyer_cache if aspect=='Buyer' else args.dataset_cache
        self.device = args.device
        self.model,self.tokenizer = self.get_pretrained_model_and_tokenizer() if use_HF_form ==True else self.get_pretrained_model_and_tokenizer(use_HF_form=False)
        #default dataset is via seller aspect
        self.chat = get_dataset(self.tokenizer, self.dataset_path, self.dataset_cache)
        self.first_talks = self.get_beginner_from_datasets()
        self.build_input = build_input_from_segments_buyer if aspect=='Buyer' else build_input_from_segments

        

    def talks(self, input_tensor, args ):
        with torch.no_grad():
            outputs_tensor = self.model.generate(input_tensor, max_length=args.max_generate_length, min_length=args.min_generate_length)
        pre_outputs = list(outputs_tensor)[0]
        outputs = self.tokenizer.decode(pre_outputs)
        outputs = post_process(outputs)
        
        return outputs



    def get_pretrained_model_and_tokenizer(self, is_fine_tuned =True, use_HF_form = True):
    
        tokenizer_class, model_class = (T5Tokenizer, T5ForConditionalGeneration) 
        model = model_class.from_pretrained(self.model_type)
        tokenizer = tokenizer_class.from_pretrained(self.model_type)
        if is_fine_tuned:
            if self.aspect == 'Buyer':
                add_special_tokens_buyer(model, tokenizer)
            else:
                add_special_tokens_(model, tokenizer)
            if not use_HF_form:
                model_weights = torch.load(self.model_path)['model']
                model.load_state_dict(model_weights)
            else:
                model = model_class.from_pretrained(self.model_path)
    
        model.to(self.device)
        model.eval()
    
        return model, tokenizer

    def get_beginner_from_datasets(self ):
        """ Prepare the dataset for training and evaluation """
        print(f"Build inputs started by {self.aspect}")
        if 'test' in self.dataset_path:
            datasets = {'test' :defaultdict(list)}
        elif 'valid' in self.dataset_path:
            datasets = {"valid": defaultdict(list)}
        else:
            datasets = {"train": defaultdict(list)}
        
        for dataset_name, dataset in self.chat.items():
            for dialog in dataset:
                utterance = dialog["utterances"][0] 
                reward_buyer = dialog["reward_buyer"]
                reward_seller = dialog["reward_seller"]
                instance = None
                if self.aspect == 'Seller': 
                    #sellers' first utterances are for buyers and vice versa.
                    scenario =  dialog['scenario']
                    scenario_other =  dialog['scenario_buyer']
                    history = None
                    if dialog['starts'] == 1:
                        history =  utterance["history"]
                        reply = utterance["candidates"][0]#doesn't matter
                        instance = build_input_from_segments(scenario, history, reply, self.tokenizer) #tokenizer may be wrong
                #Buyer
                else:
                    scenario =  dialog["scenario_buyer"]
                    scenario_other =  dialog['scenario']
                    if dialog['starts'] == 0:
                        history = utterance["history"]
                        reply = utterance["candidates"][0]#doesn't matter
                        instance = build_input_from_segments_buyer(scenario, history, reply, self.tokenizer)
                if instance:
                    datasets[dataset_name]['input_ids'].append(instance['input_ids'])
                    datasets[dataset_name]['scenario'].append(scenario)
                    datasets[dataset_name]['scenario_other'].append(scenario_other)
                    datasets[dataset_name]['reward_buyer'].append(reward_buyer)
                    datasets[dataset_name]['reward_seller'].append(reward_seller)
                    datasets[dataset_name]['preference'].append(dialog['preference'])
                    datasets[dataset_name]['quality_profit'].append(dialog['quality_profit'])
        return datasets

def post_process(snt:str)->str:
    snt_list = snt.split()
    if '<eos>' in snt_list:
        eos_index = snt_list.index('<eos>')+1
    else:
        snt_list.append('<eos>')
        eos_index = len(snt_list)
    if '<bos>' in snt_list:
        bos_index = snt_list.index('<bos>')
    elif '<pad>' in snt_list:
        bos_index = snt_list.index('<pad>')
    else:
        bos_index = -1
    return ' '.join(snt_list[bos_index+1:eos_index])
        
def preprocessing_for(agent:Agent, outputs, history, scenario):
    reply = [''] #dummy    
    outputs_plaintext = ' '.join(outputs.split()[1:-1])                
    outputs_ids = agent.tokenizer.convert_tokens_to_ids(agent.tokenizer.tokenize(outputs_plaintext))
    history = history + [outputs_ids]
    instance = agent.build_input(scenario, history, reply, agent.tokenizer) 
    input_ids = instance['input_ids']
    input_tensor = torch.tensor([input_ids]).to(agent.device)
    return input_ids, input_tensor, history

def get_output_dict(agent:Agent, input_ids, outputs, dialog_num, reward_buyer, reward_seller, preference, quality_profit, selection=None):
    if selection:
        out_dict = {'Input':agent.tokenizer.decode(input_ids), 
                'Generate': outputs,
                'dialog_num': dialog_num,
                'reward_buyer':reward_buyer,
                'reward_seller':reward_seller,
                'preference':preference,
                'quality_profit':quality_profit,
                'selection':selection}
    else:
        out_dict = {'Input': agent.tokenizer.decode(input_ids), 
                'Generate': outputs,
                'dialog_num': dialog_num,
                'reward_buyer':reward_buyer,
                'reward_seller':reward_seller,
                'preference':preference,
                'quality_profit':quality_profit}
    return out_dict

def get_selection(outputs):
    outputs_split = outputs.split()
    selection_idx = outputs_split.index('<selection>')+1
    selection = outputs_split[selection_idx]
    return selection

def get_max_score(reward_buyer:list, reward_seller:list):
    max_value_b = max(reward_buyer)
    max_value_s = max(reward_seller)
    Fruits_dict = {0:'apples', 1:'bananas', 2:'oranges'}

    def get_indices(max_value:int, _list:list):
        indices=[]
        for i,v in enumerate(_list):
            if max_value == v:
                indices.append(i)
        assert len(indices)>0,'max_value should exist'
        return indices

    max_reward_b = [Fruits_dict[x] for x in get_indices(max_value_b, reward_buyer)]
    max_reward_s = [Fruits_dict[x] for x in get_indices(max_value_s, reward_seller)]
    win_win_choice = ['not existed']
    if len([x for x in max_reward_b if x in max_reward_s])>0:
        win_win_choice = [x for x in max_reward_b if x in max_reward_s]
    return max_reward_b, max_reward_s, win_win_choice


def get_statistics(output_list, s_dict)->dict:
    
    o_dict = s_dict
    for item in output_list:
        if 'selection' in item:
            o_dict['num_selection'] += 1
            max_reward_b, max_reward_s, win_win_choice = get_max_score(item['reward_buyer'], item['reward_seller'])
            if item['selection'] in max_reward_b:
                o_dict['Buyer'] += 1
            if item['selection'] in max_reward_s:
                o_dict['Seller'] += 1 
            if item['selection'] in win_win_choice:
                o_dict['WW'] += 1
            if not "not existed" in win_win_choice:
                o_dict['WW_max'] +=1 

    return  o_dict

def use_small_testing_data(args):
    if args.use_small_testing_data:
        args.dataset_path = '../data/dataset_valid20_reward.json'
        args.dataset_cache = '../data/dataset_valid20_reward_cache'
        args.dataset_buyer_path = "../data/dataset_valid20_reward_buyer_reply.json"
        args.dataset_buyer_cache = '../data/dataset_valid20_reward_buyer_reply_cache'
    return args

def use_valid_data(args):
    if args.valid:
        args.dataset_path = '../data/dataset_valid423_reward.json'
        args.dataset_cache = '../data/dataset_vali423_reward_cache'
        args.dataset_buyer_path = "../data/dataset_valid423_reward_buyer_reply.json"
        args.dataset_buyer_cache = '../data/dataset_valid423_reward_buyer_reply_cache'
    return args


def get_scenario_table(pre_list:list, qp:list)->str:
    f_list=['Apples', 'Bananas','Oranges']
    
    q_list=[]
    profit_list=[]
    for i,value in enumerate(qp):
        if i<=2: 
            q_list.append(value)
        else:
            profit_list.append(value)
    
    s = f"{'Item':8}|{'Preference':11}|{'Quality':8}|{'Profit':7}|{'Buyer Score':12}|{'Seller Score':12}\n"
    for i in range(3):
        buyer_score = pre_list[i]*q_list[i]
        seller_score = buyer_score + profit_list[i]
        s = s+ f"{f_list[i]:8}|{pre_list[i]:11}|{q_list[i]:8}|{profit_list[i]:7}|{buyer_score:12}|{seller_score:12}\n"
    return s
                        
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/dataset_test423_reward.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='../data/dataset_test423_reward_cache', help="Path or url of the dataset cache")
    parser.add_argument("--dataset_buyer_path", type=str, default="../data/dataset_test423_reward_buyer_reply.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_buyer_cache", type=str, default='../data/dataset_test423_reward_buyer_reply_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="t5-base", help="Model type ")  
    parser.add_argument("--model_checkpoint_buyer", type=str, default="~/model_foler", help="Path, url or short name of the model")
    parser.add_argument("--model_checkpoint_seller", type=str, default="~/model_foler", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--value_only", action='store_true', help="If assigned get rid of description quality... profit...")
    parser.add_argument("--max_generate_length", type=int, default=35, help="Number of max_generate_length")
    parser.add_argument("--min_generate_length", type=int, default=5, help="Number of min_generate_length")
    parser.add_argument("--valid", action='store_true', help="If assigned, test on validation set.")
    parser.add_argument("--use_small_testing_data", action='store_true', help="If assigned, test on sampled set.")
    parser.add_argument("--bs_num", type=int, default=0, help="which buyer/seller pair is tested?")
    
    args = parser.parse_args()

    if args.use_small_testing_data:
        args = use_small_testing_data(args)
    if args.valid :
        args = use_valid_data(args)


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(pformat(args))

    Buyer = Agent(args, aspect='Buyer',use_HF_form=True) #HF_form = True: for RL version
    Seller = Agent(args, aspect='Seller',use_HF_form=True)#False: for supervised version
    n = 0 # of buyer starts
    m = 0 # of seller starts
    max_round = 10
    for dataset_name, items in Buyer.first_talks.items():
        buyer_model = args.model_checkpoint_buyer.split('_')[-1]
        seller_model = args.model_checkpoint_seller.split('_')[-1]
        with open (f'{args.model_checkpoint_seller}{args.bs_num:02}_OUTPUTS_of_self_play_both_maxlen{args.max_generate_length}.txt', 'w') as f:
            output_list = []
            n = len(items['input_ids'])

            for i in range(n):
                round_ = 0
                input_ids = items['input_ids'][i]#Buyer's
                scenario = items['scenario'][i]
                scenario_other = items['scenario_other'][i]
                reward_buyer = items['reward_buyer'][i]
                reward_seller = items['reward_seller'][i]
                preference = items['preference'][i]
                quality_profit = items['quality_profit'][i]
                history = []
                input_tensor = torch.tensor([input_ids]).to(args.device)
                decided = False
                while round_ < max_round and not decided:
                    #Buyer talks first
                    outputs = Buyer.talks(input_tensor, args)
                     
                    if '<selection>' in outputs:
                        selection = get_selection(outputs)
                        output_list.append(get_output_dict(Buyer, input_ids, outputs, i, reward_buyer, reward_seller, preference, quality_profit, selection=selection))
                        decided = True
                        break
                    else:
                        output_list.append(get_output_dict(Buyer, input_ids, outputs, i, reward_buyer, reward_seller, preference, quality_profit, selection=None))
                    
                    #Seller talks
                    input_ids, input_tensor, history = preprocessing_for(Seller, outputs, history, scenario_other)
                    outputs = Seller.talks(input_tensor, args)
                    output_list.append(get_output_dict(Seller, input_ids, outputs, i, reward_buyer, reward_seller, preference, quality_profit, selection=None))
                    #prepare Buyer's input
                    input_ids, input_tensor, history = preprocessing_for(Buyer, outputs, history, scenario)
                    round_ += 1
            
            #pre-write
            pre = 0
            s_n = 0
            final_list = []
            dialog_list = []
            scenario_list = []
            reward_list = []
            for x,row in enumerate(output_list):
                if row["dialog_num"] != pre:
                
                    final_list.append({'dataset_name':dataset_name,
                                        "scenario":scenario_list,
                                        "dialog":dialog_list,
                                        "dialog_num":pre,
                                        'reward_list':reward_list,
                                        'preference_list':preference_list,
                                        'quality_profit_list':quality_profit_list})
                    pre += 1
                    s_n = 0
                    dialog_list = []
                    scenario_list = []
                
                dialog_list.append(row["Generate"])
                reward_list = [row['reward_buyer'],row['reward_seller']]
                preference_list = row['preference']
                quality_profit_list = row['quality_profit']
                if x == len(output_list)-1:
                    final_list.append({'dataset_name':dataset_name,
                                        "scenario":scenario_list,
                                        "dialog":dialog_list,
                                        "dialog_num":pre,
                                        'reward_list':reward_list,
                                        'preference_list':preference_list,
                                        'quality_profit_list':quality_profit_list})
            
            statistics_dict = {'Buyer':0, 'Seller':0, 'WW':0,'WW_max':0, 'num_selection': 0}
            statistics_dict = get_statistics(output_list,statistics_dict)    

        
            #Seller, bad code
            items = Seller.first_talks[dataset_name]
            output_list = []
            m = len(items['input_ids'])#Seller changes here

            for i in range(m):
                round_ = 0
                input_ids = items['input_ids'][i]#Seller's
                scenario = items['scenario'][i]
                scenario_other = items['scenario_other'][i]
                reward_buyer = items['reward_buyer'][i]
                reward_seller = items['reward_seller'][i]
                preference = items['preference'][i]
                quality_profit = items['quality_profit'][i]
                history = []
                input_tensor = torch.tensor([input_ids]).to(args.device)
                decided = False
                while round_ < max_round and not decided:
                    #Seller talks first
                    outputs = Seller.talks(input_tensor, args)
                    output_list.append(get_output_dict(Seller, input_ids, outputs, n+i, reward_buyer, reward_seller,preference, quality_profit, selection=None))
                    #Buyer talks
                    input_ids, input_tensor, history = preprocessing_for(Buyer, outputs, history, scenario_other)
                    outputs = Buyer.talks(input_tensor, args)
                    if '<selection>' in outputs:
                        selection = get_selection(outputs)
                        #note seller dialogue number n+i
                        output_list.append(get_output_dict(Buyer, input_ids, outputs, n+i, reward_buyer, reward_seller,preference, quality_profit, selection=selection))
                        decided = True
                        break
                    else:
                        output_list.append(get_output_dict(Buyer, input_ids, outputs, n+i, reward_buyer, reward_seller,preference, quality_profit, selection=None))
                    
                    #prepare Seller's input
                    input_ids, input_tensor, history = preprocessing_for(Seller, outputs, history, scenario)
                    round_ += 1
            
            #write out
            value_only = 'value_only' if args.value_only else 'with description'
            f.write(f'{dataset_name} {args.model_checkpoint_buyer[8:]} {args.model_checkpoint_seller[8:]}\n')
        
            #Seller changes here
            pre = n
            s_n = 0
            dialog_list = []
            scenario_list = []
            reward_list = []
            for x,row in enumerate(output_list):
                if row["dialog_num"] != pre:
                    
                    final_list.append({'dataset_name':dataset_name,
                                        "scenario":scenario_list,
                                        "dialog":dialog_list,
                                        "dialog_num":pre,
                                        'reward_list':reward_list,
                                        'preference_list':preference_list,
                                        'quality_profit_list':quality_profit_list})
                    pre += 1
                    s_n = 0
                    dialog_list = []
                    scenario_list = []
                if s_n <2:
                    s_n+=1
                    s = ', '.join(row["Input"].split('<sep>')[:-1]).strip('<bos>')
                    scenario_list.append(s)
                dialog_list.append(row["Generate"])
                reward_list = [row['reward_buyer'],row['reward_seller']]
                preference_list = row['preference']
                quality_profit_list = row['quality_profit']
                if x == len(output_list)-1:
                    final_list.append({'dataset_name':dataset_name,
                                        "scenario":scenario_list,
                                        "dialog":dialog_list,
                                        "dialog_num":pre,
                                        'reward_list':reward_list,
                                        'preference_list':preference_list,
                                        'quality_profit_list':quality_profit_list})
            
            statistics_dict = get_statistics(output_list,statistics_dict)    
            #'BEST' rate
            assert len(final_list) != 0
            f.write(f'statistics\n')
            f.write(f"{'Buyer best-choice rate':35} {statistics_dict['Buyer']/len(final_list):.6f}\n")
            f.write(f"{'Seller best-choice rate':35} {statistics_dict['Seller']/len(final_list):.6f}\n")
            f.write(f"{'Win-win-choice rate (if possible)':35} {statistics_dict['WW']/statistics_dict['WW_max']:.6f}\n")
            f.write(f"{'Max win-win-choice rate':35} {statistics_dict['WW_max']/len(final_list):.6f}\n")



            
            for i,row in enumerate(final_list):
                #Buyer
                if i < n:
                    f.write(f'=============================\n#{row["dialog_num"]}\n')
                    f.write(get_scenario_table(row['preference_list'], row['quality_profit_list']))
                    f.write(f"\n")
                    for utterance in row["dialog"]:
                        f.write(f'{utterance}\n')
                    

                #Seller
                else: 
                    f.write(f'=============================\n#{row["dialog_num"]}\n')
                    #Seller changes here
                    f.write(get_scenario_table(row['preference_list'], row['quality_profit_list']))
                    f.write(f"\n")
                    for utterance in row["dialog"]:
                        f.write(f'{utterance}\n')
                    
    

if __name__ == '__main__':
    main()
