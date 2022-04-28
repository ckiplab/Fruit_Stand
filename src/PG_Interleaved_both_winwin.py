#----#
#usage: RL(interleave), interleave RL and SL to both buyer and seller.
#models are updated only when they reach mutually optimal solutions.
#----#
import logging
import time
import os
import random
from collections import deque
from argparse import ArgumentParser
from itertools import chain
from collections import defaultdict
from pprint import pformat
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import AdamW,T5Tokenizer, get_scheduler
from tqdm.auto import tqdm

from utils import get_dataset, make_logdir, post_process
from train_buyer import build_input_from_segments_buyer, add_special_tokens_buyer
from train_seller import build_input_from_segments, add_special_tokens_
from self_play import get_scenario_table
from model.modeling_t5 import T5ForConditionalGeneration


MODEL_INPUTS_TEST = ["input_ids", 'labels']

class Agent:
    def __init__(self, args, aspect='Buyer'):
        print(f'Initializing {aspect}...')
        self.model_type = args.model
        self.aspect = aspect #Note: To be modified on newly trained models.
        self.model_path = args.model_checkpoint_buyer if aspect=='Buyer' else args.model_checkpoint_seller
        self.dataset_path =  args.dataset_buyer_path if aspect=='Buyer' else args.dataset_path
        self.dataset_cache = args.dataset_buyer_cache if aspect=='Buyer' else args.dataset_cache
        
        self.device = args.device
        self.model,self.tokenizer = self.get_pretrained_model_and_tokenizer()
        self.chat = get_dataset(self.tokenizer, self.dataset_path, self.dataset_cache)
        self.build_input = build_input_from_segments_buyer if self.aspect=='Buyer' else build_input_from_segments
        
        #for SL
        self.dataset_path_seller =  args.dataset_path
        self.dataset_cache_seller = args.dataset_cache
        self.tokenizer_seller = self.get_tokenizer(aspect='Seller')
        self.chat_seller = get_dataset(self.tokenizer_seller, self.dataset_path_seller, self.dataset_cache_seller)

        self.dataset_path_buyer =  args.dataset_buyer_path
        self.dataset_cache_buyer = args.dataset_buyer_cache
        self.tokenizer_buyer = self.get_tokenizer(aspect='Buyer')
        self.chat_buyer = get_dataset(self.tokenizer_buyer, self.dataset_path_buyer, self.dataset_cache_buyer)

        #for PG and SL
        self.first_talks = self.get_beginner_from_datasets()

        

    def talks(self, input_tensor, args ):
        #do_sample = False if self.aspect == 'Buyer' else True
        do_sample = True
        with torch.no_grad():
            outputs_tensor = self.model.generate(input_tensor,do_sample=do_sample, max_length=args.max_generate_length, min_length=args.min_generate_length)
        pre_outputs = list(outputs_tensor)[0]
        outputs = self.tokenizer.decode(pre_outputs)
        outputs_for_history, outputs_for_labels = post_process(outputs, self.aspect)
        return outputs_for_history, outputs_for_labels



    def get_pretrained_model_and_tokenizer(self, is_fine_tuned =True):
    
        tokenizer_class, model_class = (T5Tokenizer, T5ForConditionalGeneration) 
        model = model_class.from_pretrained(self.model_type)
        tokenizer = tokenizer_class.from_pretrained(self.model_type)
        if is_fine_tuned:
            if self.aspect == 'Buyer':
                add_special_tokens_buyer(model, tokenizer)
            else:
                add_special_tokens_(model, tokenizer)
            model_weights = torch.load(self.model_path)['model']
            model.load_state_dict(model_weights)
    
        model.to(self.device)
        model.eval()
    
        return model, tokenizer

    def get_tokenizer(self, aspect='Seller'):
        tokenizer_class = T5Tokenizer
        tokenizer = tokenizer_class.from_pretrained(self.model_type)
        if aspect == 'Buyer':
            from train_buyer import ATTR_TO_SPECIAL_TOKEN
        elif aspect == 'Seller':
            from train import ATTR_TO_SPECIAL_TOKEN
        tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        return tokenizer

    def get_beginner_from_datasets(self ):
        """ Prepare the dataset for training and evaluation """
        print(f"Build inputs started by {self.aspect}")
        datasets = []
        
        for dataset_name, dataset in self.chat.items():
            for dialog_index, dialog in enumerate(dataset):
                utterance = dialog["utterances"][0] 
                reward_buyer = dialog["reward_buyer"]
                reward_seller = dialog["reward_seller"]
                instance = None
                supervised_inputs_seller = []
                supervised_inputs_buyer = []
                if self.aspect == 'Seller': 
                    #sellers' first utterances are for buyers and vice versa.
                    scenario =  dialog['scenario']
                    scenario_other =  dialog['scenario_buyer']
                    history = None
                    if dialog['starts'] == 1:
                        history =  utterance["history"]
                        reply = utterance["candidates"][0]#doesn't matter
                        instance = build_input_from_segments(scenario, history, reply, self.tokenizer)
                        #SL, target Seller
                        for u in dialog["utterances"]:
                            h = u["history"][:]
                            r = u["candidates"][0]
                            instance_other = build_input_from_segments(scenario, h, r, self.tokenizer)
                            supervised_inputs_seller.append({f'{input_name}':input_array for input_name, input_array in instance_other.items() if input_name!='token_type_ids'})
                        #SL, target Buyer
                        for u in self.chat_buyer[dataset_name][dialog_index]["utterances"]:
                            h = u["history"][:]
                            r = u["candidates"][0]
                            instance_other = build_input_from_segments_buyer(scenario_other, h, r, self.tokenizer_buyer)
                            supervised_inputs_buyer.append({f'{input_name}':input_array for input_name, input_array in instance_other.items() if input_name!='token_type_ids'})
                
                #Buyer
                else:
                    scenario =  dialog["scenario_buyer"]
                    scenario_other =  dialog['scenario']
                    if dialog['starts'] == 0:
                        history = utterance["history"]
                        reply = utterance["candidates"][0]#doesn't matter
                        instance = build_input_from_segments_buyer(scenario, history, reply, self.tokenizer)
                        #SL, target Seller
                        for u in self.chat_seller[dataset_name][dialog_index]["utterances"]:
                            h = u["history"][:]
                            r = u["candidates"][0]
                            instance_other = build_input_from_segments(scenario_other, h, r, self.tokenizer_seller)
                            supervised_inputs_seller.append({f'{input_name}':input_array for input_name, input_array in instance_other.items() if input_name!='token_type_ids'})
                        #SL, target Buyer
                        for u in dialog["utterances"]:
                            h = u["history"][:]
                            r = u["candidates"][0]
                            instance_other = build_input_from_segments_buyer(scenario, h, r, self.tokenizer)
                            supervised_inputs_buyer.append({f'{input_name}':input_array for input_name, input_array in instance_other.items() if input_name!='token_type_ids'})
                        
                
                if instance:
                    starts = 'Buyer' if dialog['starts'] == 0 else 'Seller'
                    datasets.append({'input_ids':instance['input_ids'],
                                     'scenario':scenario,
                                     'scenario_other':scenario_other,
                                     'reward_buyer':reward_buyer,
                                     'reward_seller':reward_seller,
                                     'quality_profit':dialog['quality_profit'],
                                     'preference':dialog['preference'],
                                     'starts':starts,
                                     'supervised_inputs_seller':supervised_inputs_seller,
                                     'supervised_inputs_buyer':supervised_inputs_buyer,
                                     'max_reward_b':self.tokenizer.decode(dialog["max_reward_b"][0]),
                                     'max_reward_s':self.tokenizer.decode(dialog["max_reward_s"][0]),
                                     'win_win_choice':self.tokenizer.decode(dialog["win_win_choice"][0])})
                    
        return datasets
    
    def check_selection(self, output):
        if self.aspect == 'Seller':
            return False
        if '<selection>' in output.split():
            return True 
        return False
     
def preprocessing_for(agent:Agent, outputs, history, scenario):
    reply = [''] #dummy    
    outputs_ids = agent.tokenizer.encode(outputs) # encode == convert_tokens_to_ids(agent.tokenizer.tokenize(outputs_plaintext))
    if len(history)>0:
        history[-1] = history[-1][:-1]  #exclud inner <eos>
    history = history + [outputs_ids]
    instance = agent.build_input(scenario, history, reply, agent.tokenizer,with_eos=False) 
    input_ids = instance['input_ids']
    input_tensor = torch.tensor([input_ids]).to(agent.device)
    return input_ids,input_tensor, history

def get_output_dict(agent:Agent, input_ids, outputs, reward=None, selection=None):
    if selection:
        out_dict = {'input_ids':agent.tokenizer.decode(input_ids),
                'labels': outputs,
                'reward': reward
                }

    else:
        out_dict = {'input_ids':agent.tokenizer.decode(input_ids),
                'labels': outputs}
    return out_dict

def get_output_dict_v(agent:Agent, input_ids, outputs, reward=None, selection=None):
    if selection:
        out_dict = {'input_ids':agent.tokenizer.decode(input_ids),
                'labels': outputs,
                'reward': reward,
                'selection':selection}

    else:
        out_dict = {'input_ids':agent.tokenizer.decode(input_ids),
                'labels': outputs}
    return out_dict

def get_selection(outputs):
    outputs_split = outputs.split()
    selection_idx = outputs_split.index('<selection>')
    #if selection_idx == len(outputs_split)-1:
    #    outputs_split.append('nothing')
    selection = 'OutOfIndex'
    if selection_idx < len(outputs_split)-1:
        selection = outputs_split[selection_idx+1]

    return selection



def use_small_testing_data(args):
    args.dataset_path = '../data/dataset_valid20_reward.json'
    args.dataset_cache = '../data/dataset_valid20_reward_cache'
    args.dataset_buyer_path = "../data/dataset_valid20_reward_buyer_reply.json"
    args.dataset_buyer_cache = '../data/dataset_valid20_reward_buyer_reply_cache'
        
    return args

def use_medium_testing_data(args):
    args.dataset_path = '../data/dataset_valid100_reward.json'
    args.dataset_cache = '../data/dataset_valid100_reward_cache'
    args.dataset_buyer_path = "../data/dataset_valid100_reward_buyer_reply.json"
    args.dataset_buyer_cache = '../data/dataset_valid100_reward_buyer_reply_cache'
        
    return args


def get_dialog_reward_of_one_sample(selection:str, max_score_selection:list)->torch.Tensor:
    if selection in max_score_selection:
        return torch.Tensor([1.0])
    if selection in ['apples', 'bananas', 'oranges']:
        return torch.Tensor([0.0])
    return torch.Tensor([-1.0])

def generate_dialog(starter, follower, dataset, args, max_round=10)->list:
    _input_ids = dataset['input_ids']
    scenario_starter = dataset['scenario']
    scenario_follower = dataset['scenario_other']
    max_score_s_selection = dataset['win_win_choice']#list of str
    max_score_b_selection = dataset['win_win_choice']#str: apples/bananas/oranges

    _input_tensor = torch.tensor([_input_ids]).to(args.device)
    history = []#the dialog generated so far.
    output_list_seller = []
    output_list_buyer = []
    verbose_list = []
    decided = False
    round_ = 0
    def add_previous_reward(output_list:list, dialog_reward:torch.Tensor):
        for data_dictionary in output_list:
            data_dictionary['reward'] =  dialog_reward
            
    
    while round_ < max_round and not decided:
        outputs_for_history, outputs_for_labels = starter.talks(_input_tensor, args)
        #starter talks
        if starter.check_selection(outputs_for_labels):#if buyer starts
            selection = get_selection(outputs_for_labels)
            dialog_reward_seller = get_dialog_reward_of_one_sample(selection, max_score_s_selection)
            dialog_reward_buyer  = get_dialog_reward_of_one_sample(selection, max_score_b_selection)
            add_previous_reward(output_list_seller, dialog_reward_seller)
            add_previous_reward(output_list_buyer, dialog_reward_buyer)
            add_previous_reward(verbose_list, dialog_reward_seller)#for printing seller
            
            output_list_buyer.append(get_output_dict(starter,
                                                   _input_ids, 
                                                   outputs_for_labels, 
                                                   reward=dialog_reward_buyer, 
                                                   selection=selection))
            verbose_list.append(get_output_dict_v(starter, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   reward=dialog_reward_seller, 
                                                   selection=selection))
            
            decided = True #actually this is dummy
            break
        else:# seller starts or no selection
            if starter.aspect == 'Seller':
                output_list_seller.append(get_output_dict(starter, 
                                                   _input_ids,
                                                   outputs_for_labels,  
                                                   selection=None))
            else:# buyer
                output_list_buyer.append(get_output_dict(starter, 
                                                   _input_ids,
                                                   outputs_for_labels,  
                                                   selection=None))
            verbose_list.append(get_output_dict_v(starter, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   selection=None))        
        #follower talks
        #prepare
        _input_ids, _input_tensor, history = preprocessing_for(follower, outputs_for_history, history, scenario_follower)
        outputs_for_history, outputs_for_labels = follower.talks(_input_tensor, args)
        if follower.check_selection(outputs_for_labels):#if follower is Buyer and selected
            selection = get_selection(outputs_for_labels)
            dialog_reward_seller = get_dialog_reward_of_one_sample(selection, max_score_s_selection)
            dialog_reward_buyer  = get_dialog_reward_of_one_sample(selection, max_score_b_selection)
            add_previous_reward(output_list_seller, dialog_reward_seller)
            add_previous_reward(output_list_buyer, dialog_reward_buyer)
            add_previous_reward(verbose_list, dialog_reward_seller)

            output_list_buyer.append(get_output_dict(follower, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   reward=dialog_reward_buyer, 
                                                   selection=selection))
            verbose_list.append(get_output_dict_v(follower, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   reward=dialog_reward_seller, 
                                                   selection=selection))
            decided = True
            break
    
        else:
            if follower.aspect == 'Seller':
                output_list_seller.append(get_output_dict(follower, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   selection=None))
            else:#buyer
                output_list_buyer.append(get_output_dict(follower, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   selection=None))
            verbose_list.append(get_output_dict_v(follower, 
                                                   _input_ids,
                                                   outputs_for_labels, 
                                                   selection=None))
        #when reaching the max number of iteration
        if round_ == max_round-1:
            dialog_reward = get_dialog_reward_of_one_sample('None', max_score_s_selection)
            add_previous_reward(output_list_seller, dialog_reward)
            add_previous_reward(output_list_buyer, dialog_reward)
            add_previous_reward(verbose_list, dialog_reward)

            
        #prepare starter's input
        _input_ids, _input_tensor, history = preprocessing_for(starter, outputs_for_history, history, scenario_starter)
        round_ += 1
    return output_list_seller, output_list_buyer, verbose_list #list of dictionary


def generate_sampled_dialogs(dataset, starter:Agent, follower:Agent, 
                             args=None)->list:
    #generate data the exceed the the number of batch size
    dialogs_list_seller = []
    dialogs_list_buyer = []
    verbose_list = []
    start_time = time.time()
    
            
            
    #if exceed the size cut the list
    #len(dialogs_list_buyer) is supposed larger than seller's
    while len(dialogs_list_seller) < args.batch_size:
        pair_list_seller, pair_list_buyer, c_list = generate_dialog(starter, follower, dataset, args,  max_round=10)
        dialogs_list_seller += pair_list_seller
        dialogs_list_buyer  += pair_list_buyer
        verbose_list += c_list
    dialogs_list_seller = dialogs_list_seller[:args.batch_size]#TODO: set these two dialog list from sampling
    dialogs_list_buyer  = dialogs_list_buyer[:args.batch_size]
    spend_time = time.time() - start_time
    description = f'{len(dialogs_list_seller) + len(dialogs_list_buyer)} pairs finished in {spend_time:.2f} secs'
    
    return dialogs_list_seller, dialogs_list_buyer, verbose_list, description

def get_SL_inputs(input_list:list, device):
    assert len(input_list)>0
    output_dict = {}
    for d in input_list:
        for k,v in d.items():
            if k not in output_dict.keys():
                output_dict[k] = [v]
            else:
                output_dict[k].append(v)
    for k,v in output_dict.items():
        output_dict[k] = torch.tensor(v).to(device)
        
    return output_dict

def pad_list_of_inputs(input_list:list):
    max_lens = {}
    for input_dict in input_list:
        for name, value in input_dict.items():
            if name in max_lens.keys():
                if len(value) > max_lens[name] :
                    max_lens[name] = len(value)
            else:
                max_lens[name] = len(value)
    output_list = []
    for input_dict in input_list:
        output_dict = {}
        for name, value in input_dict.items():
            len_difference = max_lens[name] - len(value)
            pad_item = [-100] if name == 'labels' else [0]
            output_dict[name] = value + (pad_item*len_difference)
        output_list.append(output_dict)
    return output_list
            


def pad_batch(agent,batch,baseline_reward):   
    tensor_batch = agent.tokenizer(batch['input_ids'], padding=True, return_tensors='pt')
    tensor_batch['labels'] = agent.tokenizer(batch['labels'], padding=True).input_ids
    #the prob. of -100 should not be count
    batch_size = len(batch['labels'])
    snt_size = len(tensor_batch['labels'][0])
    tensor_batch['end_index'] = [ [snt_size-1] for i in range(batch_size)]
    for x,sequence in enumerate(tensor_batch['labels']):
        for i in range(len(sequence)-1,0,-1):
            #find padding
            if sequence[i] == 0:
                sequence[i] = -100
            else:
                tensor_batch['end_index'][x][0]=i
                break
    
    tensor_batch['labels'] = torch.tensor(tensor_batch['labels'])
    tensor_batch['reward'] = batch['reward']
    tensor_batch['end_index'] = torch.tensor(tensor_batch['end_index'])
    tensor_batch['baseline_reward'] = baseline_reward
    return tensor_batch

def generate_batch_sampled_dialogs(input_dict, Buyer:Agent, Seller:Agent, args):
    if input_dict['starts'] == 'Buyer':
        batch_sampled_dialogs_seller, batch_sampled_dialogs_buyer, verbose_list, time_passed = generate_sampled_dialogs(input_dict,
                                                      starter=Buyer, 
                                                      follower=Seller, 
                                                      args=args)
    else:
        batch_sampled_dialogs_seller, batch_sampled_dialogs_buyer, verbose_list, time_passed = generate_sampled_dialogs(input_dict,
                                                      starter=Seller, 
                                                      follower=Buyer, 
                                                      args=args)
    
    return batch_sampled_dialogs_seller, batch_sampled_dialogs_buyer, verbose_list, time_passed, input_dict['supervised_inputs_seller'], input_dict['supervised_inputs_buyer']

def get_reward_baseline_queue( baseline_type='None', previous:int=3):
    reward = torch.Tensor([0.0])
    baseline_queue = deque(maxlen=1)
    baseline_queue.append(reward)
    if baseline_type == 'previous':
        baseline_queue = deque(maxlen=previous)
        for i in range(baseline_queue.maxlen):
            baseline_queue.append(reward)
        
    return baseline_queue

def get_dialog_reward(flatted_total_sampled_dialogs):
    reward = torch.Tensor([0.0])
    for dictionary in flatted_total_sampled_dialogs:
        reward += dictionary['reward']
    return reward.item()/len(flatted_total_sampled_dialogs)



def write_dialogs(batch_dialogs, baseline_reward, file_prefix, args, epoch):

    def write_info(utterance:dict,f):
        batch_size = len(utterance['labels'])
        for i in range(batch_size):
            f.write(f'input_ids:\n{utterance["input_ids"][i]}\n')
            f.write(f'labels:\n{utterance["labels"][i]}\n')
            f.write(f'reward:\n{utterance["reward"][i].item()}\n')
            f.write(f'baseline:\n{baseline_reward.item()}\n')
            f.write(f'R(tau) - b:\n{utterance["reward"][i].item()-baseline_reward.item()}\n\n')
    
    Path(f"./{file_prefix}").mkdir(parents=True, exist_ok=True)
    small = '_small' if args.use_small_testing_data else ''
    medium = f'_medium' if args.use_medium_testing_data else ''
    file_path = f'{file_prefix}/dialogs_for_RL_{epoch}{small}{medium}.txt'
    if os.path.isfile(file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('---------------------------- batch separation (input condition changed) ---------------------------\n')
            write_info(batch_dialogs,f)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            write_info(batch_dialogs,f)


def print_dialogs(info, batch_dialogs, file_prefix, args, epoch, dialog_num):

    def write_info(batch_dialogs,f):
        f.write(f'#{dialog_num}\n')
        for utterance in batch_dialogs:
            
            f.write(f'{utterance["labels"]}\n\n')
            if 'selection' in utterance.keys():
                f.write(f'''selection: {utterance['selection']}\n''')
                f.write(f'''reward:\n{utterance["reward"].item()}\n{'*'*25}\n''')
            
    
    Path(f"./{file_prefix}").mkdir(parents=True, exist_ok=True)
    small = '_small' if args.use_small_testing_data else ''
    medium = f'_medium' if args.use_medium_testing_data else ''
    file_path = f'{file_prefix}/dialogs_for_RL_{epoch}{small}{medium}_with_n.txt'
    if os.path.isfile(file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('---------------------------- batch separation (input condition changed)-------------------\n')
            f.write(get_scenario_table(info['preference'], info['quality_profit']))
            write_info(batch_dialogs,f)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            #write hyperparameter here
            f.write(get_scenario_table(info['preference'], info['quality_profit']))
            write_info(batch_dialogs,f)

def print_labels_SL(info, agent, batch_dialogs, file_prefix, args, epoch, dialog_num):

    def write_info(batch_dialogs,f):
        f.write(f'#{dialog_num}\n')
        for utterance in batch_dialogs:
            
            f.write(f'{agent.tokenizer.decode(utterance["labels"])}\n')
        f.write('\n\n')
    
    Path(f"./{file_prefix}").mkdir(parents=True, exist_ok=True)
    small = '_small' if args.use_small_testing_data else ''
    medium = f'_medium' if args.use_medium_testing_data else ''
    file_path = f'{file_prefix}/labels_for_SL_{epoch}{small}{medium}_with_n.txt'
    if os.path.isfile(file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('---------------------------- batch separation (input condition changed)-------------------\n')
            f.write(get_scenario_table(info['preference'], info['quality_profit']))
            write_info(batch_dialogs,f)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            #write hyperparameter here
            f.write(get_scenario_table(info['preference'], info['quality_profit']))
            write_info(batch_dialogs,f)

def train_PG(Agent, RL_dataloader, optimizer,baseline_reward_queue, args):
    to_experiment = {}
    #Seller, once a batch actually
    for batch in RL_dataloader:
        batch_size = len(batch['labels'])
        baseline_reward = sum(baseline_reward_queue)/baseline_reward_queue.maxlen
        tensor_batch = pad_batch(Agent,batch, baseline_reward)
        if args.reward_baseline_type == 'previous':
            #put the avg reward in this batch to
            baseline_reward_queue.append(sum(tensor_batch['reward'])/batch_size)
        if args.reward_baseline_type == 'previous':
            to_experiment['avg_b_Rb'] = sum(tensor_batch['reward'])/batch_size -baseline_reward
                
        optimizer.zero_grad()
        Agent.model.train()
        tensor_batch = {k: v.to(args.device) for k, v in tensor_batch.items()}
        if args.reward_baseline_type !='None':
            tensor_batch['baseline_type'] = args.reward_baseline_type
        outputs = Agent.model(**tensor_batch)
        #changed the loss calculation in modeling_T5.py for env_PG_...
        loss = outputs.loss #args.gradient_accumulation_steps
        reward = -1*loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent.model.parameters(), 1.0)
        optimizer.step()
        to_experiment['batch_R_bar'] = reward
    return to_experiment

def train_SL(Agent, optimizer, supervised_inputs, save_dir, epoch, dialog_index, input_dict,args):
    optimizer.zero_grad()
    if args.write_dialogs:
        print_labels_SL(input_dict, Agent, supervised_inputs, save_dir, args, epoch+1, dialog_index)
    supervised_inputs = pad_list_of_inputs(supervised_inputs)
    tensor_batch_SL = get_SL_inputs(supervised_inputs, args.device)
    outputs_SL = Agent.model(**tensor_batch_SL, Is_PG=False)
    loss_SL = outputs_SL.loss
    loss_SL.backward()
    torch.nn.utils.clip_grad_norm_(Agent.model.parameters(), 1.0)
    optimizer.step()
    return loss_SL.item()



def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/dataset_train423_reward.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='../data/dataset_train423_reward_cache', help="Path or url of the dataset cache")
    parser.add_argument("--dataset_buyer_path", type=str, default="../data/dataset_train423_reward_buyer_reply.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_buyer_cache", type=str, default='../data/dataset_train423_reward_buyer_reply_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="t5-base", help="Model type ")  
    parser.add_argument("--model_checkpoint_buyer", type=str, default="~/model_folder/model.pt", help="Path, url or short name of the model")
    parser.add_argument("--model_checkpoint_seller", type=str, default="~/model_folder/model.pt", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--show_only_pretrained", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=6.25e-6, help="Learning rate")
    parser.add_argument("--value_only", action='store_true', help="If assigned get rid of description quality... profit...")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_generate_length", type=int, default=35, help="Number of max_generate_length")
    parser.add_argument("--min_generate_length", type=int, default=3, help="Number of min_generate_length")
    parser.add_argument("--use_small_testing_data", action='store_true', help="If assigned, test on sampled set.")
    parser.add_argument("--use_medium_testing_data", action='store_true', help="If assigned, test on medium sampled set.")
    parser.add_argument("--write_dialogs", action='store_true', help="If assigned, write out the generated dialogs.")
    parser.add_argument("--reward_baseline_type", type=str, default='previous', help="None/previous/seq_normalize/print")
    parser.add_argument("--shuffle", action='store_true', help="If assigned, the order of input condition(preference, quality, profit) is shuffled")
    args = parser.parse_args()

    if args.use_small_testing_data:
        args = use_small_testing_data(args)
    elif args.use_medium_testing_data:
        args = use_medium_testing_data(args)
    

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(pformat(args))

    Buyer = Agent(args, aspect='Buyer')
    Seller = Agent(args, aspect='Seller')
    all_first_talks = Buyer.first_talks + Seller.first_talks
    if args.shuffle:
        random.shuffle(all_first_talks)
    
    #Seller
    optimizer_seller = AdamW(Seller.model.parameters(), lr=args.lr, correct_bias=True)
    #Buyer
    optimizer_buyer = AdamW(Buyer.model.parameters(), lr=args.lr, correct_bias=True)

    file_prefix_seller = make_logdir(f'RL_Both_winwin_{args.reward_baseline_type}_b{args.batch_size}_lr{args.lr}/Seller/epoch_00')
    file_prefix_buyer = make_logdir(f'RL_Both_winwin_{args.reward_baseline_type}_b{args.batch_size}_lr{args.lr}/Buyer/epoch_00')
    
    baseline_reward_queue_seller = get_reward_baseline_queue( baseline_type=args.reward_baseline_type)
    baseline_reward_queue_buyer = get_reward_baseline_queue( baseline_type=args.reward_baseline_type)
    step=0 
    for epoch in range(args.n_epochs):
        save_dir_seller = file_prefix_seller[:-2]+f'{epoch+1:02d}'
        save_dir_buyer = file_prefix_buyer[:-2]+f'{epoch+1:02d}'
        progress_bar = tqdm(range(len(all_first_talks)))
        for i,input_dict in enumerate(all_first_talks):
            flatted_total_sampled_dialogs_seller, flatted_total_sampled_dialogs_buyer, verbose_list, time_passed, supervised_inputs_seller, supervised_inputs_buyer = generate_batch_sampled_dialogs(input_dict, Buyer, Seller, args)
            if args.write_dialogs:
                print_dialogs(input_dict,verbose_list, save_dir_seller, args, epoch+1,i)
            #Seller
            dialog_reward_seller = get_dialog_reward(flatted_total_sampled_dialogs_seller)
            RL_dataloader_seller = DataLoader(dataset=flatted_total_sampled_dialogs_seller,batch_size=args.batch_size, shuffle=False)       
            
            #Seller PG, once a batch actually
            log_experiment = train_PG(Seller, 
                                    RL_dataloader_seller, 
                                    optimizer_seller,
                                    baseline_reward_queue_seller,
                                    args)
            
            progress_bar.set_description(f'epoch:{epoch+1}, {time_passed},R_bar={log_experiment["batch_R_bar"]}')
            progress_bar.update(1)
            time.sleep(0.025)#for progress bar

            #Seller, supervised learning
            loss_SL = train_SL(Seller, 
                               optimizer_seller, 
                               supervised_inputs_seller,
                               save_dir_seller, 
                               epoch, 
                               i,
                               input_dict,
                               args)
            
            #Buyer, PG
            dialog_reward_buyer = get_dialog_reward(flatted_total_sampled_dialogs_buyer)
            RL_dataloader_buyer = DataLoader(dataset=flatted_total_sampled_dialogs_buyer,batch_size=args.batch_size, shuffle=False)       
            log_experiment = train_PG(Buyer, 
                                    RL_dataloader_buyer, 
                                    optimizer_buyer,
                                    baseline_reward_queue_buyer,
                                    args)           
            #Buyer, SL
            loss_SL = train_SL(Buyer, 
                               optimizer_buyer, 
                               supervised_inputs_buyer,
                               save_dir_buyer, 
                               epoch, 
                               i,
                               input_dict,
                               args)            

            step += 1
        Seller.model.save_pretrained(save_dir_seller)
        Buyer.model.save_pretrained(save_dir_buyer)

        progress_bar.close()

        
                    
    

if __name__ == '__main__':
    main()
