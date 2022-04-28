#Author: YHL

#Purpose: Liar detection, rule-based and NLI
from transformers import RobertaTokenizer, RobertaModelWithHeads
import torch

#liar detector
#any statement is against the truth
#NLI
def load_NLI_model_tokenizer(name='roberta-base'):
    #model = AutoModelWithHeads.from_pretrained(name)
    model = RobertaModelWithHeads.from_pretrained(name)
    tokenizer = RobertaTokenizer.from_pretrained(name)
    adapter_name = model.load_adapter(f"AdapterHub/{name}-pf-mnli", source="hf")
    model.active_adapters = adapter_name
    return model, tokenizer

def classify(model, tokenizer,snts="Apples are the best. Oranges are the best")-> int:
    Y = tokenizer(snts, return_tensors='pt')
    X = {}
    outputs = model(**Y)
    #print(f'decode input_ids: {tokenizer.decode(Y["input_ids"][0])}')
    #print(f'model outputs: {outputs}')
    softM = torch.nn.Softmax(dim=-1)
    outputs_logits = softM(outputs.logits)
    #print(f'after softmax: {outputs_logits}')
    return torch.argmax(outputs_logits, dim=-1).to('cpu').item()


def get_indices(_value:int, _list:list):
    indices=[]
    for i,v in enumerate(_list):
        if _value == v:
            indices.append(i)
    assert len(indices)>0,'_value should exist'
    return indices

def get_max_quality_idx(quality_list:list):
    max_value_s = max(quality_list)#max quality value 

    return get_indices(max_value_s, quality_list)

def get_min_quality_idx(quality_list:list):
    min_value_s = min(quality_list)#max quality value 

    return get_indices(min_value_s, quality_list)


def generate_positive_premise(quality_list:list):
    idx2Fruit = {0:'Apples', 1:'Bananas', 2:'Oranges'}
    max_quality_idx = get_max_quality_idx(quality_list)
    if len(max_quality_idx) == 1:
        f_id = max_quality_idx[0]
        return f'{idx2Fruit[f_id]} are the best.'
    if len(max_quality_idx) == 2:
        return f'{idx2Fruit[max_quality_idx[0]]} and {idx2Fruit[max_quality_idx[1]].lower()} are the best.'
    if len(max_quality_idx) == 3:
        return f'All fruits are equally good.'

def generate_negative_premise(quality_list:list):
    idx2Fruit = {0:'Apples', 1:'Bananas', 2:'Oranges'}
    min_quality_idx = get_min_quality_idx(quality_list)
    if len(min_quality_idx) == 1:
        f_id = min_quality_idx[0]
        return f'{idx2Fruit[f_id]} are the worst.'
    if len(min_quality_idx) == 2:
        return f'{idx2Fruit[min_quality_idx[0]]} and {idx2Fruit[min_quality_idx[1]].lower()} are the worst.'
    if len(min_quality_idx) == 3:
        return f'All fruits are euqally bad.'



def Is_Liar_NLI(utterance_list, quality_list, NLI_model, tokenizer):
    def Is_Contradiction(utterance:str, quality_list:list,NLI_model, tokenizer):
        positive_premise = generate_positive_premise(quality_list)
        negative_premise = generate_negative_premise(quality_list)
        utterance = utterance.capitalize()
        if utterance[-1] != '.':
            utterance = utterance + '.'
        
        positive_snt = positive_premise + ' ' + utterance
        negative_snt = negative_premise + ' ' + utterance
        if classify(NLI_model, tokenizer, positive_snt) == 2 or classify(NLI_model, tokenizer, negative_snt) == 2:
            return True
        return False
    
    previous = False
    for utterance in utterance_list:
        previous = previous or Is_Contradiction(utterance, quality_list,NLI_model, tokenizer)
            
    return previous
        

    

    #generate_premise
    #contradict to any utterance

def IsBest(fruit:str, utterance:str):
    
    for b in utterance.split('.'):
        if (f'{fruit}s are the best seller' in b):
            return False
        if (f'{fruit} is the best' in b) or (f'{fruit}s are the best' in b) or (f'{fruit}s are your best' in b) or (f'best {fruit}' in b)or (f'best are the {fruit}' in b) or (f'{fruit}s are my best' in b):
            return True
        #(quality,highest);freshest
    return False

def IsWorst(fruit:str, utterance:str):
    
    for b in utterance.split('.'):
        if (f'{fruit} is the worst' in b) or (f'{fruit}s are the worst' in b) or (f'{fruit}s are your worst' in b) or(f'worst {fruit}' in b)or (f'worst are the {fruit}' in b)or (f'{fruit}s are my worst' in b) :
            return True
    return False
def IsSuperlative(utterance:str):
    fruit_tuple = tuple(['apple','banana','orange'])
    for fruit in fruit_tuple:
        if IsBest(f'{fruit}', utterance):
            return True    
        if IsWorst(f'{fruit}', utterance):
            return True
    return False



#rule_based
def IsLiar_rule_based(utterance_list,  quality_list:list):
    def against_fact(utterance:str, quality_list:list):
        fruit_dict = {'apple':0, 'banana':1, 'orange':2}
        for fruit, code in fruit_dict.items():
            if IsBest(f'{fruit}', utterance) and quality_list[code]!=max(quality_list):
                return True
                  
            if IsWorst(f'{fruit}', utterance) and quality_list[code]!=min(quality_list):
                return True
            
        return False
    
    previous = False
    for utterance in utterance_list:
        previous = previous or against_fact(utterance, quality_list)
            
    return previous



