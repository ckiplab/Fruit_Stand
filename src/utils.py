# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from transformers import cached_path


logger = logging.getLogger(__file__)



def get_dataset(tokenizer, dataset_path, dataset_cache):
    dataset_path = dataset_path 
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using another cache for T5 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, int):
                return obj
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir

def post_process(snt:str, aspect:str)->str:
    snt_list = snt.split()
    #for history, we don't need <Seller>, <Buyer>, <bos>, <pad>, or <eos>
    if '<Seller>' in snt_list:
        bos_index = snt_list.index('<Seller>')+1 #actullay the first word, not <bos> token
    elif '<Buyer>' in snt_list:
        bos_index = snt_list.index('<Buyer>')+1
    elif '<bos>' in snt_list:
        bos_index = snt_list.index('<bos>')+1 
    elif '<pad>' in snt_list:
        bos_index = snt_list.index('<pad>')+1
    else:
        bos_index = 0
    if '<eos>' in snt_list:
        eos_index = snt_list.index('<eos>')+1
    else:
        eos_index = len(snt_list)

    outputs_for_history = ' '.join(snt_list[bos_index:eos_index])

    #for_labels
    snt_list = ['<bos>', f'<{aspect}>']+snt_list[bos_index:eos_index]
    if '<eos>' not in snt_list:
        snt_list.append('<eos>')
        eos_index = len(snt_list)
    outputs_for_labels = ' '.join(snt_list[:eos_index]) 
    
    return outputs_for_history, outputs_for_labels

if __name__ == '__main__':
    from transformers import T5Tokenizer, T5ForConditionalGeneration

