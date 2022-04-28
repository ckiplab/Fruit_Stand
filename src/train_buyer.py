#---#
#usage: fine-tune pretrained (T5) model from a buyer's perspective.
#---#
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver#, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import AdamW, T5Model, T5Tokenizer, T5ForConditionalGeneration, WEIGHTS_NAME, CONFIG_NAME

from utils import get_dataset, make_logdir

SPECIAL_TOKENS_BUYER = ["<bos>", "<eos>", "<Buyer>", "<Seller>","<selection>", "<sep>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token':'<sep>',
                         'additional_special_tokens': ['<Buyer>', '<Seller>','<selection>']}
MODEL_INPUTS = ["input_ids", "token_type_ids", 'attention_mask', 'labels']
PADDED_INPUTS = ["input_ids", "token_type_ids", 'attention_mask','labels']

logger = logging.getLogger(__name__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        if name!='labels':
            dataset[name] = [x + [padding if name!= 'attention_mask' else 0] * (max_l - len(x)) for x in dataset[name]]
        else:
            dataset[name] = [x +[-100]*(max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_buyer(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = tokenizer.vocab_size #T5
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments_buyer( scenario, history, reply, tokenizer, with_eos=True):
    #note the order of (speaker2, speaker1) is swapped
    bos, eos, speaker2, speaker1,_,_ = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS_BUYER[:-1])
    history[-1] = history[-1] + ([eos] if with_eos else [])
    sequence = [[bos]+list(chain(*scenario))]  + history
    #add <Buyer>, <Seller> except for the scenario
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence)) #note that the length of result after decoding may vary.
    instance["labels"] = [bos,speaker2] + reply +([eos] if with_eos else [])
    instance["token_type_ids"] = [speaker2 for _ in sequence[0]]+[speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
    instance["attention_mask"] = [1]*len(instance["input_ids"])
    return instance


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    chat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in chat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            scenario = dialog["preference"] if args.value_only else dialog["scenario_buyer"]
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                reply = utterance["candidates"][-num_candidates]
                instance = build_input_from_segments_buyer(scenario, history, reply, tokenizer)
                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)
                    
        datasets[dataset_name]["n_candidates"] = num_candidates
                
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS_BUYER[-1]))
        #convert inputs into tensor
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])            
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True)

    logger.info("Train dataset (Batch, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/dataset_train3386_valid423_reward_buyer_reply.json", help="Path or url of the dataset. ")
    parser.add_argument("--dataset_cache", type=str, default='../data/dataset_train3386_valid423_reward_buyer_reply_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="t5-base", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=1, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=5, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--value_only", action='store_true', help="If true get rid of description quality... profit...")
    
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = T5Tokenizer if "t5" in args.model_checkpoint  else T5TokenizerFast # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = T5ForConditionalGeneration if "t5" in args.model_checkpoint else T5Model
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_buyer(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    
    logger.info("Prepare datasets")
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        #under construction
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        #token_tpye_ids is a part of legacy code
        input_ids, token_type_ids, attention_mask, labels  = batch
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss /args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, token_type_ids,attention_mask, labels = batch
            logger.info(tokenizer.decode(input_ids[0, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            
            logits = outputs.logits
            logits_flat_shifted = logits[..., :].contiguous().view(-1, logits.size(-1))
            labels_flat_shifted = labels[..., :].contiguous().view(-1)
            return (logits_flat_shifted, logits), (labels_flat_shifted, labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    
    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0]))}#-100 was for mask
    #,"accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    #,"average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(f'{args.model_checkpoint}_buyer')
        tb_logger = TensorboardLogger(log_dir)
        objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer}

        gst = lambda *_: trainer.state.epoch
        checkpoint_handler = Checkpoint(objects_to_checkpoint, DiskSaver(f'{log_dir}'), n_saved=args.n_epochs, global_step_transform=gst, score_name=f'_Buyer_avgPPL_{metrics["average_ppl"]}')
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)  

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
