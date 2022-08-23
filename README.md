# Fruit_Stand

This site was built for experiments in this paper: [Conversational AI for Positive-sum Retailing under Falsehood Control](https://aclanthology.org/2022.nlp4convai-1.3/), accepted in [NLP4ConvAI 2022](https://sites.google.com/view/4thnlp4convai).

## 1. Fine-tune a baseline model for buyer/ seller
Use ```train_buyer/ train_seller.py``` to build a baseline buyer/seller model.

Run script like: 
```
python train_buyer.py --dataset_path "data/<buyer/seller_aspect>/train_valid.json" n_epoch 20
```

You may use test.py or test_batch.py to get the perplexity for each model.

Run script like: 
```
python test_batch.py --models_dir "~/model_folder" --aspect_buyer --train_valid
```
Note that we pick the model with lost perplexity in training epochs as our baseline model.

Run 
```self_play.py ```
to see how a buyer model interact with a seller(retailer) model.

## 2. Reinforce retailing policies
Use program with prefix PG_ (e.g. PG_Interleaved_both.py) to enhance representative retailer and buyer models.

Run code like: 
```
python PG_Interleaved_both.py --dataset_path "~/dataset_train423_reward.json" --dataset_buyer_path "/dataset_train423_reward_buyer_reply.json" --n_epochs 10 --batch_size 32
```

Run 
```self_play_tuned_both.py``` 
to see how they interact with each other, or you may use self_play_batch_both.py to see pairwisely.

## 3. use program with prefix Liar_ to do deduction mechanism

Run 
```Liar_PG_Interleaved_both.py

self_play_tuned_both.py 
```
to see how they interact with each other, or you may use self_play_batch_both.py to see pairwisely.




Please cite our paper if you find the data and code helpful, thank you!
```
@inproceedings{liao-etal-2022-conversational,
    title = "Conversational {AI} for Positive-sum Retailing under Falsehood Control",
    author = "Liao, Yin-Hsiang  and
      Dong, Ruo-Ping  and
      Chang, Huan-Cheng  and
      Ma, Wilson",
    booktitle = "Proceedings of the 4th Workshop on NLP for Conversational AI",
    year = "2022",
}
```
