#----#
#usage: let model pairs affer RL talk to each other.
#----#
import os
import glob
from argparse import ArgumentParser
from time import sleep
def main(args):
    #files = glob.glob(PATH+'/*.pt')#[f for f in os.listdir(PATH) if re.match(r'*/.pt', f)]
    for n_epoch in range(1,11,1):
        path_s = f'{args.model_checkpoint_seller}{n_epoch:02}/'
        path_b = f'{args.model_checkpoint_buyer}{n_epoch:02}/'
        if os.path.isfile(path_s+'pytorch_model.bin') and os.path.isfile(path_b+'pytorch_model.bin'):
            os.system(f'python self_play_tuned_both.py --bs_num {n_epoch} --model_checkpoint_seller {path_s} --model_checkpoint_buyer {path_b}')
        else:
            print(f'''models #{n_epoch:02} doesn't exist''')
        sleep(2)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_seller", type=str, 
        default="~/model_folder/Seller/epoch_", 
        help="Path, url or short name of the seller model")
    parser.add_argument("--model_checkpoint_buyer", type=str, 
        default="~/model_folder/Buyer/epoch_", 
        help="Path, url or short name of the buyer model")
    args = parser.parse_args()
    main(args)
