#------#
#useage: test each checkpoint.
#------#
import os
import glob
from argparse import ArgumentParser
from time import sleep
def main(args):
    PATH=args.models_dir
    value_only = '--value_only' if args.value_only  else ''
    is_buyer = '--aspect_buyer' if args.aspect_buyer else ''
    is_train_valid = '--train_valid' if args.train_valid else ''
    files = glob.glob(PATH+'/*.pt')#[f for f in os.listdir(PATH) if re.match(r'*/.pt', f)]
    for file in files:
        print(f'testing {file.split("/")[-1]}')
        os.system(f'python test.py --model_checkpoint {file} {value_only} {is_buyer} {is_train_valid}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--models_dir", type=str, default='~/model_folder')
    parser.add_argument("--value_only", action='store_true', help="If true get rid of description quality... profit...")
    parser.add_argument("--aspect_buyer", action='store_true', help="If assigned, test the buyer.")
    parser.add_argument("--train_valid", action='store_true', help="If assigned, test on training set and validation set.")
    
    args = parser.parse_args()
    main(args)
