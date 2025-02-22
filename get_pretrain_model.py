import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import sys
import random

from torch.utils.data import DataLoader

sys.path.insert(0, "/home/hbs/TS/my_p/shapeX/scr/Medformer_1/")
from models import ProtoPTST

#sys.path.append("/home/hbs/TS/my_p/shapeX/scr/Medformer")
#sys.path.insert(0, "/home/hbs/TS/my_p/shapeX/scr/PatchTST/PatchTST_supervised")

#sys.path.insert(0, "/home/hbs/TS/my_p/shapeX/scr/PatchTST")
#from PatchTST_supervised.models import PatchTST

def get_pretrain_model(model_name):
    
    if model_name == "seqcomb_single_pretrain":
        device="cuda:3"
        device_n="3"
        dataset_name="freqshape_274" 
        SEQ_LEN=200
        CHANNEL=1
        DATASET_PATH="/home/hbs/TS/XTS/TimeX/datasets/SeqCombSingle"
        CLASS_NUMBER=4

        # Simulate command-line arguments
        update_dict= {
            "--random_seed": "42",
            #"--seed": "42",
            "--is_training": "1",
            "--root_path": "./dataset/",
            "--data_path": "ETTh1.csv",
            "--model_id": dataset_name,
            "--model": "PatchTST",
            "--data": dataset_name,
            "--features": "S",
            "--seq_len": SEQ_LEN,
            "--pred_len": SEQ_LEN, ##!!!!
            "--enc_in": "1",
            "--e_layers": "3",
            "--n_heads": "4",
            "--d_model": "16",
            "--d_ff": "128", # !!!
            "--dropout": "0.3",
            "--fc_dropout": "0.3",
            "--head_dropout": "0",
            "--patch_len": "1",
            "--stride": "1",
            "--des": "Exp",
            "--train_epochs": "100",
            "--itr": "1",
            "--batch_size": "16", # !!!
            "--learning_rate": "0.0001"
        }
        
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

        # random seed
        parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

        # basic config
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')

        # data loader
        parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


        # DLinear
        #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

        # PatchTST
        parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')
        parser.add_argument('--stride', type=int, default=8, help='stride')
        parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
        parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
        parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
        parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

        # Formers 
        parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size') # !!!
        parser.add_argument('--c_out', type=int, default=1, help='output size') # !!!
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
        parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=device_n, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

        #args = parser.parse_args()
        ##############-------------------------------------#############
        args_list = []
        for key, value in update_dict.items():
            args_list.append(key)
            args_list.append(str(value))  # Convert the value to a string

        # Step 3: Parse the arguments using the default ones and override with new ones from the dictionary
        args = parser.parse_args(args_list)
        
        model = PatchTST.Model(args).float()
        #model.load_state_dict(torch.load("/home/hbs/TS/my_p/shapeX/scr/PatchTST/PatchTST_supervised/checkpoints/seqcomb_Transformer_seqcomb_ftS_sl200_ll48_pl152_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"))
        model.load_state_dict(torch.load("/home/hbs/TS/my_p/shapeX/scr/PatchTST/PatchTST_supervised/checkpoints/seqcomb_PatchTST_seqcomb_ftS_sl200_ll48_pl200_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"))
        #model.load_state_dict(torch.load("/home/hbs/TS/my_p/shapeX/scr/PatchTST/PatchTST_supervised/checkpoints/freqshape_274_PatchTST_freqshape_274_ftS_sl1400_ll48_pl1400_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"))
        return model
    
    elif model_name == "freqshape_274":
        print("$"*100)
        device="cuda:3"
        device_n="3"
        dataset_name="freqshape_274" 
        SEQ_LEN=1400
        CHANNEL=1
        DATASET_PATH="/home/hbs/TS/XTS/TimeX/datasets/FreqShapeUD"
        CLASS_NUMBER=4

        # Simulate command-line arguments
        update_dict= {
            "--random_seed": "42",
            #"--seed": "42",
            "--is_training": "1",
            "--root_path": "./dataset/",
            "--data_path": "ETTh1.csv",
            "--model_id": dataset_name,
            "--model": "PatchTST",
            "--data": dataset_name,
            "--features": "S",
            "--seq_len": SEQ_LEN,
            "--pred_len": SEQ_LEN, ##!!!!
            "--enc_in": "1",
            "--e_layers": "3",
            "--n_heads": "4",
            "--d_model": "16",
            "--d_ff": "128", # !!!
            "--dropout": "0.3",
            "--fc_dropout": "0.3",
            "--head_dropout": "0",
            "--patch_len": "1",
            "--stride": "1",
            "--des": "Exp",
            "--train_epochs": "100",
            "--itr": "1",
            "--batch_size": "16", # !!!
            "--learning_rate": "0.0001"
        }
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

        # random seed
        parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

        # basic config
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')

        # data loader
        parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


        # DLinear
        #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

        # PatchTST
        parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')
        parser.add_argument('--stride', type=int, default=8, help='stride')
        parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
        parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
        parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
        parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

        # Formers 
        parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size') # !!!
        parser.add_argument('--c_out', type=int, default=1, help='output size') # !!!
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
        parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=device_n, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

        #args = parser.parse_args()
        ##############-------------------------------------#############
        args_list = []
        for key, value in update_dict.items():
            args_list.append(key)
            args_list.append(str(value))  # Convert the value to a string

        # Step 3: Parse the arguments using the default ones and override with new ones from the dictionary
        args = parser.parse_args(args_list)
        
        model = PatchTST.Model(args).float()
        model.load_state_dict(torch.load("/home/hbs/TS/my_p/shapeX/scr/PatchTST/PatchTST_supervised/checkpoints/freqshape_274_PatchTST_freqshape_274_ftS_sl1400_ll48_pl1400_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"))
        return model
    
    elif model_name == "freqshape_264":
        print("$"*100)
        device="cuda:2"
        device_n="3"
        dataset_name="freqshape_264" 
        SEQ_LEN=1400
        CHANNEL=1
        DATASET_PATH="/home/hbs/TS/XTS/TimeX/datasets/FreqShapeUD"
        CLASS_NUMBER=4

        # Simulate command-line arguments
        update_dict= {
            "--random_seed": "42",
            #"--seed": "42",
            "--is_training": "1",
            "--root_path": "./dataset/",
            "--data_path": "ETTh1.csv",
            "--model_id": dataset_name,
            "--model": "PatchTST",
            "--data": dataset_name,
            "--features": "S",
            "--seq_len": SEQ_LEN,
            "--pred_len": SEQ_LEN, ##!!!!
            "--enc_in": "1",
            "--e_layers": "3",
            "--n_heads": "4",
            "--d_model": "16",
            "--d_ff": "128", # !!!
            "--dropout": "0.3",
            "--fc_dropout": "0.3",
            "--head_dropout": "0",
            "--patch_len": "1",
            "--stride": "1",
            "--des": "Exp",
            "--train_epochs": "100",
            "--itr": "1",
            "--batch_size": "16", # !!!
            "--learning_rate": "0.0001"
        }
        parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

        # random seed
        parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

        # basic config
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')

        # data loader
        parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


        # DLinear
        #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

        # PatchTST
        parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')
        parser.add_argument('--stride', type=int, default=8, help='stride')
        parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
        parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
        parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
        parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

        # Formers 
        parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        parser.add_argument('--enc_in', type=int, default=1, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size') # !!!
        parser.add_argument('--c_out', type=int, default=1, help='output size') # !!!
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=2, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='mse', help='loss function')
        parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
        parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=device_n, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

        #args = parser.parse_args()
        ##############-------------------------------------#############
        args_list = []
        for key, value in update_dict.items():
            args_list.append(key)
            args_list.append(str(value))  # Convert the value to a string

        # Step 3: Parse the arguments using the default ones and override with new ones from the dictionary
        args = parser.parse_args(args_list)
        
        model = PatchTST.Model(args).float()
        model.load_state_dict(torch.load("/home/hbs/TS/my_p/shapeX/scr/PatchTST/PatchTST_supervised/checkpoints/freqshape_264_PatchTST_freqshape_264_ftS_sl1400_ll48_pl1400_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"))
        
        return model
    

    
    
    
    
    
    else:
        print("error!! no attention model choosed")
    

def get_model(model_name):
    
    if model_name == "ProtopTST":
  
        gpu_n=3
        device=f"cuda:{gpu_n}"
        dataset_name="mitecg"  # freqshape  seqcomb_single, mitecg, 'seqcomb_single_better'
        SEQ_LEN=360
        CHANNEL=1
        DATASET_PATH="/home/hbs/TS/XTS/TimeX/datasets/MITECG" # 
        CLASS_NUMBER=2
        SPLIT=1
        proto_len=30
        PROTO_NUM=4

        BATCH_SIZE=128
        
    

        
        config = {
        "task_name": "classification",  # Task name: classification
        "is_training": 1,  # Training status
        "model_id": dataset_name,  # Model ID
        "model": "ProtoPTST",  # Model: Transformer  PatchTST ProtoPTST
        
        # Data settings
        "data": dataset_name,  # Dataset: APAVA
        "root_path": "/home/hbs/TS/XTS/TimeX/datasets/SeqCombSingle",  # Root path of the dataset
        
        # Forecasting task settings (not used in this specific task)
        "seq_len": 96,  
        "label_len": 48,  
        "pred_len": 96,  
        "features": "S",
        "seasonal_patterns": "Monthly",
        "inverse": False,
        
        # Imputation and anomaly detection task settings (not used)
        "mask_rate": 0.25,  
        "anomaly_ratio": 0.25,
        
        # Model settings for classification
        "e_layers": 6,  # Number of encoder layers
        "d_model": 128,  # Model dimension
        "d_ff": 256,  # Feedforward network dimension
        "enc_in": 7,  # You might want to adjust these values
        "dec_in": 7,  
        "c_out": 7,  
        "n_heads": 8,  
        "top_k": 5,  
        "num_kernels": 6,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": True,
        "no_inter_attn": False,
        "chunk_size": 16,
        "patch_len": 4, # 16
        "stride": 1,    # 8
        "sampling_rate": 256,
        "patch_len_list": [2, 4, 8],
        "single_channel": False,
        "augmentations": "flip,shuffle,jitter,mask,drop",
        
        # Optimization settings
        "batch_size": BATCH_SIZE,  # Batch size
        "train_epochs": 100,  # Number of training epochs
        "learning_rate": 0.0001,  # Learning rate
        "itr": 5,  # Number of iterations
        "patience": 10,  # Early stopping patience
        "des": "Exp",  # Experiment description
        "loss": "MSE",  
        "lradj": "type1",
        "use_amp": False,
        "swa": False,
        "d_layers":1,
        
        # GPU settings
        "use_gpu": True,  # Use GPU
        "gpu": gpu_n,  # Specify the GPU ID
        "use_multi_gpu": False,
        "devices": f"{gpu_n}",
        "device": f"cuda:{gpu_n}",
        
        # De-stationary projector parameters
        "p_hidden_dims": [128, 128],  
        "p_hidden_layers": 2,
        'freq':"h",
        'num_workers':1,
        "seq_len":SEQ_LEN,
        
        # building model args:
        "pred_len": 0,
        "enc_in": CHANNEL,
        "num_class": CLASS_NUMBER,
        
        
        # protoptst args
        "num_prototypes": PROTO_NUM,
        #"prototype_shape": [1,197,1],
        "prototype_len": proto_len

    
        }
        # Example: Accessing an updated parameter

        args = argparse.Namespace(**config)
        print(args.data)
        print(config['model_id'])  # Output: APAVA-Indep


        model = ProtoPTST.Model(args).float().to(device)
        model.load_state_dict(torch.load("/home/hbs/TS/my_p/shapeX/scr/Medformer_1/checkpoints/classification/mitecg/ProtoPTST/classification_mitecg_ProtoPTST_mitecg_ftS_sl360_ll48_pl0_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrue_Exp_seed41/checkpoint.pth"))
        print("!!!!!!!!!!!!!!!!Model loaded")
    
        return model
    
    elif model_name  in ["freqshape_311", "freqshape_315"]:

        
        split_no=int(model_name[-3:])
        
        dataset_name="freqshape"  # freqshape  seqcomb_single, mitecg, 'seqcomb_single_better'
        SEQ_LEN=800 # 500
        CHANNEL=1
        gpu_n=1
        device=f"cuda:{gpu_n}"
        DATASET_PATH="/home/hbs/TS/XTS/TimeX/datasets/FreqShapeUD" # 
        CLASS_NUMBER=2
        SPLIT=split_no  # 101
        proto_len=103
        PROTO_NUM=2

        BATCH_SIZE=128
        
        config = {
        "task_name": "classification",  # Task name: classification
        "is_training": 1,  # Training status
        "model_id": dataset_name,  # Model ID
        "model": "ProtoPTST",  # Model: Transformer  PatchTST ProtoPTST
        
        # Data settings
        "data": dataset_name,  # Dataset: APAVA
        "root_path": "/home/hbs/TS/XTS/TimeX/datasets/SeqCombSingle",  # Root path of the dataset
        
        # Forecasting task settings (not used in this specific task)
        "seq_len": 96,  
        "label_len": 48,  
        "pred_len": 96,  
        "features": "S",
        "seasonal_patterns": "Monthly",
        "inverse": False,
        
        # Imputation and anomaly detection task settings (not used)
        "mask_rate": 0.25,  
        "anomaly_ratio": 0.25,
        
        # Model settings for classification
        "e_layers": 6,  # Number of encoder layers
        "d_model": 128,  # Model dimension
        "d_ff": 256,  # Feedforward network dimension
        "enc_in": 7,  # You might want to adjust these values
        "dec_in": 7,  
        "c_out": 7,  
        "n_heads": 8,  
        "top_k": 5,  
        "num_kernels": 6,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": True,
        "no_inter_attn": False,
        "chunk_size": 16,
        "patch_len": 4, # 16
        "stride": 1,    # 8
        "sampling_rate": 256,
        "patch_len_list": [2, 4, 8],
        "single_channel": False,
        "augmentations": "flip,shuffle,jitter,mask,drop",
        
        # Optimization settings
        "batch_size": BATCH_SIZE,  # Batch size
        "train_epochs": 100,  # Number of training epochs
        "learning_rate": 0.0001,  # Learning rate
        "itr": 5,  # Number of iterations
        "patience": 10,  # Early stopping patience
        "des": "Exp",  # Experiment description
        "loss": "MSE",  
        "lradj": "type1",
        "use_amp": False,
        "swa": False,
        "d_layers":1,
        
        # GPU settings
        "use_gpu": True,  # Use GPU
        "gpu": gpu_n,  # Specify the GPU ID
        "use_multi_gpu": False,
        "devices": f"{gpu_n}",
        "device": f"cuda:{gpu_n}",
        
        # De-stationary projector parameters
        "p_hidden_dims": [128, 128],  
        "p_hidden_layers": 2,
        'freq':"h",
        'num_workers':1,
        "seq_len":SEQ_LEN,
        
        # building model args:
        "pred_len": 0,
        "enc_in": CHANNEL,
        "num_class": CLASS_NUMBER,
        
        
        # protoptst args
        "num_prototypes": PROTO_NUM,
        #"prototype_shape": [1,197,1],
        "prototype_len": proto_len,
        "split": SPLIT

    
    
        }
        # Example: Accessing an updated parameter

        args = argparse.Namespace(**config)
        #print(args.data)
        #print(config['model_id'])  # Output: APAVA-Indep

            # pass args to model
        model = ProtoPTST.Model(args).float().to(device)
        model.load_state_dict(torch.load(f"/home/hbs/TS/my_p/shapeX/scr/Medformer_1/checkpoints/classification/freqshape/ProtoPTST/classification_freqshape_ProtoPTST_freqshape_{SPLIT}_ftS_sl800_ll48_pl0_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrue_Exp_seed41/checkpoint.pth"))

        return model
    
    elif model_name in ["PhalangesOutlinesCorrect","Yoga","FaceAll","UWaveGestureLibraryAll","MixedShapesRegularTrain","StarLightCurves","TwoPatterns"]:
        
        
        gpu_n=3
        device=f"cuda:{gpu_n}"
        dataset_name= model_name# "MixedShapesRegularTrain"# FaceAll, UWaveGestureLibraryAll

        dataset_dict = {"FaceAll": ["FaceAll",14,131,21], # name, class number, seq_len, proto_len
                        "UWaveGestureLibraryAll": ["UWaveGestureLibraryAll",8,945,81],
                        "MixedShapesRegularTrain": ["MixedShapesRegularTrain",5,1024,361],
                                        "StarLightCurves":["StarLightCurves",3,1024,51],
                
                        "TwoPatterns":["TwoPatterns",4,128,21],
                                               "Yoga":["Yoga",2,426,21],
                        "PhalangesOutlinesCorrect":["PhalangesOutlinesCorrect",2,80,11]
                        
                       
                        
                        
                        }

        CLASS_NUMBER=dataset_dict[dataset_name][1]    
        
        SEQ_LEN=dataset_dict[dataset_name][2]
        CHANNEL=1
        DATASET_PATH="" # 
        CLASS_NUMBER=dataset_dict[dataset_name][1]
        SPLIT=1  # 101
        proto_len= dataset_dict[dataset_name][3]
        PROTO_NUM=3

        BATCH_SIZE=128


        config = {
        "task_name": "classification",  # Task name: classification
        "is_training": 1,  # Training status
        "model_id": dataset_name,  # Model ID
        "model": "ProtoPTST",  # Model: Transformer  PatchTST ProtoPTST
        
        # Data settings
        "data": dataset_name,  # Dataset: APAVA
        "root_path": "/home/hbs/TS/XTS/TimeX/datasets/SeqCombSingle",  # Root path of the dataset
        
        # Forecasting task settings (not used in this specific task)
        "seq_len": 96,  
        "label_len": 48,  
        "pred_len": 96,  
        "features": "S",
        "seasonal_patterns": "Monthly",
        "inverse": False,
        
        # Imputation and anomaly detection task settings (not used)
        "mask_rate": 0.25,  
        "anomaly_ratio": 0.25,
        
        # Model settings for classification
        "e_layers": 6,  # Number of encoder layers
        "d_model": 128,  # Model dimension
        "d_ff": 256,  # Feedforward network dimension
        "enc_in": 7,  # You might want to adjust these values
        "dec_in": 7,  
        "c_out": 7,  
        "n_heads": 8,  
        "top_k": 5,  
        "num_kernels": 6,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": True,
        "no_inter_attn": False,
        "chunk_size": 16,
        "patch_len": 4, # 16
        "stride": 1,    # 8
        "sampling_rate": 256,
        "patch_len_list": [2, 4, 8],
        "single_channel": False,
        "augmentations": "flip,shuffle,jitter,mask,drop",
        
        # Optimization settings
        "batch_size": BATCH_SIZE,  # Batch size
        "train_epochs": 100,  # Number of training epochs
        "learning_rate": 0.0001,  # Learning rate
        "itr": 5,  # Number of iterations
        "patience": 10,  # Early stopping patience
        "des": "Exp",  # Experiment description
        "loss": "MSE",  
        "lradj": "type1",
        "use_amp": False,
        "swa": False,
        "d_layers":1,
        
        # GPU settings
        "use_gpu": True,  # Use GPU
        "gpu": gpu_n,  # Specify the GPU ID
        "use_multi_gpu": False,
        "devices": f"{gpu_n}",
        "device": f"cuda:{gpu_n}",
        
        # De-stationary projector parameters
        "p_hidden_dims": [128, 128],  
        "p_hidden_layers": 2,
        'freq':"h",
        'num_workers':1,
        "seq_len":SEQ_LEN,
        
        # building model args:
        "pred_len": 0,
        "enc_in": CHANNEL,
        "num_class": CLASS_NUMBER,
        
        
        # protoptst args
        "num_prototypes": PROTO_NUM,
        #"prototype_shape": [1,197,1],
        "prototype_len": proto_len,
        "split": SPLIT

        
        
        }
        # Example: Accessing an updated parameter

        args = argparse.Namespace(**config)
     
        



        model = ProtoPTST.Model(args).float().to(device) 
        
        #print(f"{dataset_dict[dataset_name][0]}model loaded")
        
        model_path=f"/home/hbs/TS/my_p/shapeX/scr/Medformer_1/checkpoints/classification/{dataset_dict[dataset_name][0]}/ProtoPTST/classification_{dataset_dict[dataset_name][0]}_ProtoPTST_{dataset_dict[dataset_name][0]}_ftS_sl{dataset_dict[dataset_name][2]}_ll48_pl0_dm128_nh8_el6_dl1_df256_fc1_ebtimeF_dtTrue_Exp_seed41/checkpoint.pth"


        model.load_state_dict(torch.load(model_path))
        
        return model
    else:
        print("error!! no attention model choosed")

