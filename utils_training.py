import torch
from shapelet_encoder.models import ProtoPTST
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os

from matplotlib import pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

def find_project_root(current_dir, marker=".git"):
    # 循环查找父目录，直到找到标记文件或目录
    while current_dir != os.path.dirname(current_dir):
        if marker in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Metric score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "Autoformer": Autoformer,
            "Crossformer": Crossformer,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "iTransformer": iTransformer,
            "MTST": MTST,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "PatchTST": PatchTST,
            "Reformer": Reformer,
            "Transformer": Transformer,
            "Medformer": Medformer,
            "ProtoPTST":ProtoPTST
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

def plot_visualize_some(X,generated_exps,gt_exps,times,y,choice,saving_path,place_holder_1=None,if_norm=False,mo_s=None):
    

    # Pick out with each label:
    #uni = torch.unique(y).numpy()

    #print('uni_number:',uni)
    

    for i in [0]:
        
        plt.clf()
        #choice = np.random.choice((y == i).nonzero(as_tuple=True)[0].numpy())
        
        print("YYYYYY:",y[choice])


        Xc, gt_exps_c,generated_exps_c = X[:,choice,:].numpy(), gt_exps[:,choice,:].numpy(),generated_exps[:,choice,:].numpy()

        #print('Xc, gtc :', Xc.shape, gtc.shape )
        
        fig, ax1 = plt.subplots()

        ax1.plot(times[:,choice], Xc[:,0])
        ax1.set_ylabel('original series', color='black')
        
        if place_holder_1 is not None:
            ax2 = ax1.twinx()
            ax2.plot(times[:,choice], place_holder_1, 'b-', label='attention',color='g')  # x轴默认使用索引
            #ax2.plot(times[:,choice], place_holder_1[choice], 'b-', label='attention',color='g')  # x轴默认使用索引
            ax2.set_ylabel('pertrubed series', color='g')
            ax2.tick_params('y', colors='g')
        

      
        #generated exps
        highlight_y=generated_exps_c[:,0]
        plot_x=times[:,choice]
        for i in range(len(highlight_y)):
            
            if if_norm == True:
                ax1.fill_between([plot_x[i] - 0.45, plot_x[i] + 0.45], -10, 10, color='blue', alpha=abs(highlight_y[i]-np.mean(highlight_y)))
            else:
            #plt.fill_between([plot_x[i] - 0.45, plot_x[i] + 0.45], -10, 10, color='blue', alpha=abs(highlight_y[i]-np.mean(highlight_y))*10)  #for TimeX S2C
                ax1.fill_between([plot_x[i] - 0.45, plot_x[i] + 0.45], -10, 10, color='blue', alpha=abs(highlight_y[i]))
                pass
        #real exps:
        highlight_r_y=gt_exps_c[:,0]
        for j in range(len(highlight_y)):
             
            ax1.fill_between([plot_x[j] - 0.45, plot_x[j] + 0.45], -3, 3, color='red', alpha=highlight_r_y[j])   
        
        
        #plt.plot(times[:,choice], mo_s[0])    
        if mo_s is not None:
            break
            for s in mo_s:
                plt.plot(times[:,choice], s)
                
        plt.savefig(saving_path + '/'+str(choice)+'.png')
          
          
def normalize_exp(exps):
    norm_exps = torch.empty_like(exps)
    for i in range(exps.shape[1]):
        norm_exps[:,i,:] = (exps[:,i,:] - exps[:,i,:].min()) / (exps[:,i,:].max() - exps[:,i,:].min() + 1e-9)
    return norm_exps

def normalize_one_exp(exps):
    norm_exps = (exps - exps.min()) / (exps.max() - exps.min() + 1e-9)
    return norm_exps         
          


def ground_truth_xai_eval(generated_exps, gt_exps, penalize_negatives=True, times=None):
    """
    Compute AUPRC of generated explanation against ground-truth explanation.
    - AUPRC is computed for each sample and averaged across all samples.

    Args:
        generated_exps (torch.Tensor): Explanations generated by method under evaluation.
        gt_exps (torch.Tensor): Ground-truth explanations on which to evaluate.
        penalize_negatives (bool): Whether to penalize negative examples.
        times (torch.Tensor, optional): Mask for selecting valid time steps.

    Returns:
        dict: Contains lists of AUPRC, AUP, and AUR for all samples.
    """
    # Normalize generated explanations
    generated_exps = normalize_exp(generated_exps).detach().clone().cpu().numpy()
    gt_exps = gt_exps.detach().clone().cpu().numpy()
    gt_exps = gt_exps.astype(int)

    all_auprc, all_aup, all_aur = [], [], []
    for i in range(generated_exps.shape[1]):
        if times is not None:
            gte = (gt_exps[:, i, :][times[:, i] > -1e5]).flatten()
            gene = normalize_one_exp(generated_exps[:, i, :][times[:, i] > -1e5]).flatten()
        else:
            gte = gt_exps[:, i, :].flatten()
            gene = generated_exps[:, i, :].flatten()

        # Compute Average Precision Score
        auprc = average_precision_score(gte, gene)
        prec, rec, thres = precision_recall_curve(gte, gene)

        # Handle AUC calculation
        if len(thres) >= 2 and len(rec) >= 2 and len(prec) >= 2:
            aur = auc(thres, rec[:-1])  # Recall curve
            aup = auc(thres, prec[:-1])  # Precision curve
        else:
            aur, aup = 0.0, 0.0  # Default values for insufficient data

        all_auprc.append(auprc)
        all_aup.append(aup)
        all_aur.append(aur)

    output_dict = {
        'auprc': all_auprc,
        'aup': all_aup,
        'aur': all_aur
    }

    return output_dict  