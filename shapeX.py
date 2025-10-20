
#from get_attention_model import get_model

#from his.get_pretrain_model import get_pretrain_model,get_model
import torch
import numpy as np
#from maskCut_TS import *

from joblib import Parallel, delayed
from tqdm import tqdm


DISTANCE = 10   


def moving_average_centered(time_series, window_size):
    """
    Centered moving average: window centered alignment.
    """
    padded_series = np.pad(time_series, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(padded_series, np.ones(window_size), 'valid') / window_size


def fill_short_negative_sequences(time_series, threshold=7):
    """
    Fill negative subsequences shorter than `threshold` with value 1.
    
    Args:
        time_series (np.ndarray): Input time series.
        threshold (int): Negative subsequences shorter than this will be filled to 1.
    
    Returns:
        np.ndarray: The modified time series.
    """
    # Create an offset view to help detect negatives
    filled_series = time_series-1
    
    # Track start of negative subsequence
    start = None
    
    for i, value in enumerate(filled_series):
        if value < 0:
            # Found a negative value: record start if not set
            if start is None:
                start = i
        else:
            # Transition to non-negative and we had a running negative subsequence
            if start is not None:
                end = i  # end position (exclusive)
                length = end - start  # length of the negative subsequence
                
                # If shorter than threshold, fill to 1 (set to 2 before later offset)
                if length < threshold:
                    time_series[start:end] = 2
                
                # Reset start tracker
                start = None
    
    
    return time_series


def interpolate_tensor(input_tensor,new_T=480,dim_to_change=0):

    # Swap dims to move T to the last position for interpolate
    last_dim=len(input_tensor.shape)-1
    input_tensor = input_tensor.transpose(dim_to_change,last_dim)  # Now (batch_size, D, T)

    # Use interpolate
    output_tensor = torch.nn.functional.interpolate(input_tensor, size=new_T, mode='linear', align_corners=True)
    print(output_tensor.shape)

    # Swap dims back
    output_tensor = output_tensor.transpose(dim_to_change,last_dim)  # Now (batch_size, new_T, D)

    return output_tensor


def segment_sequence_ones(a):
    """
    Segments a sequence based on consecutive '1' values, recording
    the start and end indices of each '1' segment.
    
    Parameters:
    - a (list): The input list or sequence to segment.

    Returns:
    - segments (list of tuples): Each tuple contains the start and end indices 
      of a segment where the value is '1'.
    """
    segments = []
    start = None

    for i in range(len(a)):
        if a[i] == 1:
            if start is None:
                start = i  # Start a new segment
        else:
            if start is not None:
                segments.append((start, i - 1))  # End the current segment
                start = None  # Reset start

    # Handle the case where the sequence ends with a segment of '1's
    if start is not None:
        segments.append((start, len(a) - 1))
    
    torch.cuda.empty_cache()

    return segments




def get_seg_ProtopTST(signal ,seg_model): # for ECG dataset
    """
    signal: [seq_len,dim]
    
    """
    
    
    
    threshold = 1
    
    if seg_model is not None:
        ProtopTST=seg_model
    else: 
        print("no seg model chooesed")
    
    #ProtopTST=get_model(model_name) #  "ProtopTST"
    device = ProtopTST.device
    
    
    x_in=signal.reshape(1,-1,1).to(device)
    #print("x_in.device:",x_in.device,ProtopTST.device)
    
    out,actions,prototype = ProtopTST(x_in,x_in,x_in,x_in)
    
    actions_sum = torch.sum(actions,dim=-1).reshape(-1)
    
    actions_sum = fill_short_negative_sequences(actions_sum)
    
    actions_sum_step = (actions_sum  > threshold).int()
    
    segs = segment_sequence_ones(actions_sum_step) ##!!!!
    #segs=segment_sequence_ones_with_max_indicator(actions_sum_step,actions_sum)
    
    torch.cuda.empty_cache()
    #print("segs!!!:",segs)
    
    return segs , actions_sum



def get_seg_ProtopTST_SNC(signal,seg_model): # for mcce,mcch,mtce,mtch dataset
    """
    signal: [seq_len,dim]
    
    """
    threshold = 0.4
    #print("###########################################")
    
    if seg_model is not None:
        ProtopTST=seg_model
    else: 
        print("no seg model chooesed")
        
    
   
    device = ProtopTST.device
    
    x_in=signal.reshape(1,-1,1).to(device)
    #print("x_in.device:",x_in.device,ProtopTST.device)
    
    out,actions,prototype = ProtopTST(x_in,x_in,x_in,x_in)
    
    #actions_sum = torch.sum(actions,dim=-1).reshape(-1)
    actions_sum = actions[:,:,0].reshape(-1).cpu().detach().numpy()
    
    
    actions_sum = moving_average_centered(actions_sum,100)
    
    actions_sum_step = (actions_sum  > threshold)#.int()
    
    segs = segment_sequence_ones(actions_sum_step) ##!!!!
    #segs=segment_sequence_ones_with_max_indicator(actions_sum_step,actions_sum)
    
    torch.cuda.empty_cache()
    #print("segs!!!:",segs)
    
    return segs , actions_sum



def get_seg_unified(signal, seg_model, dataset_name):
    """
    Unified segmentation wrapper that dispatches by dataset name and
    standardizes the return types.

    Returns:
        segs: list of (start_idx, end_idx)
        actions_sum: torch.Tensor of shape [T]
    """
    name = (dataset_name or "").lower() if isinstance(dataset_name, str) else ""
    if (name == "mitecg") or ("ecg" in name):
        segs, actions_sum = get_seg_ProtopTST(signal, seg_model)
    else:
        segs, actions_sum = get_seg_ProtopTST_SNC(signal, seg_model)

    # Normalize actions_sum to torch tensor (device follows input signal)
    if isinstance(actions_sum, np.ndarray):
        actions_sum = torch.tensor(
            actions_sum,
            dtype=signal.dtype,
            device=signal.device if hasattr(signal, 'device') else None,
        )
    return segs, actions_sum



def get_equal_segments(seq_len, seg_length):
    segments = []
    start = 0
    while start < seq_len:
        end = min(start + seg_length, seq_len)  # ensure not exceeding sequence length
        segments.append([start, end - 1])  # indices are 0-based, so end - 1
        start = end  # advance to next segment start
    return segments       

    


class ScoreSubsequences(object):
    def __init__(self,args,class_model,seg_model=None,seq_method="prototype") -> None:
        self.class_model=class_model
        
        self.seq_method=seq_method
        self.seg_model=seg_model
        self.args=args
        
        self.equal_seg_len = args.equal_seg_len
        self.device=args.device
        

    def normalize_scores(self,score_list):
        """
        Normalize a list of scores.
        If the maximum is 0, return all zeros.

        Args:
            score_list (list of float): Raw scores.

        Returns:
            list of float: Normalized scores.
        """
        if not score_list:  # empty input
            return []
        
        max_score = max(score_list)  # max value
        if max_score == 0:  # avoid divide-by-zero
            return [0 for _ in score_list]
        
        # standard normalization
        return [score / max_score for score in score_list]
        
        
    def get_classification(self,signal):
        ## signal: [T,dim]
        signal=signal.reshape(-1,1,1)
        
        #generate times:
        times=torch.arange(1, signal.shape[0] + 1, dtype=torch.float16).reshape(-1,1)
        
        #signal=signal.expand(-1,-1,4)############ for mv model
        
        classification=self.class_model(signal.to(self.device),times.to(self.device))[0,:2]
        return classification
        
        
    def get_score_vector(self,signal,signal_idx, return_place_holder =True):
        ## torch tensor: signal: [T,dim]
        """
        score_vector: [T,dim]
        
        """    
            
        if self.seq_method == "equal_seg":
            return_place_holder = False
            
            
            seq_lst = get_equal_segments(signal.shape[0], self.args.equal_seg_len)
            
        elif self.seq_method == "prototype":
            seq_lst, actions_sum = get_seg_unified(signal, self.seg_model, getattr(self.args, 'data', None))
        
        else:
            print("!!!! seq_method error !!!")
        
        score_list = []
        score_vector = torch.zeros((signal.shape))
        
        original_prediction = self.get_classification(signal)
        
        modified_series_list=[]
        
        seq_lst=np.clip(seq_lst, None, self.seq_len-1)
        
        for i,(start_idx, end_idx)  in enumerate(seq_lst):
            # Create a copy and mask the target subsequence
            modified_series = signal.clone()

            
            #print("start_idx, end_idx:",start_idx, end_idx)
            
            if self.args.ablation=="no_linaer_zero":
                modified_series[start_idx:end_idx,:] = 0
            if self.args.ablation=="no_linear_random":
                modified_series[start_idx:end_idx,:] = torch.rand((end_idx-start_idx,1,1))
            if self.args.ablation=="no_linear_mean":
                modified_series[start_idx:end_idx,:] = torch.mean(signal)
            
            else:
                # previous numpy linspace implementation (< 2.0)
                # Use pure PyTorch linear interpolation to avoid NumPy version issues
                start_val = modified_series[start_idx]
                end_val = modified_series[end_idx]
                # Create interpolation weights
                t = torch.linspace(0, 1, end_idx - start_idx, device=start_val.device, dtype=start_val.dtype)
                # Linear interpolation: start_val * (1 - t) + end_val * t
                interpolated = start_val * (1 - t.unsqueeze(-1)) + end_val * t.unsqueeze(-1)
                modified_series[start_idx:end_idx,:] = interpolated 
            
            #sub_seq=#torch.mean(signal)  #torch.mean(signal)，0              ######### disturn data
            
            
            # Predict on the modified series
            modified_prediction = self.get_classification(modified_series)
            
            # Compute score: difference between original and modified predictions
            score = (original_prediction - modified_prediction).abs().sum().item() ############ pay attention here !!!
            
            #score = score / (end_idx - start_idx+ 1) # !!!
            #score = score / (end_idx - start_idx+ 1) ** 3 # !!!
    
            # Add to list
            score_list.append(score)
            
            modified_series_list.append(modified_series)
    
        #max_score = max(score_list) if score_list else 1  # prevent all-zeros case
        #normalized_scores = [score / max_score for score in score_list]  ### normalize 
        
        
        
        normalized_scores = self.normalize_scores(score_list)
        
        for i,(start_idx, end_idx) in enumerate(seq_lst):
            score_vector[start_idx:end_idx,:]=normalized_scores[i]
        
        
        
            
        if self.args.ablation=="no_segment":
            # normalize action_sum :
         
            max = torch.max(actions_sum)
            min = torch.min(actions_sum)
            
            
            # max-min normalization:
            actions_sum = (actions_sum - min) / (max - min)+1e-6
            
            #print("actions_sum:",actions_sum.shape)
            #print("score_vector:",score_vector.shape)
            action_len=actions_sum.shape[0]
            base = torch.zeros_like(score_vector)
            
            base[:action_len,:] = actions_sum.reshape(-1,1)
            score_vector = base
                        
        if return_place_holder :
            return score_vector , actions_sum #  attention_slice , attention_slice,modified_series_list 
        else :
            return score_vector 
        
    
    def get_score_as_GTEXP(self,X,if_place_hodler=False,only_max=False):
        ## X:[N,T,dim] batch-first
        self.seq_len = X.shape[1]
        print("X.shape:", X.shape)

        scores_list = []  # list of (T,1)
        place_holder_list = []

        sample_n = X.shape[0]

        with torch.no_grad(): # reduce compute overhead
            for i in tqdm(range(sample_n), desc="counting saliency score", position=0):
                signal = X[i, :, :]  # (T,dim)
                if only_max:
                    score_vector, place_holder = self.get_score_vector_max(signal, i)
                if self.args.seq_method == "equal_seg":
                    score_vector = self.get_score_vector(signal, i)
                    place_holder = []
                else:
                    score_vector, place_holder = self.get_score_vector(signal, i)
                    place_holder = place_holder.reshape(-1, 1)

                    # Ensure place_holder is a PyTorch tensor
                    if isinstance(place_holder, np.ndarray):
                        place_holder = torch.from_numpy(place_holder).float()

                    # normalize the place_holder using PyTorch to avoid NumPy version issues:
                    place_holder = place_holder / torch.max(place_holder)

                    T = score_vector.shape[0]
                    if T < place_holder.shape[0]:
                        place_holder = place_holder[:T]
                    else: # pad using PyTorch to avoid NumPy version issues
                        padding_size = T - place_holder.shape[0]
                        place_holder = torch.cat([place_holder, torch.zeros(padding_size, 1, device=place_holder.device)], dim=0)

                    # Ensure place_holder is on the same device as score_vector
                    place_holder = place_holder.to(score_vector.device)
                    score_vector = score_vector + 0.1 * place_holder

                scores_list.append(score_vector.to(dtype=torch.float16))
                place_holder_list.append(place_holder)

                torch.cuda.empty_cache() ## !!!

                del signal, score_vector, place_holder
                torch.cuda.empty_cache()

        # stack to (N,T,1)
        GT_EXP = torch.stack(scores_list, dim=0)
        print("GT_EXP.shape:", GT_EXP.shape)
        return GT_EXP
    
    
    def get_score_as_GTEXP_para(self, X, place_hodler=False, only_max=False):
        """
        Compute GT_EXP saliency scores in parallel.
        """
        self.seq_len = X.shape[1]
        sample_n = X.shape[0]

        # Define worker
        def process_sample(i):
            signal = X[i, :, :]
            if only_max:
                return self.get_score_vector_max(signal, i)
            else:
                return self.get_score_vector(signal, i)

        # Parallel processing
        results = Parallel(n_jobs=-1)(delayed(process_sample)(i) for i in tqdm(range(sample_n), desc="counting saliency score"))

        # Unzip results
        score_vectors, place_holder_list = zip(*results)

        # Stack results (N, T, 1)
        GT_EXP = torch.stack(score_vectors, dim=0)

        return GT_EXP
    
    