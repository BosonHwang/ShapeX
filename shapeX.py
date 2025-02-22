
#from get_attention_model import get_model

from get_pretrain_model import get_pretrain_model,get_model
import torch
import numpy as np
#from maskCut_TS import *

from joblib import Parallel, delayed
from tqdm import tqdm

DISTANCE = 10   
def step_waveform(series,ratio=1):
    # 平滑处理, series : T
    if not torch.is_tensor(series):
        series = torch.tensor(series)
    
    threshold=torch.mean(series)*ratio
    
    smoothed = np.convolve(series, np.ones(5)/5, mode='same')
    
    # 阈值化
    stepped = np.where(smoothed > threshold.numpy(), 1, 0)
    
    return stepped


def step_waveform_his(series,ratio=1):
    # 平滑处理, series : T
    
    threshold=torch.mean(series)*ratio
    
    smoothed = np.convolve(series, np.ones(5)/5, mode='same')
    
    # 阈值化
    stepped = np.where(smoothed > threshold.numpy(), 1, 0)
    
    return stepped


def moving_average_centered(time_series, window_size):
    """
    中心滑动平均：窗口中心对齐
    """
    padded_series = np.pad(time_series, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(padded_series, np.ones(window_size), 'valid') / window_size


def moving_average_right(time_series, window_size):
    """
    向右滑动平均：窗口向右对齐
    """
    padded_series = np.pad(time_series, (0, window_size - 1), mode='edge')
    return np.convolve(padded_series, np.ones(window_size), 'valid') / window_size


def fill_short_negative_sequences(time_series, threshold=7):
    """
    将长度小于 threshold 的负值子序列填充为 1。
    
    参数:
    - time_series: numpy 数组，表示时间序列
    - threshold: 小于该长度的负值子序列将被填充为 1，默认为 5
    
    返回:
    - 填充后的时间序列
    """
    # 创建一个副本来保存填充后的结果
    
    filled_series = time_series-1
    
    # 用于记录负值子序列的开始和结束位置
    start = None
    
    for i, value in enumerate(filled_series):
        if value < 0:
            # 如果检测到负值，记录开始位置
            if start is None:
                start = i
        else:
            # 如果检测到非负值，且之前记录了负值子序列的开始位置
            if start is not None:
                end = i  # 当前索引就是子序列的结束位置
                length = end - start  # 计算负值子序列的长度
                
                # 如果负值子序列长度小于阈值，则将其填充为 1
                if length < threshold:
                    time_series[start:end] = 2
                
                # 重置开始位置
                start = None
    
    
    return time_series


def interpolate_tensor(input_tensor,new_T=480,dim_to_change=0):

    # 需要先交换维度，将 T 维度放在最后，方便使用 interpolate
    last_dim=len(input_tensor.shape)-1
    input_tensor = input_tensor.transpose(dim_to_change,last_dim)  # 现在形状变为 (batch_size, D, T)

    # 使用 interpolate 函数
    output_tensor = torch.nn.functional.interpolate(input_tensor, size=new_T, mode='linear', align_corners=True)
    print(output_tensor.shape)

    # 再次交换维度，将 D 维度放回到最后
    output_tensor = output_tensor.transpose(dim_to_change,last_dim)  # 现在形状变为 (batch_size, new_T, D)

    return output_tensor


def segment_sequence(a):
    segments = []
    start = None
    current_value = None

    for i in range(len(a)):
        if a[i] != current_value:  # 当值发生变化时
            if start is not None:
                segments.append((start, i - 1))  # 记录结束索引
            start = i  # 记录新的起始索引
            current_value = a[i]  # 更新当前值

    # 处理序列结尾仍然是相同值的情况
    if start is not None:
        segments.append((start, len(a) - 1))

    return segments


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


def segment_sequence_ones_with_max_indicator(a, T):
    """
    Segments a sequence based on consecutive '1' values, only retaining the segment
    that includes the maximum value from an indicator sequence T.
    
    Parameters:
    - a (list): The input list or sequence to segment.
    - T (list): The indicator sequence, where we find the maximum value.

    Returns:
    - selected_segment (tuple): A tuple containing the start and end indices 
      of the segment where the value is '1' and includes the maximum value from T.
      Returns None if no such segment is found.
    """
    segments = []
    start = None

    # Find the index of the maximum value in T
    max_index = torch.argmax(T).item()

    # Segment based on consecutive '1's in 'a'
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

    # Find the segment that includes the max_index
    for segment in segments:
        if segment[0] <= max_index <= segment[1]:
            return [segment]

    return None  # Return None if no segment includes max_index



def get_seg_pretrain(signal,class_model = None,#get_pretrain_model("freqshape_274"),
                      class_model_seq_len=1400,class_model_channel=1,
                      
                      moving_average_window=8,
                      return_attention=True, step_attention=True):
    # signal.shape: tensor, [T], 
    
    sig_dim=signal.shape[0]
    device=next(class_model.parameters()).device
    class_model.to(device)
    
    ### one for seqcomb dataset
    signal=signal+2
    print("@@@@@@@@@@@@@@@@@@@@@")
    
    batch_x=signal.reshape(1,-1,1)
    #batch_x=torch.abs(batch_x)
    #batch_x=torch.tensor(np.square(np.square(torch.abs(batch_x)))) #!!!
    
    # change to APAVA size:
    
    batch_x= batch_x.expand(-1, sig_dim, class_model_channel)
    #batch_x=batch_x[:,:256,:]
    batch_x=interpolate_tensor(batch_x,class_model_seq_len,1) # 
    
    
    
    # get pretrain model outout:
    atten_slice=class_model(batch_x)
    atten_slice=atten_slice.reshape(-1).tolist()
  
    
    ## add layer and head 
    
  
    atten_slice=(atten_slice/np.max(atten_slice)) #### !!!
    
    if step_attention:
        print("step_attentionstep_attentionstep_attentionstep_attention")
        atten_slice_steped=torch.tensor(step_waveform(atten_slice,1),dtype=torch.float16)
        
    
    # change length
    feats=torch.tensor(atten_slice).reshape(-1,1,1)
    feats=interpolate_tensor(feats,sig_dim,0).reshape(-1,1)
    
    
    # get bipartitions by maskCut
    
    # to be implemented:
    maskcut_forward_TS=None
    bipartitions, A= maskcut_forward_TS(feats,feats.shape[0],1,sig_dim,0,1,device)
    bipartition_0=bipartitions[0].astype(int)
    
    #bipartition_list.append(bipartition_0)
    #atten_slice_list.append(atten_slice)
    
    # return some inter vector
    #seg=segment_sequence(bipartition_0)
    seg=segment_sequence(atten_slice_steped.tolist())
    return_atten_slice=interpolate_tensor(torch.tensor(atten_slice).reshape(-1,1,1),sig_dim,0).reshape(-1)
    return_bipartition_0=torch.tensor(bipartition_0,dtype=torch.float16)
    
    print("return_atten_slice!!!!!!:",return_atten_slice.shape)
    
    if return_attention:
        return seg, atten_slice_steped
    else:
        return seg






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


def get_seg_ProtopTST_SNC(signal,seg_model): # for SNC dataset
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






class ShapeX:
    def __init__(self) -> None:
        pass
    
    def get_attention_model(self):
        self.attention_model=Atention_model
        
def get_equal_segments(seq_len, seg_length):
    segments = []
    start = 0
    while start < seq_len:
        end = min(start + seg_length, seq_len)  # 确保不超过序列长度
        segments.append([start, end - 1])  # 索引从 0 开始，因此 end - 1
        start = end  # 下一个段的起始索引
    return segments       

    


class ScoreSubsequences(object):
    def __init__(self,args,class_model,seg_model=None,seq_method="prototype") -> None:
        self.class_model=class_model
        
        self.seq_method=seq_method
        self.seg_model=seg_model
        self.device=args.device
        self.args=args
        
        self.equal_seg_len = args.equal_seg_len
        

    def get_subsequence_peaks_and_valleys(self,y, smoothing_window=5, distance=DISTANCE): ####################### 

    #参数:
   
    #- smoothing_window: 平滑窗口大小
    #- distance: 用于找到波峰和波谷的最小距离

    # Step 1: 平滑处理
       
        print("Yyyyyyyyyyyy:", y.shape)
        def smooth(y, window_len=11):
            window = np.ones(window_len) / window_len
            return np.convolve(y, window, mode='same')
    
        if AD:
            y = get_TranAD_y(y)[:,1]
            
        y=y.reshape(-1)
        y_smooth=y
        #y_smooth = smooth(y, window_len=smoothing_window)
    
    # Step 2: 寻找波峰和波谷
        #peaks, _ = find_peaks(y_smooth, distance=distance)
        valleys, _ = find_peaks(-y_smooth, distance=distance)
    
    # Step 3: 根据波峰和波谷划分子序列
        split_points = np.sort( valleys)
    
        sub_sequences = []
        start_idx = 0
        for idx in split_points:
            sub_sequences.append((start_idx, idx))
            start_idx = idx + 1

        if start_idx < len(y):
            sub_sequences.append((start_idx, len(y) - 1))
    
        return sub_sequences
    
    def normalize_scores(self,score_list):
        """
        对分数列表进行归一化。
        如果最大值为 0，则返回全零列表。

        Args:
            score_list (list of float): 原始分数列表。

        Returns:
            list of float: 归一化后的分数列表。
        """
        if not score_list:  # 如果列表为空，返回空列表
            return []
        
        max_score = max(score_list)  # 获取最大值
        if max_score == 0:  # 如果最大值为 0，返回全零列表
            return [0 for _ in score_list]
        
        # 正常归一化
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
            
        elif self.seq_method == "prototype":  ## for ECG data
            #print("###################protype")
            
            if self.args.data=="mitecg":
                seq_lst,actions_sum = get_seg_ProtopTST(signal,self.seg_model)
                
            else:
            
                seq_lst,actions_sum = get_seg_ProtopTST_SNC(signal,self.seg_model)
        
        else:
            print("!!!! seq_method error !!!")
        
        score_list = []
        score_vector = torch.zeros((signal.shape))
        
        original_prediction = self.get_classification(signal)
        
        modified_series_list=[]
        
        seq_lst=np.clip(seq_lst, None, self.seq_len-1)
        
        for i,(start_idx, end_idx)  in enumerate(seq_lst):
            # 创建时间序列的副本，并将指定子序列置为零
            modified_series = signal.clone()
            
            
            
            #print("start_idx, end_idx:",start_idx, end_idx)
            
            if self.args.ablation=="no_linaer_zero":
                modified_series[start_idx:end_idx,:] = 0
            if self.args.ablation=="no_linear_random":
                modified_series[start_idx:end_idx,:] = torch.rand((end_idx-start_idx,1,1))
            if self.args.ablation=="no_linear_mean":
                modified_series[start_idx:end_idx,:] = torch.mean(signal)
            
            else:
                modified_series[start_idx:end_idx,:] =torch.from_numpy(np.linspace(modified_series[start_idx], modified_series[end_idx], end_idx - start_idx)) 
            
            #sub_seq=#torch.mean(signal)  #torch.mean(signal)，0              ######### disturn data
            
            
            # 对修改后的时间序列进行预测
            modified_prediction = self.get_classification(modified_series)
            
            # 计算得分：原始预测与修改后预测的差值
            score = (original_prediction - modified_prediction).abs().sum().item() ############ pay attention here !!!
            
            #score = score / (end_idx - start_idx+ 1) # !!!
            #score = score / (end_idx - start_idx+ 1) ** 3 # !!!
    
            # 将得分加入列表
            score_list.append(score)
            
            modified_series_list.append(modified_series)
    
        #max_score = max(score_list) if score_list else 1  # 防止出现全零的情况
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
        
 
    
    def get_score_as_GTEXP(self,X,place_hodler=False,only_max=False):
        ## X:[T,sample,dim]
        self.seq_len=X.shape[0]
        print("X.shape:",X.shape)
        
        GT_EXP=torch.tensor([]).to(dtype=torch.float16)
        
        place_holder_list=[]
        
        sample_n=X.shape[1]
        
        with torch.no_grad(): # 减小计算资源占用
            for i in tqdm(range(sample_n), desc="counting saliency score",position=0):
                #print(i)
                signal=X[:,i,:]
                if only_max:
                    score_vector,place_holder=self.get_score_vector_max(signal,i)
                    
                if self.args.seq_method == "equal_seg":
                    score_vector=self.get_score_vector(signal,i)
                    place_holder=[]
                    
                else:
                    score_vector,place_holder =self.get_score_vector(signal,i)#
                #score_vector=self.get_score_vector(signal,i)
                GT_EXP=torch.cat((GT_EXP, score_vector), dim=1).to(dtype=torch.float16)
                
                
                place_holder_list.append(place_holder)
                
                torch.cuda.empty_cache() ## ！！！
                
                del signal, score_vector, place_holder
                torch.cuda.empty_cache() 
                
        
        print("GT_EXP.shape:",GT_EXP.shape)
        return GT_EXP #,place_holder_list
    
    
    def get_score_as_GTEXP_para(self, X, place_hodler=False, only_max=False):
        """
        使用并行化计算 GT_EXP 显著性得分
        """
        self.seq_len = X.shape[0]
        sample_n = X.shape[1]

        # 定义处理函数
        def process_sample(i):
            signal = X[:, i, :]
            if only_max:
                return self.get_score_vector_max(signal, i)
            else:
                return self.get_score_vector(signal, i)

        # 并行化处理
        results = Parallel(n_jobs=-1)(delayed(process_sample)(i) for i in tqdm(range(sample_n), desc="counting saliency score"))

        # 解压结果
        score_vectors, place_holder_list = zip(*results)

        # 拼接结果
        GT_EXP = torch.cat(score_vectors, dim=1)

        return GT_EXP
    
