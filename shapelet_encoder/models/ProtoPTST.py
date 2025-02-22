import torch
from torch import nn
import sys
sys.path.insert(0, "/home/hbs/TS/my_p/shapeX/scr/Medformer_1/")

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
#from layers.Embed import PatchEmbedding
from  models import PatchTST
from torch.nn import functional as F
import numpy



class Model(PatchTST.Model):
    
    
    def __init__(self, configs):
        super().__init__(configs)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding = configs.stride
        self.patch_num = (self.seq_len - self.patch_len + self.padding) // self.stride+1
        self.d_model = configs.d_model
        self.device= configs.device
        self.prototype_num_selected = 1
        
        self.num_prototypes = configs.num_prototypes
        self.num_classes = configs.num_class
        self.prototype_len = configs.prototype_len
        self.prototype_shape = [configs.num_prototypes,self.prototype_len,configs.enc_in] # [num_prototypes x d_model x enc_in]
        self.prototype_activation_function ='log'
        
        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape), requires_grad=True) # ？？？
        #self.prototype_vectors = nn.Parameter(torch.zeros(self.prototype_shape), requires_grad=True)
        
        if configs.prototype_init == 'kaiming':
            self.prototype_vectors = nn.init.kaiming_normal_(self.prototype_vectors)
        elif configs.prototype_init == 'xavier':
            self.prototype_vectors = nn.init.xavier_normal_(self.prototype_vectors)
        else:
            
            print("random init prototype")
            
      

        # 使用 Xavier 初始化
        print("self.seq_len",self.seq_len)
      
        self.projection = nn.Linear(
                (self.seq_len-1)* self.num_prototypes, configs.num_class  #### !!!
            )
        
        self.projection_1= nn.Linear(
                 self.num_prototypes*101, configs.num_class)
        
        self.linear_1 = nn.Linear(128, 1)
        self.args = configs
        
        print("!!!!prototype_shape",self.prototype_shape)
        print("!!!!self.patch_num",self.patch_num)
        print("configs.d_model",configs.d_model)
        
    def prototype_layer(self,prototype, prototype_patch=True, if_normalize=False):
        """
        prototyes: [num_prototypes x prototype_len x enc_in]: torch.tensor
        """
        # Normalization from Non-stationary Transformer
        if if_normalize:
            means = prototype.mean(dim=1, keepdim=True).detach()
            prototype = prototype - means
            stdev = prototype.std(dim=1, keepdim=True, unbiased=False) + 1e-5
            prototype = prototype / stdev 
            
        
        prototype_enc=prototype # [num_prototypes, prototype_len, enc_in]
        ##print("!!1!enc_out.shape",prototype_enc.shape)# 
        
        if prototype_patch:
            # do patching and embedding
            prototype_enc = prototype_enc.permute(0, 2, 1) # [num_prototypes,enc_in, prototype_len]
            ##print("!!1.5!enc_out.shape",prototype_enc.shape)
            prototype_out, n_vars = self.patch_embedding(prototype_enc)# [num_prototypes, patch_num, d_model]
            
            ##print("!!2!enc_out.shape",prototype_out.shape)

            #  Encoder
            # z: [bs * nvars x patch_num x d_model]
            prototype_out, attns = self.encoder(prototype_out) # [num_prototypes, patch_num, d_model]
            ##print("!!3!prototype.shape",prototype_out.shape)
            prototype_out=self.linear_1(prototype_out)  # [num_prototypes, patch_num, 1]
    
        
            #prototype=self.prototype_vectors # !!! test
            
            ##print("!!4!prototype.shape",prototype_out.shape)
        
                
            
        
        return prototype_out
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        input:
        x_enc: [bs x seq_len x nvars ]
        
        output: [bs x num_classes]
        """
        x_enc=x_enc+2.5### ！ for freaqshape
        
        x=x_enc # ([4, 1400, 1])
        #print("x.shape",x.shape)
        

        # z: [bs x nvars x patch_num x d_model]
        
        #print("!!3!prototype_out.shape",prototype_out.shape)
        #assert prototype_out.shape[-1]==self.patch_num, "prototype_out.shape[-1] != self.patch_num"

        
        
        prototype=self.prototype_layer(self.prototype_vectors) # [num_prototypes, patch_num, 1]
        if self.args.ablation == 'no_prototype_layer':
            prototype = self.prototype_vectors
        
        # prototype layer distance calculation
        activations=self.convolution_distance_x_as_input(x,prototype) # [bs x seq_len x num_prototypes]
        #activations=self.convolution_distance(prototype_out,prototype)
        
        output=activations.reshape(activations.shape[0],-1)
        ##print("output.shape",activations.shape) 
        
        if False:
            indices=self.prototype_selection(activations,prototype) #[bs x n_selected]
            indices= indices.unsqueeze(-1).expand(-1, -1, self.seq_len) # #[bs x n_selected,seq_len]
            indices=indices.permute(0,2,1) # [bs x seq_len x n_selected]
            output= torch.gather(activations, dim=2, index=indices).reshape(activations.shape[0],-1)

        
        
        output = self.projection(output)
        
        #output=self.projection_1(x.reshape(x.shape[0],-1)) # test
        
        return output, activations, prototype
        
        
    def forward_17_11(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        input:
        x_enc: [bs x seq_len x nvars ]
        
        output: [bs x num_classes]
        """
        x_enc=x_enc+2.5### ！ for freaqshape
        
        x=x_enc # ([4, 1400, 1])
        #print("x.shape",x.shape)
        

        # z: [bs x nvars x patch_num x d_model]
        
        #print("!!3!prototype_out.shape",prototype_out.shape)
        #assert prototype_out.shape[-1]==self.patch_num, "prototype_out.shape[-1] != self.patch_num"

        
        prototype=self.prototype_layer(self.prototype_vectors) # [num_prototypes, patch_num, 1]
        # prototype layer distance calculation
        activations=self.convolution_distance_x_as_input(x,prototype) # [bs x seq_len x num_prototypes]
        #activations=self.convolution_distance(prototype_out,prototype)
        
        
        select_activations=self.seg_activations(activations,prototype) #[ bs x  prototype_len x num_prototypes]
        select_activations=select_activations.permute(0,2,1) # [bs x num_prototypes x prototype_len]
        
        
        avg_pool = torch.nn.AvgPool1d(kernel_size=select_activations.shape[-1])
        
        #select_activations=avg_pool(select_activations)
        #print("select_activations.shape",select_activations.shape)
        
        output=select_activations.reshape(select_activations.shape[0],-1)
        #print("output.shape",output.shape) 
        
       

        
        
        output = self.projection_1(output)
        
        #output=self.projection_1(x.reshape(x.shape[0],-1)) # test
        
        return output, activations, prototype
        
    
    def convolution_distance_x_as_input(self, x, prototype_vectors):
        """
        x: [bs x patch_num x nvars]
        prototype_vectors: [num_prototypes, porto_len, 1]
        """
        
        x=x.permute(0,2,1) # [bs, nvars, patch_num  ]
        
  
        prototype_vectors = prototype_vectors.permute(0, 2, 1)  # [num_prototypes, 1 , porto_len]
        
        # 进行卷积操作
        ##print("In convolution_distance, x.shape",x.shape,"prototype_vectors.shape",prototype_vectors.shape)
        conv_out = F.conv1d(x, prototype_vectors, padding=(prototype_vectors.shape[-1]-1)//2  ,stride=1)  # 输出 [bs, num_prototypes, patch_num]
        
        # 调整输出形状并应用 sigmoid 激活
        conv_out = conv_out.permute(0, 2, 1)  # [bs, patch_num, num_prototypes]
        
        layer_norm = nn.LayerNorm(conv_out.shape[1:]).to(self.device)  # 根据 conv_out 的形状定义 LayerNorm
        conv_out = layer_norm(conv_out)
        # conv_out = torch.sigmoid(conv_out)
        conv_out=F.leaky_relu(conv_out, negative_slope=0.01) ## !!!
        conv_out=F.leaky_relu(conv_out, negative_slope=0.0001) ## !!!
        
        ##print("conv_out.shape",conv_out.shape) # [bs x patch_num x num_prototypes]
        return conv_out

  
    def smoothness_loss(self,sequence):
        """
        not in use
        """
        
        sequence=sequence.reshape(-1)
        # sequence shape: [batch_size, sequence_length]
        diff = sequence[:1:] - sequence[:-1]  # Compute differences between adjacent elements
        return torch.mean(diff ** 2)  # Square differences and take the mean
    
    
    def seg_prototype_loss(self,activations,x,prototype_vectors,outputs,select_prototype=True,no_variance_loss=False):
        
        """
        activatetion:  [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        if self.args.ablation == 'no_variances_loss':
            no_variance_loss = True
        
        
        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes
       
        half_len = prototype_len // 2
        x=x.reshape(batch_size,-1)
        
        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            ##print("p_index",p_index)
            # 使用 torch.topk 获取每个样本中的前10个最大值及其索引
            activation = activations[:,:,p_index].reshape(batch_size, -1)
            
            #get the top values and indices
            ##print("activations.shape",activation.shape)
            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]
            # 创建一个与 activations 形状相同的全零张量


            # 初始化一个张量来存储子序列片段
            subsequences = torch.zeros(batch_size,prototype_len).to(self.device)
            

            # 遍历每个样本的最大值索引位置，提取对应的子序列
            for i in range(batch_size):
                center = top_k_indices[i, 0].item()  # 获取中心索引位置
                # 计算子序列的开始和结束位置，确保不超出边界
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)
                
                # 如果子序列长度不足 50，在序列两端进行补零
                sub_seq = x[i, start:end]
                
                if len(sub_seq) < prototype_len:
                    # 如果在序列的开头不足
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    # 如果在序列的结尾不足
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))
                
                # 将结果存储在 subsequences 中
                
                mean = sub_seq.mean()

                # 将 sub_seq 的均值调整为 0
                sub_seq = sub_seq - mean
                
                subsequences[i] = sub_seq
            
            subsequences=subsequences.unsqueeze(0) # [1 x bs x prototype_len]
            
            subsequence_all_prototype=torch.cat((subsequence_all_prototype,subsequences),dim=0) # [num_prototypes x bs x prototype_len]
            
        
        subsequence_all_prototype=subsequence_all_prototype.permute(1,0,2) # [ bs x num_prototypes x prototype_len]
        
        prototype_repeat=prototype_vectors.permute(2,0,1) #  [1,num_prototypes, patch_num ]
        prototype_repeat=prototype_repeat.repeat(batch_size, 1,1)
        ##print("subsequence_all_prototype.shape",subsequence_all_prototype.shape,"prototype_vectors.shape",prototype_vectors.shape)
        
        
        if select_prototype and self.num_prototypes>1:
            
            #indices=self.prototype_selection(activations,prototype_vectors) #[bs x n_selected]
            
            indices=self.prototype_selection_class(outputs,prototype_vectors) #[bs x 1]
            #print("############indices",indices)
            
            indices= indices.unsqueeze(-1).expand(-1, -1, prototype_len) #
            
            #print("indices.shape",indices[:,0,0])
            
            
       
            
            
            subsequence_all_prototype= torch.gather(subsequence_all_prototype, dim=1, index=indices)
            prototype_repeat= torch.gather(prototype_repeat, dim=1, index=indices)
            
            #print("subsequence_all_prototype.shape",subsequence_all_prototype.shape)
            
        
        #print("subsequence_all_prototype.shape",prototype_repeat)
        
        prototype_loss = torch.nn.functional.mse_loss(subsequence_all_prototype,prototype_repeat)
        #print("@@@prototype_loss",prototype_loss)
        variance_loss = self.total_variance_loss(prototype_vectors)
        
        
        if  self.args.ablation == 'no_matching_loss':
            prototype_loss = 0
        
        
        #print("proto_loss:",prototype_loss,"variance_loss:",variance_loss)
        if self.num_prototypes==1:
            return  prototype_loss
        
        if no_variance_loss:
            return prototype_loss
        else:
            return self.dynamic_loss(prototype_loss,variance_loss)

  
    def seg_activations(self,activations,prototype_vectors):
        
        """
        activatetion:  [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        
        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes
       
        half_len = prototype_len // 2
        #x=x.reshape(batch_size,-1)
        
        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            ##print("p_index",p_index)
            # 使用 torch.topk 获取每个样本中的前10个最大值及其索引
            activation = activations[:,:,p_index].reshape(batch_size, -1)
            
            #get the top values and indices
            ##print("activations.shape",activation.shape)
            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]
            # 创建一个与 activations 形状相同的全零张量


            # 初始化一个张量来存储子序列片段
            subsequences = torch.zeros(batch_size,prototype_len).to(self.device)
            

            # 遍历每个样本的最大值索引位置，提取对应的子序列
            for i in range(batch_size):
                center = top_k_indices[i, 0].item()  # 获取中心索引位置
                # 计算子序列的开始和结束位置，确保不超出边界
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)
                
                # 如果子序列长度不足 50，在序列两端进行补零
                sub_seq =  activations[i,start:end,p_index]  #x[i, start:end] !!!
                
                if len(sub_seq) < prototype_len:
                    # 如果在序列的开头不足
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    # 如果在序列的结尾不足
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))
                
                # 将结果存储在 subsequences 中
                
                mean = sub_seq.mean()

                # 将 sub_seq 的均值调整为 0
                sub_seq = sub_seq - mean
                
                subsequences[i] = sub_seq
            
            subsequences=subsequences.unsqueeze(0) # [1 x bs x prototype_len]
            
            subsequence_all_prototype=torch.cat((subsequence_all_prototype,subsequences),dim=0) # [num_prototypes x bs x prototype_len]
            
        
        subsequence_all_prototype=subsequence_all_prototype.permute(1,2,0) # [ bs x  prototype_len x num_prototypes]
        
       
        ##print("subsequence_all_prototype.shape",subsequence_all_prototype.shape,"prototype_vectors.shape",prototype_vectors.shape)
        
        return subsequence_all_prototype
    
        
        
        


  
  
  
  
    def prototype_selection(self,conv_out,prototype_vectors):
        """
        conv_out: [bs x patch_num x num_prototypes]
        prototype_vectors: [num_prototypes, porto_len, 1]
        """
        # 选择前 n 个原型
        n_selected = self.prototype_num_selected
        
        top_k_values, _ = torch.topk(conv_out, 1, dim=1) # [bs x 1 x num_prototypes]
        
        top_k_values, indices = torch.topk(top_k_values, n_selected, dim=2) # [bs x 1 x n_selected]
        
        indices = indices.squeeze(1) # [bs x n_selected]
        
        #selected_prototypes = torch.gather(conv_out, 1, indices)

        # selected_prototypes has shape [batch_size, n_selected, proto_len]
        
        #print('indices:',indices)
        return indices #[bs x n_selected]
    
    
    def prototype_selection_class(self,output,prototype_vectors):
        """
        output: [bs x num_classes]
        prototype_vectors: [num_prototypes, porto_len, 1]
        
        """        
        
        n_selected = self.prototype_num_selected
        
        indices = torch.argmax(output, dim=1) # [bs]
        
        indices = indices.unsqueeze(-1)
        
        return indices #[bs x 1]
        




    def total_variance_loss(self,prototype_vectors):
        """
            prototype_vectors: [num_prototypes, patch_num, 1]
            """
        prototype_vectors=prototype_vectors.reshape(prototype_vectors.shape[0],-1)
        
        
        
        mean_embedding = torch.mean(prototype_vectors, dim=0, keepdim=True)
        deviations = prototype_vectors - mean_embedding
        variance = torch.mean(torch.sum(deviations ** 2, dim=1))
        # 最大化方差（最小化负的方差）
        loss = -torch.log(variance)
        
        return loss
    
    def dynamic_loss(self,loss1,loss2):
        
        epsilon = 1e-5
        
        loss1=abs(loss1)
        loss2=abs(loss2)

        # 计算权重，使用损失值的倒数
        w1 = 1.0 / (loss1 + epsilon)
        w2 = 1.0 / (loss1 + epsilon)

        # 归一化权重
        sum_w = w1 + w2
        w1 = w1 / sum_w
        w2 = w2 / sum_w

        # 计算总损失
        
        normalized_loss1 = w1 * loss1
        normalized_loss2 = -(w2 * loss2)
        
        #print("normalized_loss1",normalized_loss1,"normalized_loss2",normalized_loss2)
        total_loss = normalized_loss1 + normalized_loss2
        
        return total_loss
    
    def activation_cosine_loss(self,activations):
        """
        activations: [bs x seq_len x num_prototypes]
        """
        # 计算激活值的均值和方差
        if self.num_prototypes==1:
            return 0
        
        else:
        
            X=activations[:,:,0]
            Y=activations[:,:,1]
            
            cosine_sim = F.cosine_similarity(X, Y, dim=1)
            
            #print("mean",mean,"variance",variance)
            # 计算激活值的损失
            loss = 1 - cosine_sim.mean()
            return loss
    
    
    
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