import torch
import torch.nn as nn
from torch_geometric import nn as tnn
from .gpnnutil import GlobalNorm4
import einops
class GPNNCell(torch.nn.Module):
    def __init__(self,config):
        super(GPNNCell, self).__init__()
        self.normtype=config.normtype
        self.message_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*2,config.dims),nn.GELU())
        self.edge_fun=nn.Sequential(nn.Linear(config.dims*3,config.dims),nn.GELU(),nn.Dropout(config.dropout),
                                    nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        # self.link_fun=nn.Sequential(nn.Linear(config.dims,config.dims//4),nn.LayerNorm(config.dims//4),nn.GELU(),
        #     nn.Dropout(config.dropout),nn.Linear(config.dims//4,1),nn.Sigmoid())
        self.link_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,1),nn.Sigmoid())
        

        self.residual=tnn.MessageNorm(learn_scale=True)


        self.residual_obj=tnn.MessageNorm(learn_scale=True)
        if self.normtype==0:
            pass
        elif self.normtype==1:
            self.norm=nn.Sequential(nn.Linear(config.dims,config.dims),nn.BatchNorm2d(config.frames),nn.GELU())
            self.norm_obj=nn.Sequential(nn.Linear(config.dims,config.dims),nn.BatchNorm2d(config.frames),nn.GELU())

        elif self.normtype==4:
            self.norm=nn.Sequential(nn.Linear(config.dims,config.dims),GlobalNorm4(config.dims,1,config.worldsize),nn.GELU())
            self.norm_obj=nn.Sequential(nn.Linear(config.dims,config.dims),GlobalNorm4(config.dims,9,config.worldsize),nn.GELU())   
        elif self.normtype==5:
            # layer norm
            self.norm=nn.Sequential(nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims),nn.GELU())
            self.norm_obj=nn.Sequential(nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims),nn.GELU())   
        self.merging=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.tfm = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.gpnn.enc_layer
        )
        self.edges=[]
        self.visual=False

    def clear_visual(self):
        self.edges=[]

    def normalize_score(self,score):
        score=score/(score.max(dim=-2,keepdim=True)[0])
        return score
    
    def forward(self, human_feature,obj_features,edge_features,mask=None,tfm_mask=None):
        B,F,N,D=obj_features.shape
        human_features=human_feature.repeat(1, 1, N, 1)

        tmp_edge=self.edge_fun(torch.cat([torch.cat([human_features,edge_features,obj_features],dim=-1), # human-obj
                                          torch.cat([obj_features,edge_features,human_features],dim=-1)],dim=-2))# obj-human

        if tfm_mask is not None:
            tmp_edge=self.tfm(einops.rearrange(tmp_edge,'b f n d -> (b n) f d'),
                            src_key_padding_mask=tfm_mask)
        else:
            tmp_edge=self.tfm(einops.rearrange(tmp_edge,'b f n d -> (b n) f d'))
        tmp_edge=einops.rearrange(tmp_edge,'(b n) f d -> b f n d',b=B,n=N*2)


        weight_edge=self.link_fun(tmp_edge)
        if self.visual:
            if mask is not None:
                
                # weight_edge_=self.normalize_score((weight_edge*mask)[:,:,:9,:])
                weight_edge_=(weight_edge*mask)[:,:,:9,:]
                # breakpoint()
            else:
                raise ModuleNotFoundError
            self.edges.append(weight_edge_.cpu().detach())
        node_features=torch.cat([human_features,obj_features],dim=-2)


        m_v=self.message_fun(torch.cat([node_features,tmp_edge],dim=-1))
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)

        edge_weighted=weight_edge*m_v
        edge_weighted_human=edge_weighted[:,:,:N,:]
        edge_weighted_obj=edge_weighted[:,:,N:,:]
        edge_weighted_human=torch.sum(edge_weighted_human,-2,keepdim=True)

        human_feature=self.norm(self.residual(human_feature,edge_weighted_human)+human_feature)
        obj_features=self.norm_obj(self.residual_obj(obj_features,edge_weighted_obj)+obj_features)

        return human_feature,obj_features