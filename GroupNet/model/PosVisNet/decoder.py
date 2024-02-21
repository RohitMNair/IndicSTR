import torch.nn as nn
import torch
import lightning.pytorch as pl

from torch import Tensor

class PosVisDecoder(pl.LightningModule):
    """
    Position Visual Decoder which performs MHA over learned positional encodings and encoder visual embeddings
    """
    def __init__(self, hidden_size:int= 1024, mlp_ratio:float= 4.0, layer_norm_eps:float= 1.0e-12, max_grps:int= 25,
                 num_heads:int= 6, hidden_dropout_prob:float= 0.0, attention_probs_dropout_prob:float= 0.0, qkv_bias:bool= True)-> None:
        """
        Constructor for GroupDecoder
        Args:
        - hidden_size (int): sizes of linear layers in the Multi-head attention blocks (default: 1024)
        - mlp_ratio (float): size of the hidden_layer to intermediate layer in MLP block (default: 4.0)
        - layer_norm_eps (float): layer norm epsilon
        - max_grps (int): Maximum groups the decoder will need to predict (default: 25)
        - num_heads (int): # of heads in the position-visual MHA block (default: 6)
        - hidden_dropout_prob (float): dropout probability for the hidden linear layers (default: .0)
        - attention_probs_dropout_prob (float): dropout probability for MHA Blocks
        """
        super().__init__()
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.max_grps = max_grps
        self.num_heads = num_heads
        self.intermediate_size = int(self.mlp_ratio * self.hidden_size)
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.qkv_bias = qkv_bias

        self.positional_encodings = nn.Parameter(torch.randn(self.max_grps, self.hidden_size))
        self.hidden_dropout1 = nn.Dropout(self.hidden_dropout_prob)
        self.pos_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps=self.layer_norm_eps)
        self.pos_vis_mha = nn.MultiheadAttention(
                                            embed_dim= self.hidden_size, 
                                            num_heads= self.num_heads, 
                                            dropout= self.attention_probs_dropout_prob, 
                                            add_bias_kv= self.qkv_bias,
                                            batch_first= True,
                                        )
        self.hidden_dropout2 = nn.Dropout(self.hidden_dropout_prob)
        self.mlp_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps= self.layer_norm_eps)
        self.mlp_1 = nn.Linear(self.hidden_size, self.intermediate_size, bias = True)
        self.act = nn.GELU()
        self.hidden_dropout3 = nn.Dropout(self.hidden_dropout_prob)
        self.mlp_2 = nn.Linear(self.intermediate_size, self.hidden_size, bias = True)
        self.hidden_dropout4 = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, x:Tensor)->tuple:
        """
        Performs Position-Visual attention for identifying the positions of the character groups
        Args:
        - x (Tensor): Vector patch representations from a Vision encoder; shape: (BS x N x hidden size)

        Returns:
        - tuple(Tensor, Tensor): output patch representations shape: (BS x Max Grps x hidden size)
        """
        x = self.hidden_dropout1(x)
        # ViT outputs are layer norm'd
        x_1, pos_vis_attn_weights = self.pos_vis_mha( # expand positional enc. along the batch dimension
                                                query= self.pos_norm(self.positional_encodings.expand(x.shape[0],-1,-1)),
                                                key= x,
                                                value= x,
                                                need_weights = True,
                                            )
        x = self.positional_encodings + self.hidden_dropout2(x_1)
        # structure kept same as PARSeq decoder
        x_1 = self.mlp_2(self.hidden_dropout3(self.act(self.mlp_1(self.mlp_norm(x)))))
        x = x + self.hidden_dropout4(x_1)
        
        return x_1, pos_vis_attn_weights