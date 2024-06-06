import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (constant_init, normal_init)
from .modules import MixerLayer, FCBlock, BasicBlock
import sys, os
sys.path.append(os.path.join(__file__.replace(os.path.basename(__file__), ''), '..', '..', '..', '..'))

from tokenization.models.vanilla_pose_vqvae import DecodeTokens as VanillaDecodeTokens

class Proxy(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.set_gpu = False
    def tokenize(self, x):
        if not self.set_gpu:
            self.tokenizer = self.tokenizer.to(x.device)
            self.set_gpu = True
        return self.tokenizer(x)

class TokenClassfier(nn.Module):
    """ Head of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

        The pipelines of two stage during training and inference:

        Tokenizer Stage & Train: 
            Joints -> (Img Guide) -> Encoder -> Codebook -> Decoder -> Recovered Joints
            Loss: (Joints, Recovered Joints)
        Tokenizer Stage & Test: 
            Joints -> (Img Guide) -> Encoder -> Codebook -> Decoder -> Recovered Joints

        Classifer Stage & Train: 
            Img -> Classifier -> Predict Class -> Codebook -> Decoder -> Recovered Joints
            Joints -> (Img Guide) -> Encoder -> Codebook -> Groundtruth Class
            Loss: (Predict Class, Groundtruth Class), (Joints, Recovered Joints)
        Classifer Stage & Test: 
            Img -> Classifier -> Predict Class -> Codebook -> Decoder -> Recovered Joints
            
    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        in_channels (int): Feature Dim of the backbone feature.
        image_size (tuple): Input image size.
        num_joints (int): Number of annotated joints in the dataset.
        cls_head (dict): Config for PCT classification head. Default: None.
        tokenizer (dict): Config for PCT tokenizer. Default: None.
        loss_keypoint (dict): Config for loss for training classifier. Default: None.
    """

    def __init__(self, in_channels=2048, token_num=40, token_class_num=2046, token_code_dim=None, \
                 tokenizer_checkpoint_path=None, tokenizer_type='Vanilla'):
        super().__init__()

        self.conv_num_blocks = 1
        self.dilation = 1
        self.conv_channels = 256
        self.hidden_dim = 64
        self.num_blocks = 4
        self.hidden_inter_dim = 256
        self.token_inter_dim = 64
        self.dropout = 0.0

        self.token_num = token_num # number of token
        self.token_class_num = token_class_num # token class number

        self.token_code_dim = token_code_dim

        #input_size = 12 * 14
        self.mixer_trans = FCBlock(
            in_channels, #self.conv_channels * input_size, 
            self.token_num * self.hidden_dim)

        self.mixer_head = nn.ModuleList(
            [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                self.token_num, self.token_inter_dim,  
                self.dropout) for _ in range(self.num_blocks)])
        self.mixer_norm_layer = FCBlock(
            self.hidden_dim, self.hidden_dim)

        self.class_pred_layer = nn.Linear(self.hidden_dim, self.token_class_num)
        
        # Use the pretrained decoder
        tokenizer_proxy = Proxy(eval(f'{tokenizer_type.capitalize()}DecodeTokens')(tokenizer_checkpoint_path))
        self.tokenize = tokenizer_proxy.tokenize


    def forward(self, x):
        """Forward function."""
        batch_size = x.shape[0]
        # B x 1024 -> B x (token_num x hidden_dim)
        cls_feat = self.mixer_trans(x)
        cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)

        for mixer_layer in self.mixer_head:
            cls_feat = mixer_layer(cls_feat)
        cls_feat = self.mixer_norm_layer(cls_feat)

        # logits: B x token_num x token_class_num
        cls_logits = self.class_pred_layer(cls_feat) 

        # cls_logits_softmax ((B * token_number)x token_dim)
        cls_logits_softmax = cls_logits.softmax(-1)
        smpl_thetas6D = self.tokenize(cls_logits_softmax) # B x 21 x 6

        smpl_thetas6D = smpl_thetas6D.reshape(batch_size, -1)
        return smpl_thetas6D, cls_logits_softmax

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self, conv_channels, conv_num_blocks, dilation):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            conv_channels,
            conv_channels,
            conv_num_blocks,
            dilation=dilation)
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def frozen_tokenizer(self):
        self.tokenize.eval()
        for name, params in self.tokenize.named_parameters():
            params.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
