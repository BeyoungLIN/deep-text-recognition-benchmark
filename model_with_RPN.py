"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from faster_rcnn.RPN import RPN
from faster_rcnn.shape_spec import ShapeSpec
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

from .model import Model



class Model_with_RPN(Model):

    def __init__(self, opt):
        super(Model_with_RPN, self).__init__(opt)
        input_shape = {'faster_rcnn': ShapeSpec(channels=512, height=51, width=1, stride=0)}
        config = RPN.from_config(input_shape=input_shape)
        self.rpn_net = RPN(
            in_features=config['in_features'],
            head=config['head'],
            anchor_matcher=config['anchor_matcher'],
            box2box_transform=config['box2box_transform'],
            batch_size_per_image=config['batch_size_per_image'],
            positive_fraction=config['positive_fraction'],
            pre_nms_topk=config['pre_nms_topk'],
            post_nms_topk=config['post_nms_topk'],
            nms_thresh=config['nms_thresh'],
            min_box_size=config['min_box_size'],
            anchor_boundary_thresh=config['anchor_boundary_thresh'],
            loss_weight=config['loss_weight'],
            box_reg_loss_type=config['box_reg_loss_type'],
            smooth_l1_beta=config['smooth_l1_beta']
        )

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        if self.opt.page_orient == 'horizontal':
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        elif self.opt.page_orient == 'vertical':
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 2, 1, 3))  # [b, c, h, w] -> [b, h, c, w]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
            return prediction
        elif self.stages['Pred'] == 'Attn':
            if is_train:
                prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                             batch_max_length=self.opt.batch_max_length)
                return prediction
            else:
                prediction, alphas = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                                     batch_max_length=self.opt.batch_max_length)
                return prediction, alphas
        else:
            raise ValueError