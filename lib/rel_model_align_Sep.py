########### test in sep on FPN ###########
import numpy as np
import torch
torch.manual_seed(2019)
np.random.seed(2019)

import torch.nn as nn
import torch.nn.parallel
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.functional as F
from config import BATCHNORM_MOMENTUM, BOX_SCALE, IM_SCALE, ModelConfig
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size, bbox_preds, nms_overlaps
from lib.get_union_boxes import NewUnionBoxesAndFeats, UnionBoxesAndFeats
from lib.fpn.proposal_assignments.new_rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg, Result
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener, get_dropout_mask
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import time
from config import IM_SCALE, ROOT_PATH, CO_OCCOUR_PATH
#from lib.ggnn import GGNNObj
#from torch.distributions.uniform import Uniform
#import torch.distributions

MODES = ('sgdet', 'sgcls', 'predcls')
conf = ModelConfig()

def myNNLinear(input_dim, output_dim, bias=True):
    ret_layer = nn.Linear(input_dim, output_dim, bias=bias)
    ret_layer.weight = torch.nn.init.xavier_normal_(ret_layer.weight, gain=1.0)
    return ret_layer

class DynamicFilterContext(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', use_vision=True,
                 embed_dim=200, hidden_dim=512, obj_dim=4096, pooling_dim=4096,
                 pooling_size=7, dropout_rate=0.2, use_bias=True, use_tanh=True, 
                 limit_vision=True, sl_pretrain=False, num_iter=-1, use_resnet=False,
                 reduce_input=False, debug_type=None, post_nms_thresh=0.5, 
                 output_dim=512,):
##               time_step_num=3, use_knowledge=True, knowledge_matrix=''):

        super(DynamicFilterContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.use_vision = use_vision 
        self.use_bias = use_bias
        self.use_tanh = use_tanh
        self.use_highway = True
        self.limit_vision = limit_vision
        self.pooling_dim = pooling_dim 
        self.pooling_size = pooling_size
        self.nms_thresh = post_nms_thresh

        self.obj_compress = myNNLinear(self.pooling_dim, self.num_classes, bias=True)

        self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        if self.use_bias:
            self.freq_bias = FrequencyBias()
        self.reduce_dim = 256
        self.reduce_obj_fmaps = nn.Conv2d(512, self.reduce_dim, kernel_size=1)

        similar_fun = [myNNLinear(self.reduce_dim, self.reduce_dim),
##        similar_fun = [myNNLinear(self.reduce_dim*2, self.reduce_dim),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.5),
#                       nn.BatchNorm2d(num_features=self.reduce_dim),
#                      myNNLinear(self.reduce_dim, self.reduce_dim),
             #          nn.ReLU(inplace=True),
                       myNNLinear(self.reduce_dim, 1)]
        self.similar_fun = nn.Sequential(*similar_fun)

        roi_fmap = [Flattener(),
                    nn.Linear(self.reduce_dim*2*self.pooling_size*self.pooling_size, 4096, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096, bias=True)]
        self.roi_fmap = nn.Sequential(*roi_fmap)

        self.hidden_dim = hidden_dim
        self.rel_compress = myNNLinear(self.hidden_dim*3, self.num_rels)
##        self.rel_compress = myNNLinear(self.pooling_dim, self.num_rels, bias=True)
        self.post_obj = myNNLinear(self.pooling_dim, self.hidden_dim*2)
        self.mapping_x = myNNLinear(self.hidden_dim*2, self.hidden_dim*3)
        self.reduce_rel_input = myNNLinear(self.pooling_dim, self.hidden_dim*3)
        
        def obj_feature_map(self, features, rois):
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        feature_pool2 = RoIAlignFunction(self.pooling_size-2, self.pooling_size-2, spatial_scale=1 / 16)(
            features, rois)
        feature_pool3 = RoIAlignFunction(1, 1, spatial_scale=1 / 16)(
            features, rois)
###        return feature_pool
        return feature_pool, feature_pool2, feature_pool3
       # return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    @property
    def num_classes(self):
        return len(self.classes)
    @property
    def num_rels(self):
        return len(self.rel_classes)

    @property
    def is_sgdet(self):
        return self.mode == 'sgdet'

    @property
    def is_sgcls(self):
        return self.mode == 'sgcls'

    def forward(self, *args, **kwargs):

        results = self.base_forward(*args, **kwargs)
        return results
        
    def base_forward(self, fmaps, obj_logits, im_inds, rel_inds, msg_rel_inds, reward_rel_inds, im_sizes, boxes_priors=None, boxes_deltas=None,$
        assert self.mode == 'sgcls'
        num_objs = obj_logits.shape[0]
#       rel_inds = rel_labels[:, :3].data.clone()
        num_rels = rel_inds.shape[0]  ##506
        rois = torch.cat((im_inds[:, None].float(), boxes_priors), 1)  #[23, 5]
##################################################################
        obj_fmaps = self.obj_feature_map(fmaps, rois)  ##[23, 512, 7, 7]
    
##############################################################################################################################
###############################################################################################################################
        reduce_obj_fmaps_0 = torch.nn.functional.interpolate(self.reduce_obj_fmaps(obj_fmaps[0]), size=(4,4), scale_factor=None, mode='nearest'$
##        reduce_obj_fmaps_1 = self.reduce_obj_fmaps(obj_fmaps[2])  ##[23, 256, 5, 5]
        reduce_obj_fmaps_1 = torch.nn.functional.interpolate(self.reduce_obj_fmaps(obj_fmaps[1]), size=(4,4), scale_factor=None, mode='nearest'$
        reduce_obj_fmaps_2 = self.reduce_obj_fmaps(obj_fmaps[2])  ##[23, 256, 1, 1]

        S_fmaps_0 = reduce_obj_fmaps_0[rel_inds[:, 1]]  ##[506, 256, 4, 4]
        O_fmaps_0 = reduce_obj_fmaps_0[rel_inds[:, 2]]

        S_fmaps_1 = reduce_obj_fmaps_1[rel_inds[:, 1]]  ##[506, 256, 4, 4]
        O_fmaps_1 = reduce_obj_fmaps_1[rel_inds[:, 2]]
  
        S_fmaps_2 = reduce_obj_fmaps_2[rel_inds[:, 1]]  ##[506, 256, 1, 1]
        O_fmaps_2 = reduce_obj_fmaps_2[rel_inds[:, 2]]

        pooling_size_sq = 16

        S_fmaps_trans_0 = S_fmaps_0.view(num_rels, self.reduce_dim, pooling_size_sq).transpose(2, 1)
        O_fmaps_trans_0 = O_fmaps_0.view(num_rels, self.reduce_dim, pooling_size_sq).transpose(2, 1)

        S_fmaps_trans_1 = S_fmaps_1.view(num_rels, self.reduce_dim, pooling_size_sq).transpose(2, 1)
        O_fmaps_trans_1 = O_fmaps_1.view(num_rels, self.reduce_dim, pooling_size_sq).transpose(2, 1)
  
        S_fmaps_trans_2 = S_fmaps_2.view(num_rels, self.reduce_dim, 1).transpose(2, 1)
        O_fmaps_trans_2 = O_fmaps_2.view(num_rels, self.reduce_dim, 1).transpose(2, 1)
        
                SO_fmaps_extend_0 = (S_fmaps_trans_0.unsqueeze(1).expand(-1, pooling_size_sq, -1, -1) + O_fmaps_trans_0.unsqueeze(2).expand(-1, -1, pooling_size_sq, -1)).view(num_rels, pooling_size_sq*pooling_size_sq, self.reduce_dim)
        SO_fmaps_extend_1 = (S_fmaps_trans_1.unsqueeze(1).expand(-1, pooling_size_sq, -1, -1) + O_fmaps_trans_1.unsqueeze(2).expand(-1, -1, pooling_size_sq, -1)).view(num_rels, pooling_size_sq*pooling_size_sq, self.reduce_dim)
        SO_fmaps_extend_2 = (S_fmaps_trans_2.unsqueeze(1).expand(-1, 1, -1, -1) + O_fmaps_trans_2.unsqueeze(2).expand(-1, -1, 1, -1)).view(num_rels, 1, self.reduce_dim)
        
        presence_vector_0 =  torch.sigmoid(self.similar_fun(SO_fmaps_extend_0)).view(num_rels, pooling_size_sq, pooling_size_sq)
        presence_vector_1 =  torch.sigmoid(self.similar_fun(SO_fmaps_extend_1)).view(num_rels, pooling_size_sq, pooling_size_sq) 
        presence_vector_2 =  torch.sigmoid(self.similar_fun(SO_fmaps_extend_2))  

        m_vector_0 = torch.mean(SO_fmaps_extend_0.transpose(2,1), dim=1).view(-1)  ##129536
        m_vector_1 = torch.mean(SO_fmaps_extend_1.transpose(2,1), dim=1).view(-1)
        m_vector_2 = torch.mean(SO_fmaps_extend_2.transpose(2,1), dim=1).view(-1) ##506

        latent_mask_0 = torch.where(F.softmax(presence_vector_0.view(-1), dim=0) >= (1/len(m_vector_0)), torch.ones_like(presence_vector_0.view(-1)), torch.zeros_like(presence_vector_0.view(-1)))
        latent_mask_1 = torch.where(F.softmax(presence_vector_1.view(-1), dim=0) >= (1/len(m_vector_1)), torch.ones_like(presence_vector_1.view(-1)), torch.zeros_like(presence_vector_1.view(-1)))
        latent_mask_2 = torch.where(F.softmax(presence_vector_2.view(-1), dim=0) >= (1/len(m_vector_2)), torch.ones_like(presence_vector_2.view(-1)), torch.zeros_like(presence_vector_2.view(-1)))

        attended_vector_0 =  (m_vector_0 * latent_mask_0).view(num_rels, pooling_size_sq, pooling_size_sq)
        attended_vector_1 =  (m_vector_1 * latent_mask_1).view(num_rels, pooling_size_sq, pooling_size_sq)
        attended_vector_2 =  (m_vector_2 * latent_mask_2).view(num_rels, 1, 1)

        SO_fmaps_scores_0 = F.softmax(attended_vector_0, dim=1)
        SO_fmaps_scores_1 = F.softmax(attended_vector_1, dim=1)
        SO_fmaps_scores_2 = F.softmax(attended_vector_2, dim=1)
        
        SO_fmaps_scores = SO_fmaps_scores_0*SO_fmaps_scores_1
        SO_fmaps_scores = torch.matmul(SO_fmaps_scores.view(num_rels,1,pooling_size_sq*pooling_size_sq).transpose(2, 1) , SO_fmaps_scores_2)  ##[506, 256, 1]
     
        weighted_S_fmaps = torch.matmul(SO_fmaps_scores.view(num_rels,pooling_size_sq,pooling_size_sq).transpose(2, 1), S_fmaps_trans_0)  ##[506, 16, 256]
        last_SO_fmaps = torch.cat((weighted_S_fmaps, O_fmaps_trans_0), dim=2)
        last_SO_fmaps = nn.functional.interpolate(last_SO_fmaps.transpose(2, 1).view(num_rels,self.reduce_dim*2,4,4), size=(7,7))

        obj_feats = self.roi_fmap_obj(obj_fmaps[0].view(rois.size(0), -1))
        
        obj_logits = self.obj_compress(obj_feats)  ##[23, 151]
        obj_dists = F.softmax(obj_logits, dim=1)  ##[23, 151]
        
        pred_obj_cls = obj_dists[:, 1:].max(1)[1] + 1    ##[23]
#        sub_dist = obj_distributions.view(1, obj_num, num_class).expand(obj_num, obj_num, num_class).contiguous().view(-1, num_class)
        # for relationship classification
        rel_input = self.roi_fmap(last_SO_fmaps)  ##[506, 4096]
        subobj_rep = self.post_obj(obj_feats)  ##[23, 1024]
        sub_rep = subobj_rep[:, :self.hidden_dim][rel_inds[:, 1]]  ##[506, 512]
        obj_rep = subobj_rep[:, self.hidden_dim:][rel_inds[:, 2]]

        last_rel_input = self.reduce_rel_input(rel_input)  ##[506, 1536]
        last_obj_input = self.mapping_x(torch.cat((sub_rep, obj_rep), 1))
        triple_rep = nn.ReLU(inplace=True)(last_obj_input + last_rel_input) - (last_obj_input - last_rel_input).pow(2) ##[506, 1536]
        rel_logits = self.rel_compress(triple_rep)

        # follow neural-motifs paper
        if self.use_bias:
            if self.mode in ['sgcls', 'sgdet']:
                rel_logits = rel_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        pred_obj_cls[rel_inds[:, 1]],  ##result.obj_preds[rel_inds[:, 1]],
                        pred_obj_cls[rel_inds[:, 2]],
                        ), 1))
            elif self.mode == 'predcls':
                rel_logits = rel_logits + self.freq_bias.index_with_labels(
                torch.stack((
                        obj_labels[rel_inds[:, 1]],
                        obj_labels[rel_inds[:, 2]],
                        ), 1))
            else:
                raise NotImplementedError

        return pred_obj_cls, obj_logits, rel_logits


class RelModelAlign(nn.Module):

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=512, pooling_dim=2048, use_resnet=False, thresh=0.01,
                 use_proposals=False, rec_dropout=0.2, use_bias=True, use_tanh=True,
                 limit_vision=True, sl_pretrain=False, eval_rel_objs=False, num_iter=-1, reduce_input=False, 
                 post_nms_thresh=0.5,):

        super(RelModelAlign, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        ##self.pooling_size = conf.pooling_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim
        
        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.num_iter = num_iter
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.sl_pretrain = sl_pretrain
        ################################### Added  ################################################
        self.co_occour = np.load(CO_OCCOUR_PATH)
        self.co_occour = self.co_occour / self.co_occour.sum()
        ###########################################################################################
        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        self.context = DynamicFilterContext(self.classes, self.rel_classes, mode=self.mode,
                                            use_vision=self.use_vision, embed_dim=self.embed_dim, 
                                            hidden_dim=self.hidden_dim, obj_dim=self.obj_dim, 
                                            pooling_dim=self.pooling_dim, pooling_size=self.pooling_size, 
                                            dropout_rate=rec_dropout, 
                                            use_bias=self.use_bias, use_tanh=self.use_tanh,
                                            limit_vision=self.limit_vision,
                                            sl_pretrain = self.sl_pretrain,
                                            num_iter=self.num_iter,
                                            use_resnet=use_resnet,
                                            reduce_input=reduce_input,
                                            post_nms_thresh=post_nms_thresh,)
        @property
        def num_classes(self):
            return len(self.classes)
       @property
        def num_rels(self):
            return len(self.rel_classes)

    def get_reward_rel_inds(self, im_inds, box_priors, box_score):    ##get_rel_inds(self, rel_labels, im_inds, box_priores)

        rel_cands = im_inds.data[:, None] == im_inds.data[None]
        rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

        if self.require_overlap:
            rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                   box_priors.data) > 0)
        rel_cands = rel_cands.nonzero()
        if rel_cands.dim() == 0:
            rel_cands = im_inds.data.new(1, 2).fill_(0)
        rel_cands = rel_cands.nonzero()
        if rel_cands.dim() == 0:
            rel_cands = im_inds.data.new(1, 2).fill_(0)

        rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def get_msg_rel_inds(self, im_inds, box_priors, box_score):

        rel_cands = im_inds.data[:, None] == im_inds.data[None]
        rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

        if self.require_overlap:
            rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                   box_priors.data) > conf.overlap_thresh)
        rel_cands = rel_cands.nonzero()
        if rel_cands.dim() == 0:
            rel_cands = im_inds.data.new(1, 2).fill_(0)

        rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds
        
    def get_rel_inds(self, rel_labels, im_inds, box_priors, box_score):

        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            # Require overlap in the test stage
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)
            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)
            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds 

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        # boxes = result.boxes_assigned
        boxes_deltas = result.rm_box_deltas # sgcls is None
        boxes_all = result.boxes_all # sgcls is None
        if (self.training) and (result.rel_labels is None):
            import pdb; pdb.set_trace()
            print('debug')
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data, result.rm_obj_dists.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes, result.rm_obj_dists.data)
        ############################## change to ##########################################
##        rel_inds = self.get_reward_rel_inds(result.rel_labels, im_inds, boxes, result.rm_obj_dists.data)

        reward_rel_inds = None
        #########################################        
        if self.mode == 'sgdet':
            msg_rel_inds = self.get_msg_rel_inds(im_inds, boxes, result.rm_obj_dists.data)
            reward_rel_inds = self.get_reward_rel_inds(im_inds, boxes, result.rm_obj_dists.data)


        if self.mode == 'sgdet':
            result.rm_obj_dists_list, result.obj_preds_list, result.rel_dists_list, result.bbox_list, result.offset_list, \
                result.rel_dists, result.obj_preds, result.boxes_all, result.all_rel_logits = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(), im_inds, rel_inds, msg_rel_inds, 
                                            reward_rel_inds, im_sizes, boxes.detach(), boxes_deltas.detach(), boxes_all.detach(),
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)

        elif self.mode in ['sgcls', 'predcls']:
           # import pdb; pdb.set_trace()
            result.obj_preds, result.rm_obj_logits, result.rel_logits = self.context(
                                            result.fmap.detach(), result.rm_obj_dists.detach(),
                                            im_inds, rel_inds, None, None, im_sizes, boxes.detach(), None, None, self.co_occour,
                                            result.rm_obj_labels if self.training or self.mode == 'predcls' else None)
                                            
        else:
            raise NotImplementedError

        # result.rm_obj_dists = result.rm_obj_dists_list[-1]

        if self.training:
            return result 
       
        if self.mode == 'predcls':
            import pdb; pdb.set_trace()
            print('debug..')
            result.obj_preds = result.rm_obj_labels
            result.obj_scores = Variable(torch.from_numpy(np.ones(result.obj_preds.shape[0],)).float().cuda())
        else:
            twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
            result.obj_scores = F.softmax(result.rm_obj_logits, dim=1).view(-1)[twod_inds]

        # # Bbox regression
        if self.mode == 'sgdet':
            if conf.use_postprocess:
                bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
            else:
                bboxes = result.rm_box_priors
        else:
            bboxes = result.rm_box_priors
            
        rel_scores = F.softmax(result.rel_logits, dim=1)
##        return filter_dets(bboxes, result.obj_scores,
##                           result.obj_preds, rel_inds[:, 1:], rel_scores)
       #################################################################################################
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_scores, gt_boxes, gt_classes, gt_rels)
       ##################################################################################################
    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""

        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])
        
        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
            
        

   
        
