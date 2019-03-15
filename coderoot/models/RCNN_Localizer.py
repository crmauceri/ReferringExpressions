# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.structures.tensorlist import to_tensor_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from coderoot.models.LSTM import LanguageModel

import datetime
import logging
import time
import itertools


from maskrcnn_benchmark.utils.metric_logger import MetricLogger

class HHA_RCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(HHA_RCNN, self).__init__()

        self.image_backbone = build_backbone(cfg)
        self.hha_backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.image_backbone.out_channels*2)

        self.roi_heads = None #build_roi_heads(cfg, self.image_backbone.out_channels*2)

    def features_forward(self, image_list, HHA_list):
        RGB_features = self.image_backbone(image_list.tensors)
        HHA_features = self.hha_backbone(HHA_list.tensors)

        image_features = []
        for i in range(len(RGB_features)):  # Number of anchor boxes?
            # Dimensions are (batch, feature, height, width)
            image_features.append(torch.cat((RGB_features[i], HHA_features[i]), 1))

        return image_features

    def predictions_forward(self, image_list, features, targets):
        proposals, proposal_losses = self.rpn(image_list, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return result, losses

    def instance_prep(self, instance, device):
        images, HHAs, targets, img_idx = instance

        images = images.to(device)
        HHAs = HHAs.to(device)
        targets = [target.to(device) for target in targets]

        return images, HHAs, targets, img_idx


    def forward(self, instance, device):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images, HHAs, targets, img_idx = self.instance_prep(instance, device)

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_list = to_image_list(images)
        HHA_list = to_image_list(HHAs)

        image_features = self.features_forward(image_list, HHA_list)

        result, losses = self.predictions_forward(image_list, image_features, targets)

        if self.training:
            return losses

        return result


    def do_train(
        self,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    ):
        logger = logging.getLogger("maskrcnn_benchmark.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(data_loader)
        start_iter = arguments["iteration"]
        self.train()
        start_training_time = time.time()
        end = time.time()
        for iteration, instance in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            scheduler.step()

            loss_dict = self(instance, device)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )


class ReferExp_RCNN(HHA_RCNN):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, vocab):
        super(ReferExp_RCNN, self).__init__(cfg)

        # Text Embedding Network
        self.wordnet = LanguageModel(vocab=vocab, hidden_dim=1024)

        # Ref Localization Network
        self.ref_rpn = build_rpn(cfg, self.image_backbone.out_channels * 2 + 1024)
        self.ref_roi_heads = None  # build_roi_heads(cfg, self.image_backbone.out_channels*2)

    def instance_prep(self, instance, device):
        images, HHAs, sentences, seg_targets, img_idx = instance
        images, HHAs, seg_targets, img_idx = super().instance_prep((images, HHAs, seg_targets, img_idx), device)

        sentences = [s.to(device) for s in sentences]
        ref_targets = []
        for ind, s in enumerate(sentences):
            s.trim()
            s_target = []
            for ann in s.get_field('ann_id'):
                s_target.append(seg_targets[ind].get_field('ann_id').index(ann))
            ref_targets.extend(seg_targets[ind][torch.tensor(s_target)].to_list())

        return images, HHAs, sentences, seg_targets, ref_targets, img_idx

    def predictions_forward(self, image_list, features, targets):
        proposals, proposal_losses = self.ref_rpn(image_list, features, targets)

        if self.ref_roi_heads:
            x, result, detector_losses = self.ref_roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return result, losses

    def forward(self, instance, device):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        images, HHAs, sentences, seg_targets, ref_targets, img_idx = self.instance_prep(instance, device)

        if self.training and seg_targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Calculate image features
        image_list = to_image_list(images)
        HHA_list = to_image_list(HHAs)
        image_features = self.features_forward(image_list, HHA_list)

        # Calculate text features
        sentence_batch = to_tensor_list(sentences)
        self.wordnet.clear_gradients(batch_size=len(sentence_batch))
        sentence_targets = sentence_batch.get_target()

        text_prediction = self.wordnet(sentence_batch)
        text_loss = self.wordnet.loss_function(text_prediction, sentence_targets)
        text_features = self.wordnet.hidden[0]
        text_shape = text_features.shape

        full_feature = [0]*len(image_features)
        image_sizes = []
        for i, feature in enumerate(image_features):
            image_shape = feature.shape
            full_shape = torch.Size((text_shape[1], image_shape[1]+text_shape[2], image_shape[2], image_shape[3]))
            f = torch.zeros(full_shape)

            f[:, image_shape[1]:, :, :] = text_features.reshape((text_shape[1], text_shape[2], 1, 1)).repeat(1, 1, image_shape[2], image_shape[3])
            image_mask = [len(s) for s in sentences]
            for j, repeats in enumerate(image_mask):
                f[sum(image_mask[:j]):sum(image_mask[:j+1]), :image_shape[1], :, :] = feature[j, :, :, :].repeat(repeats, 1, 1, 1)
                image_sizes.extend(list(itertools.repeat(image_list.image_sizes[j], repeats)))
            full_feature[i] = f

        ## Losses and predictions
        result, ref_exp_loss = self.predictions_forward(image_sizes, full_feature, ref_targets)

        if self.training:
            # Normal instance segmentation
            losses = super().predictions_forward(image_list, image_features, seg_targets)[1]

            # Language model
            losses['text_loss'] = text_loss

            # Referring Expression Localization
            losses.update(ref_exp_loss)
            return losses

        return result