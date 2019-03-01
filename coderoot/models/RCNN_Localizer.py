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

from maskrcnn_benchmark.structures.bounding_box import BoxList

from coderoot.models.LSTM import LanguageModel

import datetime
import logging
import time

from maskrcnn_benchmark.utils.metric_logger import MetricLogger


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, vocab):
        super(GeneralizedRCNN, self).__init__()

        self.image_backbone = build_backbone(cfg)
        self.hha_backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.image_backbone.out_channels*2)

        self.roi_heads = build_roi_heads(cfg, self.image_backbone.out_channels*2)

        # Text Embedding Network
        self.wordnet = LanguageModel(vocab=vocab, hidden_dim=1024)

    def forward(self, images, HHA, sentences, targets=None, batch_size=5):
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
        #self.wordnet.clear_gradients(batch_size)
        #word_instances, word_targets =  self.wordnet.trim_batch(sentences)

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_list = to_image_list(images)
        RGB_features = self.image_backbone(image_list.tensors)

        HHA_list = to_image_list(HHA)
        HHA_features = self.hha_backbone(HHA_list.tensors)

        image_features = []
        for i in range(len(RGB_features)): # Number of anchor boxes?
            # Dimensions are (batch, feature, height, width)
            image_features.append(torch.cat((RGB_features[i], HHA_features[i]), 1))

        proposals, proposal_losses = self.rpn(image_list, image_features, targets)

        # text_features = self.wordnet(word_instances)
        # text_loss = self.wordnet.loss_function(text_features, word_targets)
        # hidden_text_features = self.wordnet.hidden[0].reshape(1, batch_size, 1024, 1, 1).repeat(1, 1, 1, m, n).squeeze(0)
        #
        # full_feature = [0]*len(image_features)
        # for i, feature in enumerate(image_features):
        #     full_feature[i] = torch.cat((feature, hidden_text_features), 1)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(image_features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = image_features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses.update(text_loss)
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
            images, HHAs, sentences, targets, img_idx, sent_idx = instance
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            scheduler.step()

            images = images.to(device)
            HHAs = HHAs.to(device)
            # sentences = sentences.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = self(images, HHAs, sentences, targets)

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