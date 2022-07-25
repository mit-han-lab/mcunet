import torch
import torch.nn as nn
import numpy as np

# Some of the code is adapted from gluoncv: https://cv.gluon.ai/model_zoo/detection.html

__all__ = ['standard_nms', 'StandardNMS', 'MergeNMS', 'Yolo3Output']


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, offset=0) -> torch.Tensor:
    r""" Returns the IoU of two bounding boxes """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + offset, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + offset, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset)
    b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def standard_nms(prediction: torch.Tensor, valid_thres=0.01, nms_thres=0.45, topk=-1, merge=False, pad_val=-1) -> list:
    r"""
        Input: (class_id, score, x1, y1, x2, y2)
        Returns detections with shape: (class_id, score, x1, y1, x2, y2)
    """

    output = [torch.empty(0) for _ in range(len(prediction))]
    num_boxes = prediction.shape[1]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 1] >= valid_thres]
        if not image_pred.size(0):
            output[image_i] = torch.ones(num_boxes, 6, device=prediction.device).fill_(pad_val)
            continue

        # Sort by it
        image_pred = image_pred[(-image_pred[:, 1]).argsort()]
        if topk > 0:
            # keep top k boxes
            image_pred = image_pred[:topk]

        detections = image_pred
        # Perform non-maximum suppression
        keep_boxes = []
        n_remaining = detections.size(0)
        for i in range(n_remaining):
            large_overlap = torch.gt(bbox_iou(detections[0, 2:6].unsqueeze(0), detections[:, 2:6]), nms_thres)
            label_match = detections[0, 0] == detections[:, 0]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            if merge:
                weights = detections[invalid, 4:5]
                # Merge overlapping bboxes by order of confidence
                detections[0, 2:6] = (weights * detections[invalid, 2:6]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            if detections.size(0) == 0:
                break
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


class StandardNMS(object):

    def __init__(self, nms_valid_thres=0.01, nms_thres=0.45, nms_topk=-1, post_nms=100, pad_val=-1):
        self.nms_valid_thres = nms_valid_thres
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.pad_val = pad_val

    @property
    def merge(self):
        return False

    def set_nms(self, nms_valid_thres=None, nms_thres=None, nms_topk=None, post_nms=None):
        if nms_valid_thres is not None:
            self.nms_valid_thres = nms_valid_thres
        if nms_thres is not None:
            self.nms_thres = nms_thres
        if nms_topk is not None:
            self.nms_topk = nms_topk
        if post_nms is not None:
            self.post_nms = post_nms

    def __call__(self, detections: torch.Tensor):
        if 0 < self.nms_thres < 1:
            box_num = detections.size(1)
            detections = standard_nms(
                prediction=detections,
                valid_thres=self.nms_valid_thres,
                nms_thres=self.nms_thres,
                topk=self.nms_topk,
                merge=self.merge,
                pad_val=self.pad_val,
            )

            for idx, det in enumerate(detections):
                if det.size(0) < box_num:
                    detections[idx] = torch.cat([
                        det,
                        torch.ones(box_num - det.size(0), det.size(1), device=det.device) * self.pad_val,
                    ], dim=0)
            detections = torch.stack(detections)

            if self.post_nms > 0:
                detections = detections[:, 0:self.post_nms, :]

        ids = detections[..., 0:1]
        scores = detections[..., 1:2]
        bboxes = detections[..., 2:6]
        return ids, scores, bboxes

    @staticmethod
    def build_from_config(config) -> 'StandardNMS':
        return StandardNMS(
            nms_valid_thres=config.get('nms_valid_thres', 0.01),
            nms_thres=config.get('nms_thres', 0.45),
            nms_topk=config.get('nms_topk', 400),
            post_nms=config.get('post_nms', 100),
            pad_val=config.get('pad_val', -1)
        )


class MergeNMS(StandardNMS):

    @property
    def merge(self):
        return True

    @staticmethod
    def build_from_config(config) -> 'MergeNMS':
        return MergeNMS(
            nms_valid_thres=config.get('nms_valid_thres', 0.01),
            nms_thres=config.get('nms_thres', 0.45),
            nms_topk=config.get('nms_topk', 400),
            post_nms=config.get('post_nms', 100),
            pad_val=config.get('pad_val', -1)
        )


class Yolo3Output(nn.Module):
    r""" YOLO3 output layer.
    Parameters
    ----------
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.
    alloc_size : list of int, default is [128, 128]
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    """

    def __init__(self, num_class: int, anchors: list, stride: int, alloc_size=None):
        super(Yolo3Output, self).__init__()

        self.num_class = num_class
        self.anchors = anchors
        self.stride = stride
        self.alloc_size = [128, 128] if alloc_size is None else alloc_size

        np_anchors = np.array(anchors).astype('float32')
        self._num_pred = 1 + 4 + num_class  # 1 objectness + 4 box + num_class
        self._num_anchors = np_anchors.size // 2

        # register buffer
        np_anchors = np_anchors.reshape((1, 1, -1, 2))  # (1, 1, 3, 2)
        self.register_buffer('anchors_buffer', torch.from_numpy(np_anchors))  # (1, 1, 3, 2)

        # offsets will be added to predictions
        grid_x = np.arange(self.alloc_size[1])
        grid_y = np.arange(self.alloc_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # stack to (n, n, 2)
        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
        offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
        self.register_buffer('offsets_buffer', torch.from_numpy(offsets))  # (1, 1, 128, 128, 2)

    @property
    def total_pred_num(self):
        return self._num_anchors * self._num_pred

    def forward(self, x):
        r"""
        During training, return (bbox, raw_box_centers, raw_box_scales, objness, class_pred, anchors, offsets).
        During inference, return detections.
        """
        # prediction flat to (batch, pred per pixel, height * width)
        pred = x.reshape((x.size(0), self.total_pred_num, -1))
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.permute(0, 2, 1)
        pred = pred.view(pred.size(0), -1, self._num_anchors, self._num_pred)
        # components
        raw_box_centers = pred[..., 0:2]  # (batch, height * width, num_anchor, 2)
        raw_box_scales = pred[..., 2:4]  # (batch, height * width, num_anchor, 2)
        objness = pred[..., 4:5]  # (batch, height * width, num_anchor, 1)
        class_pred = pred[..., 5:]  # (batch, height * width, num_anchor, n_classes)

        # valid offsets, (1, 1, height, width, 2)
        offsets = self.offsets_buffer[:, :, 0:x.size(2), 0:x.size(3), :]
        # reshape to (1, height * width, 1, 2)
        offsets = offsets.reshape((1, -1, 1, 2))

        box_centers = (torch.sigmoid(raw_box_centers) + offsets) * self.stride
        box_scales = torch.exp(raw_box_scales) * self.anchors_buffer
        wh = box_scales / 2.0
        bbox = torch.cat([box_centers - wh, box_centers + wh], dim=-1)  # (batch, height * width, num_anchor, 4)

        if self.training:
            # during training, we don't need to convert whole bunch of info to detection results
            return bbox.reshape((bbox.size(0), -1,
                                 4)), raw_box_centers, raw_box_scales, objness, class_pred, self.anchors_buffer, offsets

        confidence = torch.sigmoid(objness)
        class_score = torch.sigmoid(class_pred) * confidence

        # prediction per class
        bboxes = torch.repeat_interleave(bbox.unsqueeze(0), repeats=self.num_class, dim=0)
        scores = class_score.permute(3, 0, 1, 2).unsqueeze(axis=-1)
        ids = scores * 0 + torch.arange(0, self.num_class, device=x.device).reshape((self.num_class, 1, 1, 1, 1))
        detections = torch.cat([ids, scores, bboxes], dim=-1)
        # reshape to (B, xx, 6)
        detections = detections.permute(1, 0, 2, 3, 4)
        detections = detections.reshape(detections.size(0), -1, 6)
        return detections

    @staticmethod
    def build_from_config(config) -> 'Yolo3Output':
        return Yolo3Output(
            num_class=config['output']['num_class'],
            anchors=config['output']['anchors'],
            stride=config['output']['stride'],
            alloc_size=config['output']['alloc_size'],
        )
