from pathlib import Path

import cv2
from openvino.runtime import AsyncInferQueue, Core

from .utils import *


class YOLOv5Detect:
    """
    https://github.com/openvinotoolkit/openvino/blob/master/samples/python/classification_sample_async/classification_sample_async.py
    """
    def __init__(self, inferdev, net, n, **kwargs):
        net = Path(net)

        core = Core()
        model = core.read_model(model=net.with_suffix(".xml"), weights=net.with_suffix(".bin"))
        self._infersz = tuple(model.inputs[0].shape)[2:]
        cmodel = core.compile_model(model, inferdev)
        self.inferque = AsyncInferQueue(cmodel, n)
        self.inferque.set_callback(self._cb)

    def detect_multi(self, im0s):
        ys = []
        ims = (self._preprocess(im, self._infersz) for im in im0s)
        for i, (im, r) in enumerate(ims):
            self.inferque.start_async([im], (r, ys))
        self.inferque.wait_all()

        return ys

    @staticmethod
    def _cb(inferreq, args):
        r, ys = args
        y = inferreq.get_output_tensor(3).data.squeeze(0)
        boxes = y[:, :4]
        scores = y[:, 4, None] * y[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= r
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        dets = np.zeros((0, 6), dtype=np.float32) if dets is None else dets
        ys.append(dets)

    @staticmethod
    def _preprocess(img, size, swap=(2, 0, 1)):
        h, w = size
        padded_img = np.ones((h, w, 3), dtype=np.uint8) * 114
        r = min(h / img.shape[0], w / img.shape[1])
        resized_img = cv2.resize(
            img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)
        padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_img /= 255.
        return np.expand_dims(padded_img, 0), r
