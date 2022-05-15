import torch
from .iou import intersection_over_union


def non_max_suppression(
        bboxes,
        iou_threshold,
        prob_threshold,
        box_format="corners"
):

    # prdictions: [[class, class_prob, x1, y1, x2, y2], [], []]

    assert type(bboxes) is list

    bboxes = list(filter(lambda x: x[1] > prob_threshold, bboxes))
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    boxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or
            intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format) < iou_threshold
        ]

        boxes_after_nms.append(chosen_box)

    return boxes_after_nms
