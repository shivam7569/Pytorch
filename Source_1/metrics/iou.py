import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == "midpoint":
        box_preds_x1s = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box_preds_y1s = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box_preds_x2s = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box_preds_y2s = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box_labels_x1s = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box_labels_y1s = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box_labels_x2s = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box_labels_y2s = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:
        box_preds_x1s = boxes_preds[..., 0:1]
        box_preds_y1s = boxes_preds[..., 1:2]
        box_preds_x2s = boxes_preds[..., 2:3]
        box_preds_y2s = boxes_preds[..., 3:4]

        box_labels_x1s = boxes_labels[..., 0:1]
        box_labels_y1s = boxes_labels[..., 1:2]
        box_labels_x2s = boxes_labels[..., 2:3]
        box_labels_y2s = boxes_labels[..., 3:4]

    x1_common = torch.max(box_preds_x1s, box_labels_x1s)
    y1_common = torch.max(box_preds_y1s, box_labels_y1s)
    x2_common = torch.min(box_preds_x2s, box_labels_x2s)
    y2_common = torch.min(box_preds_y2s, box_labels_y2s)

    intersection = (x2_common - x1_common).clamp(0) * (y2_common - y1_common).clamp(0)

    box_preds_ares = abs((box_preds_x2s - box_preds_x1s) * (box_preds_y2s - box_preds_y1s))
    box_labels_ares = abs((box_labels_x2s - box_labels_x1s) * (box_labels_y2s - box_labels_y1s))

    union = box_preds_ares + box_labels_ares - intersection

    iou = intersection / (union + 1e-6)

    return iou
