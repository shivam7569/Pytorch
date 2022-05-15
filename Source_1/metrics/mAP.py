import torch
from collections import Counter
from .iou import intersection_over_union


def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        box_format="corners",
        num_classes=20
):
    # pred_boxes: [[train_idx, class, class_prob, x1, y1, x2, y2], [], []]

    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = list(filter(lambda x: x[1] == c, pred_boxes))
        ground_truths = list(filter(lambda x: x[1] == c, true_boxes))

        count_boxes = Counter([gt[0] for gt in ground_truths])  # Image_0 -> 3 boxes, Image_1 -> 5 boxes => {0: 3, 1: 5}
        for key, val in count_boxes.items():
            count_boxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        if total_true_boxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            img_ground_truths = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0
            best_gt_idx = None

            for gt_idx, gt in enumerate(img_ground_truths):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if count_boxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    count_boxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        recalls = torch.cat([torch.tensor([0]), recalls])
        precisions = torch.cat([torch.tensor([1]), precisions])

        average_precision.append(torch.trapz(precisions, recalls))

    return sum(average_precision) / len(average_precision)
