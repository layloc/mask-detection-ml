import numpy as np


def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5):
    aps = []
    for class_id in range(3):  # For each class
        # Filter predictions and truths by class
        class_preds = [p for p in pred_boxes if p['class'] == class_id]
        class_truths = [t for t in true_boxes if t['class'] == class_id]

        # Sort predictions by confidence
        class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)

        # Calculate precision-recall curve
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            matched = False
            for truth in class_truths:
                iou = calculate_iou(pred['box'], truth['box'])
                if iou > iou_threshold:
                    matched = True
                    class_truths.remove(truth)
                    break

            if matched:
                tp[i] = 1
            else:
                fp[i] = 1

        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(class_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # AP calculation (area under PR curve)
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            prec_at_r = precisions[recalls >= r]
            ap += prec_at_r.max() if len(prec_at_r) > 0 else 0
        ap /= 11

        aps.append(ap)

    return np.mean(aps)  # mAP