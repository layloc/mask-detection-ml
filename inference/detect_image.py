import cv2
from modeling.ensemble import EnsembleModel
from utils.visualization import draw_boxes


def detect_mask(img, conf_threshold=0.4):
    # Load ensemble model
    model = EnsembleModel([
        'modeling/best_model1.pt',
        'modeling/best_model2.pt'
    ])

    # Perform detection
    boxes, scores, labels = model.predict(img, conf_threshold)

    # Visualize
    vis_img = draw_boxes(img.copy(), boxes, scores, labels)

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
        'visualization': vis_img
    }