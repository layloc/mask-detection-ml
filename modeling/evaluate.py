from ultralytics import YOLO


def evaluate_model(model_path, data_config):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_config,
        split='test',
        conf=0.4,
        iou=0.6,
        plots=True
    )

    print(f"mAP@0.5: {metrics.box.map}")
    print(f"mAP@0.5:0.95: {metrics.box.map50_95}")

    # Per-class metrics
    for i, cls in enumerate(['with_mask', 'without_mask', 'mask_weared_incorrectly']):
        print(f"{cls}: precision={metrics.box.precision[i]:.3f}, recall={metrics.box.recall[i]:.3f}")


if __name__ == "__main__":
    evaluate_model('best.pt', 'data/dataset.yaml')