from ultralytics import YOLO
import yaml


def train_model():
    # Load config
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    # Initialize model
    model = YOLO(cfg['model'])

    # Transfer learning with fine-tuning
    model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        lr0=cfg['lr0'],
        weight_decay=cfg['weight_decay'],
        dropout=cfg['dropout'],
        patience=cfg['patience'],
        freeze=cfg['freeze'],
        cls_weights=cfg['class_weights'],
        pretrained=True,
        device='0'  # GPU
    )

    # Save best model
    model.export(format='onnx')


if __name__ == "__main__":
    train_model()