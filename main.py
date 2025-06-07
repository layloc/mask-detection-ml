import argparse
from data_preparation import convert_to_yolo, augment_data
from modeling import train, evaluate

parser = argparse.ArgumentParser(description='Face Mask Detection Pipeline')
parser.add_argument('--prepare-data', action='store_true', help='Prepare dataset')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
parser.add_argument('--run-server', action='store_true', help='Start Flask server')

args = parser.parse_args()

if args.prepare_data:
    print("Preparing data...")
    convert_to_yolo.convert_to_yolo_format('data/raw', 'data/labels', 500, 600)
    augment_data.augment_dataset('data/raw', 'data/labels', 'data/augmented')

if args.train:
    print("Training model...")
    train.train_model()

if args.evaluate:
    print("Evaluating model...")
    evaluate.evaluate_model('modeling/best.pt', 'data/dataset.yaml')

if args.run_server:
    print("Starting server...")
    from inference.app import app

    app.run()