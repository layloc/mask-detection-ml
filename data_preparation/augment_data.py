import albumentations as A
import cv2
import os


def create_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4))


def augment_dataset(images_dir, labels_dir, output_dir, augmentations_per_image=5):
    transform = create_augmentation_pipeline()
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.jpg', '.png')):
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read YOLO format annotations
            bboxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, xc, yc, w, h = map(float, line.split())
                    bboxes.append([xc, yc, w, h])

            for i in range(augmentations_per_image):
                transformed = transform(image=image, bboxes=bboxes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']

                # Save augmented image
                new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, 'images', new_img_name),
                            cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                # Save augmented labels
                new_label_name = new_img_name.replace('.jpg', '.txt')
                with open(os.path.join(output_dir, 'labels', new_label_name), 'w') as f:
                    for bbox in transformed_bboxes:
                        f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")