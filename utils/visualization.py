def draw_boxes(img, boxes, scores, labels):
    colors = {
        0: (0, 255, 0),  # with_mask - green
        1: (0, 0, 255),  # without_mask - red
        2: (0, 165, 255)  # incorrect - orange
    }
    names = {
        0: 'Mask On',
        1: 'No Mask',
        2: 'Incorrect Mask'
    }

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{names[label]}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # Draw text
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img