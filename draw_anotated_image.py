import cv2
import numpy as np
import os
import imageio.v3 as iio

# Input paths
images_folder_path = os.path.join("data","Input")

# Output path
output_dir = "data/output_anotaded_image"
os.makedirs(output_dir, exist_ok=True)

images_path = [
    os.path.join(images_folder_path, image_path)
    for image_path in os.listdir(images_folder_path)
    if image_path.endswith(".png")]

labels_path = [image_path.replace(".png", ".txt") for image_path in images_path]

for image_path, label_path in zip(images_path, labels_path):
    image_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, image_name.replace(".png", "-anotated.png"))

    # Read image using imageio (preserves correct colors)
    image = iio.imread(image_path)  # shape: (H, W, 4) RGBA

    # Drop alpha channel and convert RGB to BGR for OpenCV drawing
    image = image[:, :, :3]               # Drop alpha (now RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB → BGR

    h, w = image.shape[:2]

    # Read annotations
    with open(label_path, "r") as f:
        lines = f.readlines()

    # Draw all bounding boxes
    for idx, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) != 6:
            continue  # Skip malformed lines

        cls, x_c, y_c, bbox_w, bbox_h, angle_deg = tokens
        x_c = float(x_c) * w
        y_c = float(y_c) * h
        bbox_w = float(bbox_w) * w
        bbox_h = float(bbox_h) * h
        angle_deg = float(angle_deg)

        # Adjust angle if short edge comes first
        if bbox_w > bbox_h:
            long_edge, short_edge = bbox_w, bbox_h
        else:
            long_edge, short_edge = bbox_h, bbox_w
            angle_deg += 90

        rect = ((x_c, y_c), (long_edge, short_edge), angle_deg)
        box = cv2.boxPoints(rect).astype(np.intp)

        # Draw the rotated rectangle
        cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=1)

        # Draw the ID number slightly above the center
        text_pos = (int(x_c), int(y_c) - 10)
        cv2.putText(image, str(idx), text_pos,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),  # Red
                    thickness=2,
                    lineType=cv2.LINE_AA)

    # Save annotated image
    cv2.imwrite(output_image_path, image)
    print(f"Saved annotated image to {output_image_path}")
