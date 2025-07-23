import os
import cv2
import numpy as np
import requests
import json
import base64
from ISR.models import RDN

# Input paths
images_folder_path = os.path.join("data","Input")
# Output path
output_dir = "data/output"

ollama = "http://127.0.0.1:11434/api/generate"
rdn = RDN(weights='psnr-large')

prompt_vehicle_color = "Give me only one word saying the vehicle main color picking from this list: Black, White, Gray, Silver, Blue, Red, Brown, Gold, Green, Tan, Orange, Yellow"
prompt_vehicle_type = "Give me only one word saying the vehicle main type from this list: Car, Van, Pickup, Truck, Freight"
prompt_occlusion = "Give me only one value between 0.0 and 1.0 expressing how much of the vehicle is occluded by other objects. If you cannot determine say 0 only"
prompt_shadow = "Give me only one value between 0.0 and 1.0 expressing how much of the vehicle is inside shadow from other objects. If you cannot determine say 0 only"

os.makedirs(output_dir, exist_ok=True)

def get_image_paths(images_folder_path):
    return [
    os.path.join(images_folder_path, image_path)
    for image_path in os.listdir(images_folder_path)
    if image_path.endswith(".png")]

def get_labels_path(images_path):
    return [image_path.replace(".png", ".txt") for image_path in images_path]

def get_rotated_bbox(x_c, y_c, bbox_w, bbox_h, angle_deg, image_width, image_height):
    x_c = float(x_c) * image_width
    y_c = float(y_c) * image_height
    bbox_w = float(bbox_w) * image_width
    bbox_h = float(bbox_h) * image_height
    angle_deg = float(angle_deg)

    # Ensure long edge is aligned with angle (le90)
    if bbox_w > bbox_h:
        long_edge, short_edge = bbox_w, bbox_h
    else:
        long_edge, short_edge = bbox_h, bbox_w
        angle_deg += 90

    # Create the rotated rectangle
    rect = ((x_c, y_c), (long_edge, short_edge), angle_deg)
    box = cv2.boxPoints(rect)  # Get 4 corner points
    box = box.astype(np.intp)
    return box

def get_square_bbox(box, image_width, image_height):
    # Compute bounding square around the rotated box
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Expand to square
    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height)

    # Center the square
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    half_side = side // 2
    crop_x1 = max(center_x - half_side, 0)
    crop_y1 = max(center_y - half_side, 0)
    crop_x2 = min(center_x + half_side, image_width)
    crop_y2 = min(center_y + half_side, image_height)
    return (crop_x1, crop_y1, crop_x2, crop_y2)

def get_cropped_image(image, square_bbox):
    crop_x1, crop_y1, crop_x2, crop_y2 = square_bbox
    return image[crop_y1:crop_y2, crop_x1:crop_x2]

def get_cropped_image_rotated_bbox(image, box):
    # Ensure image is BGR
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Create a white background
    white_bg = np.ones_like(image, dtype=np.uint8) * 255

    # Create a mask for the rotated bounding box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)

    # Apply mask over white background
    masked_image = np.where(mask[:, :, None] == 255, image, white_bg)

    # Get bounding rectangle of rotated bbox
    x, y, w, h = cv2.boundingRect(box)

    # Crop the region from masked image
    cropped = masked_image[y:y+h, x:x+w]

    # Determine the size of the square (longest side)
    side = max(w, h)

    # Center the cropped image on a white square canvas
    square_image = np.ones((side, side, 3), dtype=np.uint8) * 255

    # Compute top-left corner for pasting cropped image
    x_offset = (side - w) // 2
    y_offset = (side - h) // 2

    square_image[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    return square_image

def get_thumbnail(square_crop):
    return cv2.resize(square_crop, (64, 64), interpolation=cv2.INTER_AREA)

def get_vehicle_information(square_crop):
    _, buffer = cv2.imencode('.png', square_crop)
    image_bytes = buffer.tobytes()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    color = post_model_image_prompt(image_b64, prompt_vehicle_color)
    vehicle_type = post_model_image_prompt(image_b64, prompt_vehicle_type)
    occlusion = clean_model_output_float(post_model_image_prompt(image_b64, prompt_occlusion))
    shadow = clean_model_output_float(post_model_image_prompt(image_b64, prompt_shadow))
    return (color, vehicle_type, occlusion, shadow)

def clean_model_output_float(float_string):
    float_value = 0.0
    float_string = float_string.replace(" ", "")
    try:
        float_value = float(float_string)
    except:
        print(float_string)
    return float_value

def post_model_image_prompt(image, prompt):
    headers = {'Content-Type':'application/json'}
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "images": [image],
        "stream": False
    }
    req = requests.post(ollama, json.dumps(payload), headers=headers)
    if req.status_code == 200:
        response = json.loads(req.text)
        return response["response"]
    return {"response": ""}

def get_enhanced_image(image):
    return rdn.predict(image[:,:,:3])

def main():
    images_path = get_image_paths(images_folder_path)
    labels_path = get_labels_path(images_path)

    for image_path, label_path in zip(images_path, labels_path):
        image_name = os.path.basename(image_path)
        image_id = int(image_name.replace('.png', ''))

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # RGBA
        if image is None:
            print(f"Could not open Image at {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        # Read first line of annotation file
        with open(label_path, "r") as f:
            lines = f.readlines()

        for vehicle_id, line in enumerate(lines):
            line = line.strip()
            tokens = line.split()
            cls, x_c, y_c, bbox_w, bbox_h, angle_deg = tokens
            box = get_rotated_bbox(x_c, y_c, bbox_w, bbox_h, angle_deg, image_height, image_width)
            square_bbox = get_square_bbox(box, image_width, image_height)
            #cropped_image = get_cropped_image(image, square_bbox)
            cropped_image = get_cropped_image_rotated_bbox(image, box)
            thumbnail = get_thumbnail(cropped_image)

            output_image_path = os.path.join(output_dir, f"{image_id}-{vehicle_id}.png")
            enhanced_image_path = os.path.join(output_dir, f"{image_id}-{vehicle_id}-enhanced.png")
            cv2.imwrite(output_image_path, thumbnail)
            print(f"Saved: {output_image_path}")

            output_anotated_path = os.path.join(output_dir, f"{image_id}-{vehicle_id}.txt")
            enhanced_image = get_enhanced_image(cropped_image)
            cv2.imwrite(enhanced_image_path, enhanced_image)
            color, vehicle_type, occlusion, shadow = get_vehicle_information(enhanced_image)
            with open(output_anotated_path, "w") as output_file:
                result = f"{cls} {x_c} {y_c} {bbox_w} {bbox_h} {angle_deg} {color} {vehicle_type} {occlusion} {shadow}"
                output_file.write(result)
            print(f"Saved: {output_anotated_path}")

if __name__ == "__main__":
    main()