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

prompt_vehicle_color = (
    "Return only one word indicating the most likely color of the vehicle's roof "
    "at the center of the image. Choose strictly from this list: "
    "Black, White, Gray, Silver, Blue, Red, Brown, Gold, Green, Tan, Orange, Yellow. "
    "Do not add extra text."
)

prompt_vehicle_type = (
    "Return only one word indicating the most likely type of the vehicle at the center of the image. "
    "Choose strictly from this list: Car, Van, Pickup, Truck, Freight. "
    "Do not add extra text."
)

prompt_vehicle_occlusion = (
    "Return only one numeric value between 0.0 and 1.0 indicating how much of the vehicle's roof "
    "at the center of the image is occluded by other objects. "
    "Return only the number."
)

prompt_vehicle_shadow = (
    "Return only one numeric value between 0.0 and 1.0 indicating how much of the vehicle's roof "
    "at the center of the image is in shadow from other objects. "
    "Return only the number."
)

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

    if bbox_w > bbox_h:
        long_edge, short_edge = bbox_w, bbox_h
    else:
        long_edge, short_edge = bbox_h, bbox_w
        angle_deg += 90

    rect = ((x_c, y_c), (long_edge, short_edge), angle_deg)
    box = cv2.boxPoints(rect)
    box = box.astype(np.intp)
    return box

def get_cropped_image_square_bbox(image, box):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    x, y, w, h = cv2.boundingRect(box)

    side = max(w, h)

    cx = x + w // 2
    cy = y + h // 2
    x_start = max(cx - side // 2, 0)
    y_start = max(cy - side // 2, 0)
    x_end = min(x_start + side, image.shape[1])
    y_end = min(y_start + side, image.shape[0])

    x_start = max(x_end - side, 0)
    y_start = max(y_end - side, 0)

    cropped = image[y_start:y_end, x_start:x_end]

    return cropped

def get_thumbnail(cropped_image):
    return cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)

def get_vehicle_information(square_crop):
    _, buffer = cv2.imencode(".png", square_crop)
    image_bytes = buffer.tobytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    color = post_model_image_prompt(image_b64, prompt_vehicle_color)
    vehicle_type = post_model_image_prompt(image_b64, prompt_vehicle_type)
    occlusion = clean_model_output_float(post_model_image_prompt(image_b64, prompt_vehicle_occlusion))
    shadow = clean_model_output_float(post_model_image_prompt(image_b64, prompt_vehicle_shadow))
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
    headers = {"Content-Type":"application/json"}
    payload = {
        "model": "gemma3:12b",
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
        image_id = int(image_name.replace(".png", ""))

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Could not open Image at {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for vehicle_id, line in enumerate(lines):
            line = line.strip()
            tokens = line.split()
            cls, x_c, y_c, bbox_w, bbox_h, angle_deg = tokens
            box = get_rotated_bbox(x_c, y_c, bbox_w, bbox_h, angle_deg, image_height, image_width)
            cropped_image = get_cropped_image_square_bbox(image, box)
            if cropped_image is None:
                continue
            enhanced_image = get_enhanced_image(cropped_image)
            thumbnail = get_thumbnail(enhanced_image)

            output_image_path = os.path.join(output_dir, f"{image_id}-{vehicle_id}.png")
            cv2.imwrite(output_image_path, thumbnail)
            print(f"Saved: {output_image_path}")

            output_anotated_path = os.path.join(output_dir, f"{image_id}-{vehicle_id}.txt")
            color, vehicle_type, occlusion, shadow = get_vehicle_information(enhanced_image)
            result = f"{cls} {x_c} {y_c} {bbox_w} {bbox_h} {angle_deg} {color} {vehicle_type} {occlusion} {shadow}"
            with open(output_anotated_path, "w") as output_file:
                output_file.write(result)
            print(f"Saved: {output_anotated_path}")

if __name__ == "__main__":
    main()