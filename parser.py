from OmniParser.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)
import torch
from ultralytics import YOLO
from PIL import Image
import base64
import matplotlib.pyplot as plt
import io
from typing import Any


def load_caption_model(
    model_name: str = "florence2",
    model_path: str = "/home/dheeraj/OmniParser/weights/icon_caption_florence",
    device: Any = None,
):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    som_model = get_yolo_model(model_path="/home/dheeraj/OmniParser/weights/icon_detect/best.pt")
    som_model.to(device)
    print("model to {}".format(device))

    # two choices for caption model: fine-tuned blip2 or florence2
    # caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2", device=device)
    caption_model_processor = get_caption_model_processor(
        model_name=model_name,
        model_name_or_path=model_path,
        device=device,
    )
    som_model.device, type(som_model)
    return som_model, caption_model_processor


def parse_image(
    image_path: str = None,
    box_threshold: int = 0.03,
    draw_bbox_config: dict = {
        "text_scale": 0.8,
        "text_thickness": 2,
        "text_padding": 3,
        "thickness": 3,
    },
    cnt: int = 0,
):
    som_model, caption_model_processor = load_caption_model()
    BOX_TRESHOLD = box_threshold

    image = Image.open(image_path)
    image_rgb = image.convert("RGB")

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
    )
    text, ocr_bbox = ocr_bbox_rslt

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path,
        som_model,
        BOX_TRESHOLD=BOX_TRESHOLD,
        output_coord_in_ratio=False,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=0.1,
    )
    return dino_labled_img, label_coordinates, parsed_content_list


def display_results(dino_labled_img: Any = None):
    plt.figure(figsize=(12, 12))
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    plt.axis("off")
    plt.imshow(image)
    
if __name__ == "__main__":
    dino_labled_img, label_coordinates, parsed_content_list = parse_image(image_path='/home/dheeraj/screenshot_20241101_104037.png')
    display_results(dino_labled_img)