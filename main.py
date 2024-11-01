import time
from playwright_execute import click_text_on_page
from OmniParser.omniparser import Omniparser

config = {
    'som_model_path': '/home/dheeraj/OmniParser/weights/icon_detect/best.pt',
    'device': 'cuda',
    'caption_model_path': 'florence2',
    'draw_bbox_config': {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    },
    'BOX_TRESHOLD': 0.5
}

if __name__ == "__main__":
    parser = Omniparser(config)
    image_path = '/home/dheeraj/screenshot_20241101_104037.png'

    s = time.time()
    image, parsed_content_list = parser.parse(image_path)
    print(image)
    print("||||||||||||||||||||||||||||||||||")
    text_to_click = parsed_content_list[-1]['text']
    print(text_to_click)
    device = config['device']
    print(f'Time taken for Omniparser on {device}:', time.time() - s)
    s = time.time()
    click_text_on_page(url="https://example.com", text_to_click=text_to_click)
    print(f'Time taken for playwright to execute instruction :', time.time() - s)
    