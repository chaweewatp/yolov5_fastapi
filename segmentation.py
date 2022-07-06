import torch
from PIL import Image
import io
import requests

def get_yolov5():
    # local best.pt
    model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')  # local repo
    model.conf = 0.5
    return model


def get_image_from_bytes(binary_image, max_size=416):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    # input_image = Image.open(io.BytesIO(binary_image))

    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image

def get_image_from_url(url, max_size=416):
    input_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


def get_meter():
    model = torch.hub.load('./yolov5', 'custom', path='./model/best_meter_detection_m.pt', source='local')  # local repo
    model.conf = 0.3
    return model

def get_meter_component():
    model = torch.hub.load('./yolov5', 'custom', path='./model/best_meter_component_x.pt', source='local')  # local repo
    model.conf = 0.15
    return model


def get_kwhr():
    model = torch.hub.load('./yolov5', 'custom', path='./model/best_kwhr.pt', source='local')  # local repo
    model.conf = 0.3
    return model

def get_number():
    model = torch.hub.load('./yolov5', 'custom', path='./model/best_meter_number_x.pt', source='local')  # local repo
    model.conf = 0.3
    return model
