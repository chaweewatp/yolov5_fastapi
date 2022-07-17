from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes, get_meter_component, get_kwhr, get_number,get_meter, get_image_from_url
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware


# easy ocr

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import easyocr
reader = easyocr.Reader(['en'])

# model = get_yolov5()
model_meter=get_meter()
model_meter_component = get_meter_component()
model_kwhr = get_kwhr()
model_number = get_number()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


# @app.post("/object-to-json")
# async def detect_food_return_json_result(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     # results = model(input_image)
#     results = model(input_image, size=416)
#     results.print()  
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#     detect_res = json.loads(detect_res)
#     return {"result": detect_res}


# @app.post("/object-to-img")
# async def detect_food_return_base64_img(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     # results = model(input_image)
#     results = model(input_image, size=416)
#     results.print()  

#     results.render()  # updates results.imgs with boxes and labels
#     for img in results.imgs:
#         bytes_io = io.BytesIO()
#         img_base64 = Image.fromarray(img)
#         img_base64.save(bytes_io, format="jpeg")
#     return Response(content=bytes_io.getvalue(), media_type="image/jpeg")




# @app.post("/object-to-json2")
# async def detect_meter_return_json_result(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     # results = model(input_image)
#     results = model(input_image, size=416)
#     results.print()
#     results.crop()
#     detect_res="test"
#     return {"result": detect_res}

# @app.post("/extract_meter_component")
# async def detect_meter_component_return_json_result(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model_meter_component(input_image, size=416)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#     detect_res = json.loads(detect_res)

#     return {"result": detect_res}

# @app.post("/extract_kwhr")
# async def detect_kwhr_return_json_result(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model_kwhr(input_image, size=416)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#     detect_res = json.loads(detect_res)
    
#     return {"result": detect_res}

# @app.post("/extract_meter")
# async def detect_meter_kwhr_return_json_result(file: bytes = File(...)):
#     # detect meter component
#     input_image = get_image_from_bytes(file)
    
#     newsize = (416, 416)
#     input_image = input_image.resize(newsize)
#     input_image.save("input.jpg")

#     results = model_meter_component(input_image, size=416)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#     detect_res = json.loads(detect_res)
#     kwhr_detected=False
#     for res in detect_res:
#         if res['name']=='kwhr':
#             ymin_kwhr=res['ymax']
#             ymax_kwhr=res['ymin']
#             xmin_kwhr=res['xmin']
#             xmax_kwhr=res['xmax']
#             kwhr_detected=True
#         elif res['name']=='meter_no':
#             ymin_no=res['ymax']
#             ymax_no=res['ymin']
#             xmin_no=res['xmin']
#             xmax_no=res['xmax']
#             no_detected=True
            
#     list_kwhr=[]
#     #if kwhr_component exists
#     if (kwhr_detected):
#         # crop image
#         print("croping image")

#         croped_image = input_image.crop((xmin_kwhr, ymax_kwhr, xmax_kwhr, ymin_kwhr))
#         croped_image.save("croped_kwhr.jpg")

#         print("image croped")
#         # resize image to 416x416 here
#         newsize = (416, 416)
#         croped_image = croped_image.resize(newsize)
#         croped_image.save("croped_number2.jpg")

#         # get kwhr
#         results = model_kwhr(croped_image, size=416)
#         detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#         detect_res = json.loads(detect_res)
#         print(detect_res)
#         # re-arange
#         res={}
#         for item in detect_res:
#             res[item['xmin']]=item['name']
#         list_xmin= sorted(res)
#         for _xmin in list_xmin:
#             list_kwhr.append(res[_xmin])
    
#     list_no=[]
#     if (no_detected):
#         # crop image
#         print("croping image")
#         croped_image = input_image.crop((xmin_no, ymax_no, xmax_no, ymin_no))
#         croped_image.save("croped_number.jpg")

#         print("image croped")
#         # resize image to 416x416 here
#         newsize = (416, 416)
#         croped_image = croped_image.resize(newsize)
#         croped_image.save("croped_number2.jpg")

#         # get kwhr
#         results = model_number(croped_image, size=416)
#         detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
#         detect_res = json.loads(detect_res)
#         print(detect_res)
#         # re-arange
#         res={}
#         for item in detect_res:
#             res[item['xmin']]=item['name']
#         list_xmin= sorted(res)
#         for _xmin in list_xmin:
#             list_no.append(res[_xmin])

#         return {"result": {"kwhr":list_kwhr, "no":list_no}}
#     else:
#         return {"result": " no kwhr data"}

def get_xy(dict_res, class_name):
    for res in dict_res:
        if res['name']==class_name:
            return res['ymax'], res['ymin'], res['xmin'], res['xmax'], True
    return '_', '_','_','_', False


def model_return_json(_model, _input_img, loc, img1='crop_meter.jpg', img2='resize_crop_meter.jpg', img_size=(416,416)):
    _croped_img = _input_img.crop(loc)
    _croped_img.save(img1)
    _croped_img = _croped_img.resize(img_size)
    _croped_img.save(img2)
    results = _model(_croped_img, size=416)
    _detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    _detect_res = json.loads(_detect_res)
    return _croped_img, _detect_res
    
# @app.post("/extract_easyocr")
# async def detect_meter_infomation_return_json_result(url: str = File(...)):
#     # bounds = reader.readtext('1_input_image.jpg')
#     bounds = reader.readtext('download (2).png')
#     return_data=[]
#     for bound in bounds:
#         return_data.append(bound[1])
#     return {"result": return_data}

def detect_meter(original_image:Image):
    input_size = (416, 416)
    input_image = original_image.resize(input_size)
    input_image.save("1_input_image.jpg")
    results=model_meter(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return input_image, detect_res


def extract_kWhr_num(input_image:Image, loc:tuple, detect_res:dict):
    croped_meter, detect_res= model_return_json(model_meter_component, input_image, loc, img1='2_croped_meter.jpg', img2='3_croped_meter2.jpg', img_size=(416,416))
    number_detect=False
    kwhr_detect=False
    ymin_kwhr, ymax_kwhr, xmin_kwhr, xmax_kwhr, kwhr_detect=get_xy(detect_res,'kwhr')
    ymin_num, ymax_num, xmin_num, xmax_num, number_detect=get_xy(detect_res,'meter_no')
    list_kwhr=[]
    #if kwhr_component exists
    if (kwhr_detect):
        loc=(xmin_kwhr, ymax_kwhr, xmax_kwhr, ymin_kwhr)
        _, detect_res= model_return_json(model_kwhr, croped_meter, loc, img1='4_croped_kwhr.jpg', img2='5_croped_kwhr2.jpg', img_size=(416,416))
        res={}
        for item in detect_res:
            res[item['xmin']]=item['name']
        list_xmin= sorted(res)
        for _xmin in list_xmin:
            list_kwhr.append(res[_xmin])
    list_no=[]
    if (number_detect):
        loc=(xmin_num, ymax_num, xmax_num, ymin_num)
        _, detect_res= model_return_json(model_number, croped_meter, loc, img1='6_croped_number.jpg', img2='7_croped_number2.jpg', img_size=(416,416))
        res={}
        for item in detect_res:
            res[item['xmin']]=item['name']
        list_xmin= sorted(res)
        for _xmin in list_xmin:
            list_no.append(res[_xmin])
    return list_kwhr, list_no
def extract_detail(image_path:str):
    bounds = reader.readtext('original_image.jpg')
    detail=[]
    for bound in bounds:
        detail.append(bound[1])
    return detail


@app.post("/extract_data_from_img")
async def extract_data_from_img_v1(file: bytes = File(...)):
    original_image = get_image_from_bytes(file)
    original_image.save("original_image.jpg")
    input_image, detect_res=detect_meter(original_image)
    meter_detect=False
    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')
    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
        list_kwhr,list_num= extract_kWhr_num(input_image, loc,detect_res)
        return {"result": {"kwhr":list_kwhr, "num":list_num}}

@app.post("/extract_data_from_img_v2")
async def extract_data_from_img_v2(file: bytes = File(...)):
    original_image = get_image_from_bytes(file)
    original_image.save("original_image.jpg")
    input_image, detect_res=detect_meter(original_image)
    meter_detect=False
    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')

    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
        list_kwhr,list_num= extract_kWhr_num(input_image, loc,detect_res)
        # version2
        detail = extract_detail('original_image.jpg')
        return {"result": {"kwhr":list_kwhr, "num":list_num, "detail":detail}}



@app.post("/extract_data_from_url")
async def extract_data_from_url_v1(url: str = File(...)):
    # detect meter component
    original_image = get_image_from_url(url)
    original_image.save("original_image.jpg")
    input_image, detect_res=detect_meter(original_image)
    meter_detect=False
    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')
    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
        list_kwhr, list_no=extract_kWhr_num(input_image, loc, detect_res)
        return {"result": {"kwhr":list_kwhr, "no":list_no}}

@app.post("/extract_data_from_url_v2")
async def extract_data_from_url_v2(url: str = File(...)):
    # detect meter component
    original_image = get_image_from_url(url)
    original_image.save("original_image.jpg")
    input_image, detect_res=detect_meter(original_image)
    meter_detect=False
    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')
    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
        list_kwhr, list_no=extract_kWhr_num(input_image, loc, detect_res)
        # version2
        detail = extract_detail('original_image.jpg')
        return {"result": {"kwhr":list_kwhr, "no":list_no, "detail":detail}}