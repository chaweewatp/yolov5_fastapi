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

model = get_yolov5()
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


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    # results = model(input_image)
    results = model(input_image, size=416)
    results.print()  
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    # results = model(input_image)
    results = model(input_image, size=416)
    results.print()  

    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")




@app.post("/object-to-json2")
async def detect_meter_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    # results = model(input_image)
    results = model(input_image, size=416)
    results.print()
    results.crop()
    
    detect_res="test"
    return {"result": detect_res}

@app.post("/extract_meter_component")
async def detect_meter_component_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model_meter_component(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)

    return {"result": detect_res}


@app.post("/extract_kwhr")
async def detect_kwhr_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model_kwhr(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    
    return {"result": detect_res}


@app.post("/extract_meter")
async def detect_meter_kwhr_return_json_result(file: bytes = File(...)):
    # detect meter component
    input_image = get_image_from_bytes(file)
    
    newsize = (416, 416)
    input_image = input_image.resize(newsize)
    input_image.save("input.jpg")

    results = model_meter_component(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    kwhr_detected=False
    for res in detect_res:
        if res['name']=='kwhr':
            ymin_kwhr=res['ymax']
            ymax_kwhr=res['ymin']
            xmin_kwhr=res['xmin']
            xmax_kwhr=res['xmax']
            kwhr_detected=True
        elif res['name']=='meter_no':
            ymin_no=res['ymax']
            ymax_no=res['ymin']
            xmin_no=res['xmin']
            xmax_no=res['xmax']
            no_detected=True
            
    list_kwhr=[]
    #if kwhr_component exists
    if (kwhr_detected):
        # crop image
        print("croping image")

        croped_image = input_image.crop((xmin_kwhr, ymax_kwhr, xmax_kwhr, ymin_kwhr))
        croped_image.save("croped_kwhr.jpg")

        print("image croped")
        # resize image to 416x416 here
        newsize = (416, 416)
        croped_image = croped_image.resize(newsize)
        croped_image.save("croped_number2.jpg")

        # get kwhr
        results = model_kwhr(croped_image, size=416)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        detect_res = json.loads(detect_res)
        print(detect_res)
        # re-arange
        res={}
        for item in detect_res:
            res[item['xmin']]=item['name']
        list_xmin= sorted(res)
        for _xmin in list_xmin:
            list_kwhr.append(res[_xmin])
    
    list_no=[]
    if (no_detected):
        # crop image
        print("croping image")
        croped_image = input_image.crop((xmin_no, ymax_no, xmax_no, ymin_no))
        croped_image.save("croped_number.jpg")

        print("image croped")
        # resize image to 416x416 here
        newsize = (416, 416)
        croped_image = croped_image.resize(newsize)
        croped_image.save("croped_number2.jpg")

        # get kwhr
        results = model_number(croped_image, size=416)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        detect_res = json.loads(detect_res)
        print(detect_res)
        # re-arange
        res={}
        for item in detect_res:
            res[item['xmin']]=item['name']
        list_xmin= sorted(res)
        for _xmin in list_xmin:
            list_no.append(res[_xmin])

        return {"result": {"kwhr":list_kwhr, "no":list_no}}
    else:
        return {"result": " no kwhr data"}

def get_xy(dict_res, class_name):
    print(dict_res)
    print(class_name)
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
    

@app.post("/extract_data")
async def detect_meter_data_return_json_result(file: bytes = File(...)):
    # detect meter component
    input_image = get_image_from_bytes(file)


    input_size = (416, 416)
    input_image = input_image.resize(input_size)
    input_image.save("1_input_image.jpg")
    results=model_meter(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    meter_detect=False

    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')


    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
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
        return {"result": {"kwhr":list_kwhr, "no":list_no}}


@app.post("/extract_data_from_url")
async def detect_meter_data_return_json_result(url: str = File(...)):
    # detect meter component
    # url='http://94.74.115.223/uploads/electric-meter/scaled_77ecefbb-4d77-41fc-9f5a-970b1b5a79ec3625275333049432296-c34c2d3f-8aec-4935-bd0b-0c17a42a45ef.jpg'
    input_image = get_image_from_url(url)

    # prefix_path="http://94.74.115.223/"
    # im = Image.open(requests.get(prefix_path+"uploads/electric-meter/scaled_f8b38b8a-a0a7-4cb8-b598-7d11d5745c1c413712555478209381-bfba8e8a-233b-41cf-ae8c-0458219bbf5b.jpg", stream=True).raw)

    
    input_size = (416, 416)
    input_image = input_image.resize(input_size)
    input_image.save("1_input_image.jpg")
    results=model_meter(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    meter_detect=False
    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')
    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
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
        return {"result": {"kwhr":list_kwhr, "no":list_no}}



@app.post("/extract_easyocr")
async def detect_meter_infomation_return_json_result(url: str = File(...)):
    # bounds = reader.readtext('1_input_image.jpg')
    bounds = reader.readtext('download (2).png')

    return_data=[]
    for bound in bounds:
        return_data.append(bound[1])
    return {"result": return_data}



@app.post("/extract_data_from_url_v2")
async def detect_meter_data_return_json_result_v2(url: str = File(...)):
    # detect meter component
    # url='http://94.74.115.223/uploads/electric-meter/scaled_77ecefbb-4d77-41fc-9f5a-970b1b5a79ec3625275333049432296-c34c2d3f-8aec-4935-bd0b-0c17a42a45ef.jpg'
    original_image = get_image_from_url(url)
    original_image.save("original_image.jpg")
    # prefix_path="http://94.74.115.223/"
    # im = Image.open(requests.get(prefix_path+"uploads/electric-meter/scaled_f8b38b8a-a0a7-4cb8-b598-7d11d5745c1c413712555478209381-bfba8e8a-233b-41cf-ae8c-0458219bbf5b.jpg", stream=True).raw)

    
    input_size = (416, 416)
    input_image = original_image.resize(input_size)
    input_image.save("1_input_image.jpg")
    results=model_meter(input_image, size=416)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    meter_detect=False
    ymin_meter, ymax_meter, xmin_meter, xmax_meter, meter_detect=get_xy(detect_res,'meter')
    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        loc=(xmin_meter, ymax_meter, xmax_meter, ymin_meter)
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
        
        bounds = reader.readtext('original_image.jpg')
        detail=[]
        for bound in bounds:
            detail.append(bound[1])
        return {"result": {"kwhr":list_kwhr, "no":list_no, "detail":detail}}