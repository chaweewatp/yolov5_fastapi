from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes, get_meter_component, get_kwhr, get_number,get_meter
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware


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

    print(detect_res)
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
    for res in detect_res:
        if res['name']=='meter':
            ymin_meter=res['ymax']
            ymax_meter=res['ymin']
            xmin_meter=res['xmin']
            xmax_meter=res['xmax']
            meter_detect=True

    if (not meter_detect):
        return {"result": " no meter detected"}
    else:
        croped_meter = input_image.crop((xmin_meter, ymax_meter, xmax_meter, ymin_meter))
        croped_meter.save("2_croped_meter.jpg")
        meter_size=(416, 416)
        croped_meter = croped_meter.resize(meter_size)
        croped_meter.save("3_croped_meter2.jpg")

        results = model_meter_component(croped_meter, size=416)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        detect_res = json.loads(detect_res)
        number_detect=False
        kwhr_detect=False
        for res in detect_res:
            if res['name']=='kwhr':
                ymin_kwhr=res['ymax']
                ymax_kwhr=res['ymin']
                xmin_kwhr=res['xmin']
                xmax_kwhr=res['xmax']
                kwhr_detect=True
            elif res['name']=='meter_no':
                ymin_num=res['ymax']
                ymax_num=res['ymin']
                xmin_num=res['xmin']
                xmax_num=res['xmax']
                number_detect=True
        list_kwhr=[]
        #if kwhr_component exists
        if (kwhr_detect):
            # crop image
            croped_kwhr = croped_meter.crop((xmin_kwhr, ymax_kwhr, xmax_kwhr, ymin_kwhr))
            croped_kwhr.save("4_croped_kwhr.jpg")
            kwhr_size = (416, 416)
            croped_kwhr = croped_kwhr.resize(kwhr_size)
            croped_kwhr.save("5_croped_kwhr2.jpg")
            # get kwhr
            results = model_kwhr(croped_kwhr, size=416)
            detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
            detect_res = json.loads(detect_res)
            # re-arange
            res={}
            for item in detect_res:
                res[item['xmin']]=item['name']
            list_xmin= sorted(res)
            for _xmin in list_xmin:
                list_kwhr.append(res[_xmin])
        
        list_no=[]
        if (number_detect):
            # crop image
            croped_number = croped_meter.crop((xmin_num, ymax_num, xmax_num, ymin_num))
            croped_number.save("6_croped_number.jpg")
            # resize image to 416x416 here
            number_size = (416, 416)
            croped_number = croped_number.resize(number_size)
            croped_number.save("7_croped_number2.jpg")
            # get kwhr
            results = model_number(croped_number, size=416)
            detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
            detect_res = json.loads(detect_res)
            # re-arange
            res={}
            for item in detect_res:
                res[item['xmin']]=item['name']
            list_xmin= sorted(res)
            for _xmin in list_xmin:
                list_no.append(res[_xmin])
        return {"result": {"kwhr":list_kwhr, "no":list_no}}
