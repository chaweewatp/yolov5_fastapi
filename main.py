from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes, get_meter_component, get_kwhr, get_number
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware


model = get_yolov5()
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

