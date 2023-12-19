from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

# Load YOLOv3 model
net = cv.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def get_objects(image_content):
    try:
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = classes[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            if label.lower() in ["truck", "bus", "bicycle", "train", "car"]:
                objects.append({'label': label, 'confidence': confidence, 'box': {'x': x, 'y': y, 'w': w, 'h': h}})

    return objects

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image_content = await file.read()
        hazardous_objects = get_objects(image_content)

        if any(hazardous_objects):
            print("Vehicle or a person approaching")

        return JSONResponse(content={"hazardous_objects": hazardous_objects})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
