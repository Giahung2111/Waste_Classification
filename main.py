from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2
import time

app = FastAPI()

# Tải mô hình
model = load_model('C:\\Python\\Garbage_Classification\\NEU-Bin\\model_5class_resnet_87percent\\model_5class_resnet_87%.h5')
labels = ['G&M', 'Organic', 'Other', 'Paper', 'Plastic']

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Điều chỉnh kích thước ảnh
    img = imagenet_utils.preprocess_input(img)  # Tiền xử lý ảnh theo chuẩn ImageNet
    img = img_to_array(img)  # Chuyển đổi ảnh thành mảng
    img = np.expand_dims(img / 255, 0)  # Chuẩn hóa ảnh và thêm chiều batch
    return img

# Hàm chuyển đổi giá trị thành chuỗi
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc hình ảnh từ tệp
    image = await file.read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Tiền xử lý hình ảnh
    img1 = preprocess_image(img)

    # Dự đoán
    start_time = time.time()  # Thời gian bắt đầu dự đoán
    p = model.predict(img1)  # Dự đoán lớp ảnh
    confidences = max(np.squeeze(p))  # Xác suất cao nhất
    conf = round(float(confidences), 3)  # Chuyển đổi xác suất thành float chuẩn
    predicted_class = labels[np.argmax(p[0])]  # Lớp dự đoán có xác suất cao nhất

    # In FPS (frames per second)
    fps = round(float(1.0 / (time.time() - start_time)), 2)  # Chuyển đổi FPS thành float chuẩn

    # Trả về kết quả dự đoán cùng độ tin cậy và FPS
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": conf,
        "fps": fps
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
