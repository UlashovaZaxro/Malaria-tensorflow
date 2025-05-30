import os
import io
import pickle 
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# --- Ilova sozlamalari ---
app = FastAPI(title="Rasm Klassifikatori")

# HTML shablonlari uchun
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- Modelni yuklash ---
MODEL_FILENAME = "malaria_model.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

model = None 

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"XATOLIK: Model fayli topilmadi: {MODEL_PATH}")
            print("Iltimos, MODEL_FILENAME o'zgaruvchisini to'g'ri sozlaganingizni tekshiring.")
            return

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f) 
        print(f"'{MODEL_FILENAME}' modeli muvaffaqiyatli yuklandi.")
        print(f"Yuklangan model turi: {type(model)}")
    except Exception as e:
        print(f"XATOLIK: Modelni yuklashda muammo yuz berdi: {e}")
        model = None 
def preprocess_image(image_bytes: bytes, target_size=(128, 128)):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB': 
            img = img.convert('RGB')

        img = img.resize(target_size)
        
        img_array = np.array(img) 

        img_array = img_array / 255.0

        if len(img_array.shape) == 3: 
            img_array = np.expand_dims(img_array, axis=0)

        print(f"Qayta ishlangan rasm shakli: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"Rasmni qayta ishlashda xatolik: {e}")
        return None

CLASS_NAMES = ["Parazitlanmagan", "Parazitlangan"]

# --- Marshrutlar (Routes) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Asosiy sahifani ko'rsatish
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "image_data": None, "error": None})

@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    prediction_text = None
    error_message = None
    uploaded_image_data_url = None

    if not model:
        error_message = "Model yuklanmagan. Server loglarini tekshiring va MODEL_FILENAME to'g'riligiga ishonch hosil qiling."
    elif not file.content_type.startswith("image/"):
        error_message = "Noto'g'ri fayl turi. Iltimos, rasm faylini (JPEG, PNG, va hokazo) yuklang."
    else:
        try:
            image_bytes = await file.read()

            # Yuklangan rasmni HTMLda ko'rsatish uchun base64 formatiga o'tkazish
            img_str_base64 = base64.b64encode(image_bytes).decode("utf-8")
            uploaded_image_data_url = f"data:{file.content_type};base64,{img_str_base64}"

            processed_image = preprocess_image(image_bytes)

            if processed_image is None:
                error_message = "Rasmni qayta ishlashda xatolik yuz berdi. Server loglarini tekshiring."
            else:
                # Bashorat qilish
                raw_prediction = model.predict(processed_image)
                print(f"Modeldan qaytgan xom natija: {raw_prediction}") # Tekshirish uchun muhim!

                # Bashoratni tahlil qilish logikasi
                if isinstance(raw_prediction, (list, np.ndarray)) and \
                    len(raw_prediction) == 1 and \
                    isinstance(raw_prediction[0], (list, np.ndarray)) and \
                    len(raw_prediction[0]) == 1 and \
                    isinstance(raw_prediction[0][0], (float, np.floating)):
                    
                    prob_positive_class = float(raw_prediction[0][0])
                    threshold = 0.5 # Ikkilik klassifikatsiya uchun standart chegara

                    if len(CLASS_NAMES) != 2:
                        error_message = (f"Model ikkilik natija ({prob_positive_class:.4f}) qaytardi, "
                                        f"ammo CLASS_NAMES ro'yxatida {len(CLASS_NAMES)} ta element bor. "
                                        f"Ro'yxatda 2 ta klass nomi bo'lishi kerak edi (masalan, ['Negativ', 'Pozitiv']). "
                                        f"Iltimos, CLASS_NAMES ni to'g'rilang.")
                        print(error_message) # Server logiga yozish
                    else:
                        if prob_positive_class >= threshold:
                            # Pozitiv klass
                            predicted_class_name = CLASS_NAMES[1] # Ikkinchi klass (indeks 1)
                            confidence = prob_positive_class * 100
                        else:
                            # Negativ klass
                            predicted_class_name = CLASS_NAMES[0] # Birinchi klass (indeks 0)
                            confidence = (1 - prob_positive_class) * 100
                        prediction_text = f"Bashorat: {predicted_class_name} (Ishonch: {confidence:.2f}%)"

                elif isinstance(raw_prediction, (list, np.ndarray)) and \
                    len(raw_prediction) == 1 and \
                    isinstance(raw_prediction[0], (list, np.ndarray)) and \
                    len(raw_prediction[0]) > 1:
                    probabilities = np.array(raw_prediction[0]) # np.argmax uchun numpy arrayga o'tkazamiz
                    predicted_class_index = np.argmax(probabilities)
                    
                    if 0 <= predicted_class_index < len(CLASS_NAMES):
                        predicted_class_name = CLASS_NAMES[predicted_class_index]
                        confidence = probabilities[predicted_class_index] * 100
                        prediction_text = f"Bashorat: {predicted_class_name} (Ishonch: {confidence:.2f}%)"
                    else:
                        error_message = (f"Bashorat qilingan klass indeksi ({predicted_class_index}) "
                                        f"CLASS_NAMES ro'yxati (uzunligi {len(CLASS_NAMES)}) chegarasidan tashqarida. "
                                        f"Iltimos, CLASS_NAMES ni to'g'rilang.")
                        print(error_message) # Server logiga yozish
                
                else:
                    try:
                        raw_pred_str = str(raw_prediction)
                    except Exception:
                        raw_pred_str = "Noma'lum (string formatiga o'tkazib bo'lmadi)"
                    error_message = f"Noma'lum yoki qo'llab-quvvatlanmaydigan bashorat formati: {raw_pred_str}. Server loglarini tekshiring."
                    print(f"Unsupported prediction format from model: {raw_pred_str}") # Server logiga yozish

        except AttributeError as e_attr:
            error_message = (f"Modelda bashorat qilishda muammo ({e_attr}). "
                            f"Modelning '.predict()' metodi mavjudligini yoki kirish ma'lumotlari shakli to'g'riligini tekshiring.")
            print(f"AttributeError during prediction: {e_attr}") # Server logiga yozish
        except Exception as e_generic:
            error_message = "Bashorat qilishda kutilmagan xatolik yuz berdi. Batafsil ma'lumot uchun server loglarini tekshiring."
            print(f"Generic exception during prediction: {e_generic}") # Server logiga yozish
            import traceback
            print(traceback.format_exc()) # To'liq xatolikni server logiga yozish

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction_text,
        "error": error_message,
        "image_data": uploaded_image_data_url
    })

if __name__ == "__main__":
    import uvicorn
    print("FastAPI ilovasini ishga tushirish uchun terminalda quyidagi buyruqni bajaring:")
    print("uvicorn main:app --reload")