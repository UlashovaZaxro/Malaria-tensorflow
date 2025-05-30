import pickle
import tensorflow as tf

# .pkl faylidan modelni yuklash
try:
    with open("malaria_model.pkl", "rb") as f:
        loaded_model_from_pkl = pickle.load(f)
except Exception as e:
    print(f".pkl faylini o'qishda xatolik: {e}")
    print("Model .pkl fayli Keras modeli bo'lmasligi mumkin yoki fayl buzilgan.")
    exit()

if isinstance(loaded_model_from_pkl, tf.keras.Model):
    print("Model .pkl faylidan muvaffaqiyatli yuklandi.")

    # Modelni .h5 formatida saqlash
    try:
        loaded_model_from_pkl.save("malaria_model.h5")
        print("Model muvaffaqiyatli 'malaria_model.h5' fayliga saqlandi.")
    except Exception as e:
        print(f".h5 formatida saqlashda xatolik: {e}")

else:
    print("Yuklangan obyekt Keras modeli emas.")
    print(f"Yuklangan obyekt turi: {type(loaded_model_from_pkl)}")