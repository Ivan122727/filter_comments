import joblib
from utils import tokenize_sentence

# Загрузка модели из файла
model_pipeline = joblib.load("./pipeline.joblib")

# Использование модели для предсказания
print(model_pipeline.predict(["Ой ! Не делай минет беременную голову!"]))
print(model_pipeline.predict(["то не обязаловка, а добровольно. Да и удовольствие получать не запрещается."]))