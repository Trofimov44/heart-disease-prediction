import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

st.title("Heart disease prediction")
st.write("Предсказание вероятности сердечных заболеваний у пациента")

sex_list = {'М': 1, 'Ж': 0}
exang_fbs_list = {'Нет': 0, 'Да': 1}
cp_list = {"Типичная стенокардия": 0, "Атипичная стенокардия боль" : 1, "Неангинального характера" : 2, "Бессимптомная" : 3}
restecg_list = {"Норма" : 0,
                "Наличие аномалий зубца ST-T (инверсия зубца T и/или подъём или депрессия сегмента ST более чем на 0,05 мВ" : 1,
                "Вероятная или определённая гипертрофия левого желудочка по критериям Эстеса" : 2}
slope_list = {"восходящий" : 0,
            "горизонтальный" : 1,
            "нисходящий" : 2}
thal_list = {"Норма": 0, "Фиксированный дефект": 1, "Обратимый дефект" : 2}


age = st.number_input('age(Возраст)', min_value=0, value=56, max_value=120)
sex = st.selectbox("sex(Пол)", list(sex_list))
cp = st.selectbox("cp(Тип боли в груди)", list(cp_list))
trestbps = st.number_input("trestbps(Артериальное давление в покое, в мм рт. ст. при поступлении в больницу)", min_value=0, value=130, max_value=300)
chol = st.number_input("chol(Уровень холестерина в сыворотке крови в мг/дл)", min_value=0, value=243, max_value=1000)
fbs = st.selectbox("fbs(Уровень сахара в крови натощак > 120 мг/дл (1 = да; 0 = нет))", list(exang_fbs_list))
restecg = st.selectbox("restecg(Результаты электрокардиограммы в покое)", list(restecg_list))
thalach = st.number_input("thalach(Максимальная достигнутая частота сердечных сокращений)", min_value=0, value=153, max_value=250)   
exang = st.selectbox("exang(Стенокардия, вызванная физической нагрузкой)", list(exang_fbs_list))
oldpeak = st.number_input("oldpeak(Депрессия сегмента ST, вызванная физической нагрузкой относительно состояния покоя)", min_value=0., step=0.1, value=0.8)
slope = st.selectbox("slope(Наклон пикового сегмента ST при нагрузке)", list(slope_list))
ca = st.selectbox("ca(Количество крупных сосудов (0–3), визуализированных при флюороскопии)", [0, 1, 2, 3])
thal = st.selectbox("thal(Талассемия)", list(thal_list))


patient_data = {
    "age": age,      
    "sex": sex_list[sex],
    "cp": cp_list[cp],
    "trestbps": trestbps,   
    "chol": chol,
    "fbs": exang_fbs_list[fbs],
    "restecg": restecg_list[restecg],
    "thalach": thalach,
    "exang": exang_fbs_list[exang],
    "oldpeak": oldpeak,
    "slope": slope_list[slope],
    "ca": ca,
    "thal": thal_list[thal]
}

if st.button('predict(Предсказать)'):
    with st.spinner("Предсказание..."):
        try:
            response = requests.post(API_URL, json=patient_data, timeout=5)
        except requests.ConnectionError:
            st.error("Сервер FastAPI не запущен")
        except requests.Timeout:
            st.error("Сервер не отвечает")
        else:
            if response.status_code == 200:
                result = response.json()

                probability = result["probability"]
                verdict = result["result"]
            
                st.metric("Вероятность болезни", f"{probability:.1%}")
                if verdict == "Вероятно пациент здоров":
                    st.success(verdict)
                else:
                    st.warning(verdict)

                all_factors = result["risk_factors"] + result["protective_factors"]
                shap_df = pd.DataFrame(all_factors)
                # Рисуем бар чарт
                st.subheader("Вклад признаков в предсказание:")
                st.bar_chart(shap_df.set_index("feature")["impact"])

                with st.expander("Пояснение дополнительных параметров"):
                    st.markdown("""
                - **`hr_reserve_ratio`** — % от макс. пульса для возраста `thalach / (220 − age)`

                - **`oldpeak_is_zero`** — наличие любой депрессии ST `oldpeak == 0`

                - **`chol_per_age`** — накопление холестерина с возрастом `chol / age`

                - **`chol_log`** — нормализация скошенного распределения `log1p(chol)` """)

            elif response.status_code == 422:
                # Данные не прошли валидацию Pydantic
                st.error("Ошибка в данных: " + str(response.json()["detail"]))

            else:
                # Сервер упал или другая ошибка
                st.error(f"Ошибка сервера: {response.status_code}")
