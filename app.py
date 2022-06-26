import cv2
import numpy as np
from model import FacialExpressionModel
from keras.models import load_model
import streamlit as st
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import io
from recognation import VideoCamera

model = FacialExpressionModel()

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

all_emo = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class VideoTransformer(VideoTransformerBase):
    def __init__(self, vid):
        self.video = cv2.VideoCapture(vid)
    
    def transform(self, frame):
        img = self.video.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            finalout = model.predict_emotion(roi_gray[np.newaxis, :, :, np.newaxis])
            output = str(finalout)    
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y),(x + w, y + h),(0,255,0),2)
        return img



def main():
    st.title("ПП «Всевидящее око👁‍🗨»")
    activiteis = ["Главная", "Распознование эмоций", "Тест",  "О нас"]
    choice = st.sidebar.selectbox("Выберите окно", activiteis)
    st.sidebar.markdown(
        """ Developed by human for humans || GranIT❤   
            Email : granitwithlove@gmail.com""")
    if choice == "Главная":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Каскад Хаара, tenserflow, streamlit, OpenCV, сверточная нейронная сеть </h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
    elif choice == "Тест":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Загрузите данные для распознования.</h4>
                                            </div>
                                            </br>"""
        vid = st.file_uploader("Выберите файл...")        
        if vid is not None:
            webrtc_streamer(key="example", video_transformer_factory=VideoTransformer('src/4.mp4'))    
        st.markdown(html_temp_home1, unsafe_allow_html=True)
    elif choice == "Распознование эмоций":
        st.header("Распознование эмоций с камеры")
        st.write("Нажмите на кнопку Start, чтобы запустить веб-камеру и определить эмоции вашего лица.")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice == "О нас":
        st.subheader("О приложении")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    ПП «Всевидящее око👁‍🗨», предназначенное для распознование эмоций школьников в режиме реального времени с камер видеонаблюдения, был разработан на основе</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">Разработано командой ГранIT❤</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    
    else:
        pass


if __name__ == "__main__":
    main()

