import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la calificación ''')
st.image("Dexter.jpg", caption="Predicción de la calificación de una persona.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Horas = st.number_input('Horas estudiadas:', min_value=0.0, max_value=20.0, value=0.0, step=0.1)
  Descanso = st.number_input('Horas de sueño:', min_value=0.0, max_value=14.0, value=0.0, step=0.1)
  Asistencia = st.number_input('Porcentaje de asistencias:', min_value=0.0, max_value=100.0, value=0.0, step=0.01)
  Previa = st.number_input('Calificación previa:', min_value=0.0, max_value=100.0, value=0.0, step=0.01)  

  user_input_data = {'hours_studied': Horas,
                     'sleep_hours': Descanso,
                     'attendance_percent': Asistencia,
                     'previous_scores': Previa,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
f = pd.read_csv("student_exam_scores.csv")
f.drop(columns="student_id",inplace=True)
datos = f
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1615170)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Predicción de la calificación')
st.write('La posible calificación es ', prediccion)
