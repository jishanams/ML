
import streamlit as st
import pickle
import sklearn
st.header('Titanic Survival Prediction')
st.subheader('Predicting Survival on the titanic')
st.image("Titanic.jpeg")
st.text('''Titanic was a British ocean liner that sank in the early hours of 15 April 1912 as a result of striking an iceberg on her maiden voyage from Southampton, England, to New York City, United States. It was the second time White Star Line had lost a ship on her maiden voyage, the first being RMS Tayleur in 1854. Of the estimated 2,224 passengers and crew aboard, approximately 1,500 died (estimates vary), making the incident one of the deadliest peacetime sinkings of a single ship.''')
model=pickle.load(open('model.pkl','rb'))
l_sex=pickle.load(open('l_sex.pkl','rb'))
l_emb=pickle.load(open('l_emb.pkl','rb'))

#pclass = st.number_input("Passenger Class")
pclass=st.radio('select Passenger Class',(1,2,3))
sex = st.text_input("enter sex: [male,female]")
age = st.number_input("Age")
sibsp = st.number_input("Number of Siblings/Spouses Aboard")
parch = st.number_input("Number of Parents/Children Aboard")
fare = st.number_input("Fare")
embarked = st.text_input("Embarked: [S,C,Q]")
if st.button("Predict"):
    sex_l = l_sex.transform([sex])[0]
    embarked_l = l_emb.transform([embarked])[0]
    Predict = model.predict([[pclass, sex_l, age, sibsp, parch, fare, embarked_l]])[0]
    if Predict==1:
        st.success("Survived")
    else:
        st.warning("Did not survive")