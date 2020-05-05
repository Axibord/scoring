import json, joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache
def data_load():
    data = pd.read_csv('streamlit_data.csv')
    return data

def main():
    data = data_load()
    page = st.sidebar.selectbox("Choose a Page", ["Homepage","Exploration and analysis", "Make new predictions"])

    if page == 'Homepage':
        st.title('Interpretation du score pour chaque client')
        st.sidebar.subheader("Selectionnez les pourcentage de chance qu'un client rembourse son pret :")
        
        score_data = data[['SK_ID_CURR','score']].copy()
        score_slider1 = st.sidebar.slider (' ',0.0, 100.0, value=(25.0, 75.0),)
        
        score_slider = np.subtract((100),score_slider1)
        score_slider = np.divide(score_slider,(100))
        score_data = score_data[(score_data['score']<=score_slider[0]) & (score_data['score']>=score_slider[1])]
        score_data['score'] = 100 - (score_data['score']*100).round(2)
        score_data['score'] = score_data['score'].astype(int)
        score_data['score'] = score_data['score'].astype('str')
        score_data['score'] = score_data['score'] + ' %'
        
        
        st.sidebar.subheader("Rechercher par ID :")
        number_ID = st.sidebar.text_input(" ",value=393130)
        if number_ID is not None:
            number_ID = int(number_ID)
            id_client = data[['SK_ID_CURR','score']].copy()
            id_client['score'] = 100 - (id_client['score']*100).round(2)
            id_client['score'] = id_client['score'].astype(int)
            id_client['score'] = id_client['score'].astype('str')
            id_client['score'] = id_client['score'] + ' %'
            id_client = id_client[id_client['SK_ID_CURR']==number_ID]
            id_score = list(id_client['score'].values)
           
            st.table(id_client)
           
            st.subheader("Le client portant l'id: '{}' a {} de chance de rembourser son prêt".format(number_ID, id_score[0]))
        st.title("Score des clients selon l'intervalle selectionné")
        st.subheader("Il y'a {} clients qui ont entre {}% et {}% de chance de rembourser leur prêts ".format(len(score_data),score_slider1[0],score_slider1[1]))
        st.table(score_data)
    elif page == 'Exploration and analysis':
        map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])
        st.map(map_data)
    
    elif page == 'Make new predictions':
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        if uploaded_file is not None:
            data_predict = json.load(uploaded_file)
            st.write(data_predict)
    

       

if __name__ == "__main__":
    main()
