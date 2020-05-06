import json, joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache
def data_load():
    data = pd.read_csv('streamlit_data.csv')
    return data


def process_data_all(range_choosed):
        data = data_load()
        score_data = data[['SK_ID_CURR','score']].copy()
        score_slider = np.subtract((100),range_choosed)
        score_slider = np.divide(score_slider,(100))
        score_data = score_data[(score_data['score']<=score_slider[0]) & (score_data['score']>=score_slider[1])]
        score_data['score'] = 100 - (score_data['score']*100).round(2)
        score_data['score'] = score_data['score'].astype(int)
        score_data['score'] = score_data['score'].astype('str')
        score_data['score'] = score_data['score'] + ' %'
        return score_data
    
    
def process_data_client(number_ID):
    data = data_load()
    id_client = data[['SK_ID_CURR','score','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT',
                      'DAYS_BIRTH','REGION_RATING_CLIENT']].copy()
    id_client['score'] = 100 - (id_client['score']*100).round(2)
    id_client['score'] = id_client['score'].astype(int)
    id_client['score'] = id_client['score'].astype('str')
    id_client['score'] = id_client['score'] + ' %'
    id_client = id_client[id_client['SK_ID_CURR']==number_ID]
    id_score = list(id_client['score'].values)
    
    id_client.rename(columns={'score':'Chance de remboursement'}, inplace=True)
    return id_client[['SK_ID_CURR','Chance de remboursement']],id_client, id_score

def main():
    
    page = st.sidebar.selectbox("Choose a Page", ["Score interpretation", "Informations about clients", "Make new predictions"])

    if page == 'Score interpretation':
        
        st.title("Interpretation du score d'un client")
        
        st.sidebar.subheader("Rechercher par ID :") # Feature 1
        number_ID = st.sidebar.text_input(" ",value=393130)
        
        st.sidebar.subheader("Les chances qu'un client rembourse son pret :")# Feature 2
        choosen_range = st.sidebar.slider (' ',0.0, 100.0, value=(25.0, 75.0))
        
        st.sidebar.subheader("informations descriptives :") # Feature 3
        ages = st.sidebar.slider("Tranche d'âge", 0, 100, value=(25, 75)) # ages
        gender = st.sidebar.multiselect("Sexe", ('Homme', 'Femme')) # gender
        
        score_data = process_data_all(choosen_range)
        
        if number_ID is not None:
            number_ID = int(number_ID)
            id_client_score, id_client_all, id_score = process_data_client(number_ID)
            st.table(id_client_score)
           
            st.subheader("--> Le client portant l'id: '{}' a** {} **de chance de rembourser son prêt".format(number_ID, id_score[0]))
            st.subheader('**Informations descriptives du client:**')
            id_client_all['CODE_GENDER'].replace({0:'Homme',1:'Femme'},inplace=True)
            st.write('**Sexe**: {} \n \n **Revenue total**: {}$ \n \n**Montant du prêt**: {}$ \n'.format(
                    id_client_all['CODE_GENDER'].values, id_client_all['AMT_INCOME_TOTAL'].values, id_client_all['AMT_CREDIT'].values
            ))
        st.title("Score des clients selon l'intervalle selectionné")
        st.subheader("--> Il y'a** {} **clients qui ont entre** {}% et {}% **de chance de rembourser leur prêts ".format(len(score_data),choosen_range[0],choosen_range[1]))
        st.table(score_data)
   
    elif page == 'Informations about clients':
        plt.hist(data['CODE_GENDER'])
        
        st.pyplot()

        
    
    elif page == 'Make new predictions':
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        if uploaded_file is not None:
            data_predict = json.load(uploaded_file)
            st.write(data_predict)
    

       

if __name__ == "__main__":
    main()
