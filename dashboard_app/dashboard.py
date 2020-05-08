import streamlit as st
from functions import *
import pandas as pd
import json, joblib


@st.cache
def data_load():
    data = pd.read_csv('streamlit_data.csv')
    data['DAYS_BIRTH'] = data['DAYS_BIRTH']/365*(-1)

    return data

def main():
    
    # sidebare select page
    page = st.sidebar.selectbox("Choose a Page", ["Score interpretation", "Make new predictions"])

    if page == 'Score interpretation':
        # titre
        st.title("Interpretation du score d'un client")
        
        # sidebar options
        st.sidebar.subheader("Rechercher par ID :") # Feature 1
        number_ID = st.sidebar.text_input(" ",value=393130)
        st.sidebar.subheader("Les chances qu'un client rembourse son pret :")# Feature 2
        choosen_range = st.sidebar.slider (' ',0.0, 100.0, value=(90.0, 99.0))
             
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
        if st.checkbox('Afficher les données'):
            st.write(score_data)
            
        
        st.subheader("Filtres:")
        gender = st.selectbox("Genre", ('Homme + Femme','Homme', 'Femme')) # gender
        ages = st.slider("âge", 21, 70, value=(25, 64)) # ages
        st.write(ages)
        nchilderns = st.slider("Nombre d'enfants",0, 14, value=(None)) # N of childerns
        own_house = st.checkbox("House owner") # house
        own_car = st.checkbox("Car owner") # car
        
        filters = {
            "score_range": choosen_range,
            "gender": gender,
            "age": ages,
            "number_childerns": nchilderns,
            "house_owner": own_house,
            "car_owner": own_car
        }
        
        filtred_data = scatter_plot_filters(filters)
        st.subheader("Résultats en fonction des filtres choisis:")
        st.subheader("- Nombre de clients: %d"% (len(filtred_data)))
        st.subheader("- Score moyen: {}% de chances que ces clients remboursent leur prêt   ".format (100-(100*(filtred_data['score'].mean())).round(2)))
        st.subheader("- âge moyen: %d ans "% (filtred_data['DAYS_BIRTH'].mean()))
  
    elif page == 'Make new predictions':
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        if uploaded_file is not None:
            data_predict = json.load(uploaded_file)
            st.write(data_predict)
    

       

if __name__ == "__main__":
    main()
