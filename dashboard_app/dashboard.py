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
       
        #  part 1
        if number_ID is not None:
            number_ID = int(number_ID)
           
            id_client_score, id_client_all, id_score = process_data_client(number_ID)
            st.table(id_client_score)
           
            st.subheader("--> Le client portant l'id: '{}' a** {} **de chance de rembourser son prêt".format(number_ID, id_score[0]))
            st.subheader('**Informations descriptives du client:**')
            id_client_all['CODE_GENDER'].replace({0:'Homme',1:'Femme'},inplace=True)
            st.write('**Genre**: {} \n \n **Revenue total**: {}$ \n \n**Montant du prêt**: {}$ \n'.format(
                    id_client_all['CODE_GENDER'].values, id_client_all['AMT_INCOME_TOTAL'].values, id_client_all['AMT_CREDIT'].values
            ))
            
        # Part 2     
        st.title("Score des clients selon l'intervalle selectionné")
        st.subheader("--> Il y'a** {} **clients qui ont entre** {}% et {}% **de chance de rembourser leur prêts ".format(len(score_data),choosen_range[0],choosen_range[1]))
        if st.checkbox('Afficher les données'):
            st.write(score_data)
            
        # Filters
        st.subheader("Filtres:")
        
        gender = st.selectbox("Genre", ('Homme + Femme','Homme', 'Femme')) # gender
        ages = st.slider("âge", 21, 70, value=(25, 64)) # ages
        nchilderns = st.slider("Nombre d'enfants",0, 14) # N of childerns
        own_house = st.checkbox("House owner") # house
        own_car = st.checkbox("Car owner") # car
        
        # dictionary of filters outputs to send to function to update data
        filters = {
            "score_range": choosen_range,
            "gender": gender,
            "age": ages,
            "number_childerns": nchilderns,
            "house_owner": own_house,
            "car_owner": own_car
        }
        
        # update data
        data_updated = update_data(filters)
        
        
        # descriptive results after applying filters 
        st.subheader("Résultat du filtre:")
        st.subheader("- Nombre de clients: %d"% (len(data_updated)))
        st.subheader("- Score moyen: {}% de chances que ce groupe de clients remboursent leur prêts   ".format (100-(100*(data_updated['score'].mean())).round(2)))
        st.subheader("- âge moyen: %d ans "% (data_updated['DAYS_BIRTH'].mean()))
        scatter_plot(data_updated)
        
        
    elif page == 'Make new predictions':
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        if uploaded_file is not None:
            data_predict = json.load(uploaded_file)
            st.write(data_predict)
    

       

if __name__ == "__main__":
    main()
