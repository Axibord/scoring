import streamlit as st
from functions import *
import pandas as pd
import json, joblib

# load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.joblib',mmap_mode='r+')

@st.cache
def data_load():
    data = pd.read_csv('streamlit_data.csv')
    data['DAYS_BIRTH'] = data['DAYS_BIRTH']/365*(-1)
    return data

def main():
    
    # sidebare select page
    page = st.sidebar.selectbox("Choose a Page", ["Interprétation du score", "Faire une prédiction"])

    if page == 'Interprétation du score':
        # titre
        st.title("Interpretation du score d'un client")
        
        # sidebar options
        st.sidebar.subheader("Rechercher par ID :") # Feature 1
        number_ID = st.sidebar.text_input(" ",value=393130)
        st.sidebar.subheader("Les chances qu'un client rembourse son prêt :")# Feature 2
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
            st.write('**Genre**: {} \n \n **Revenue annuel**: {}$ \n \n**Montant du prêt**: {}$ \n'.format(
                    id_client_all['CODE_GENDER'].values, id_client_all['AMT_INCOME_TOTAL'].values, id_client_all['AMT_CREDIT'].values))
            st.write(" ** âge du client:** {} ans".format((id_client_all['DAYS_BIRTH'].values).round()))
            st.write(" ** Nombre d'enfants du client:** {}".format((id_client_all['CNT_CHILDREN'].values).round()))
            
        # Part 2     
        st.title("Score des clients selon l'intervalle selectionné")
        st.subheader("--> Il y'a** {} **clients qui ont entre** {}% et {}% **de chance de rembourser leur prêts ".format(len(score_data),choosen_range[0],choosen_range[1]))
        if st.checkbox('Afficher les données'):
            st.write(score_data)
            
        # Filters
        st.header("Filtres:")
        
        gender = st.selectbox("Genre", ('Homme + Femme','Homme', 'Femme')) # gender
        credit_amount = st.slider("Motant du crédit en $", 57330, 2931600, value=(57330, 2931600)) # credit_ammount
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
            "car_owner": own_car,
            "credit_amount": credit_amount
        }
        
        # update data
        data_updated = update_data(filters)
        
        # descriptive results after applying filters 
        st.header("Résultat du filtre:")
        st.subheader("- Nombre de clients:** %d **"% (len(data_updated)))
        st.subheader("- Score moyen:** {}% **de chances que ce groupe de clients remboursent leur prêts   ".format (100-(100*(data_updated['score'].mean())).round(2)))
        st.subheader("- âge moyen:** %d ans **"% (data_updated['DAYS_BIRTH'].mean()))
        st.subheader("- Nombre d'enfants moyen: ** %d **"% data_updated['CNT_CHILDREN'].mean())
        st.subheader("-** 80% **des clients demandent un prêt de** {}$ **ayant un revenue annuelle de** {}$ **".format((np.percentile(data_updated['AMT_CREDIT'], 80, axis=0)).round(2), (np.percentile(data_updated['AMT_INCOME_TOTAL'], 80, axis=0).round(2))))
        st.header("Top 5 de clients ayant le score le plus elevée en fonction des filtres:")
        st.write(top_5_id(data_updated)['SK_ID_CURR'])
        st.header("Visualisation des résultats du filtre:")
        scatter_plot(data_updated)
        
        
    elif page == 'Faire une prédiction':
        st.header("Prédiction au format JSON")
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        if uploaded_file is not None:
            data = json.load(uploaded_file)
            # convert data into dataframe
            data = pd.DataFrame(data)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.fillna(0,inplace=True)

            # scale data
            data_scaled = scaler.transform(data)

            # predictions
            result = model.predict(data_scaled)
            result = result.astype(float)

            # transform it to dict and send back to browser
            output = dict(enumerate(result))
            st.write(output)
    

       

if __name__ == "__main__":
    main()
