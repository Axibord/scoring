from dashboard import data_load
import numpy as np
import json, joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import altair as alt

#------------------------------------------------------------------------------------------------    
data = data_load()
#------------------------------------------------------------------------------------------------    
def process_data_all(range_choosed):
    """
    Return
    --------
    return dataframe with specific format to help display
    """
    
    score_data = data[['SK_ID_CURR','score','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT',
                    'DAYS_BIRTH','REGION_RATING_CLIENT']].copy()
    score_slider = np.subtract((100),range_choosed)
    score_slider = np.divide(score_slider,(100))
    score_data = score_data[(score_data['score']<=score_slider[0]) & (score_data['score']>=score_slider[1])]
    score_data['score'] = 100 - (score_data['score']*100).round(2)
    score_data['score'] = score_data['score'].astype(int)
    score_data['score'] = score_data['score'].astype('str')
    score_data['score'] = score_data['score'] + ' %'
    return score_data


#------------------------------------------------------------------------------------------------    
def gender_update(gender_output, data, col_name):
    if len(gender_output) == 0:
        data_to_render = data
        st.write('les  2')
    if gender_output == 'Homme + Femme':
        data_to_render = data
    elif gender_output == 'Homme':
        data_to_render = data[data[col_name]==0]
    else:
        data_to_render = data[data[col_name]==1]
        
    return data_to_render

#------------------------------------------------------------------------------------------------    
def age_update(age_output, data, col_name):
    return data[(data[col_name]<=age_output[1]) & (data[col_name]>=age_output[0])]
#------------------------------------------------------------------------------------------------        
def childrens_update():
    pass
#------------------------------------------------------------------------------------------------    
def house_update():
    pass
#------------------------------------------------------------------------------------------------    
def car_update():
    pass
#------------------------------------------------------------------------------------------------    
def scatter_plot_filters(filters):
    """Parameters
    --------
    filters: dictionary of filters outputs from dashboard to filter the data 
    Return
    --------
    Return scatter plot with filters applied
     """
     
    
    
    score_data = data[['SK_ID_CURR','score','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT',
                    'DAYS_BIRTH','REGION_RATING_CLIENT','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN']].copy()
    
    score_slider = np.subtract((100), filters['score_range'])
    score_slider = np.divide(score_slider,(100))
    score_data = score_data[(score_data['score']<=score_slider[0]) & (score_data['score']>=score_slider[1])]
    score_data = gender_update(filters['gender'], score_data, col_name='CODE_GENDER')
    score_data = age_update(filters['age'], score_data, col_name='DAYS_BIRTH')
  
    
    plot_data = score_data.head(1000)
    
    st.vega_lite_chart(plot_data, {
            'width': 'container',
            'height': 400,
            'mark':'circle',
            'encoding':{
                'x':{
                'field':'DAYS_BIRTH',
                'type': 'quantitative'
                },
                'y':{
                'field':'score',
                'type':'quantitative'
                },
                'size':{
                'field':'AMT_CREDIT',
                'type':'quantitative'
                },
                'color':{
                'field':'REGION_RATING_CLIENT',
                'type':'nominal'}
                }
            }, use_container_width=True)
    return score_data
    
    
    
#------------------------------------------------------------------------------------------------     
    
def process_data_client(number_ID):
    
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