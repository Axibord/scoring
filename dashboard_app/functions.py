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
                    'DAYS_BIRTH','REGION_RATING_CLIENT','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN']].copy()
    score_slider = np.subtract((100),range_choosed)
    score_slider = np.divide(score_slider,(100))
    score_data = score_data[(score_data['score']<=score_slider[0]) & (score_data['score']>=score_slider[1])]
    score_data['score'] = 100 - (score_data['score']*100).round(2)
    score_data['score'] = score_data['score'].astype(int)
    score_data['score'] = score_data['score'].astype('str')
    score_data['score'] = score_data['score'] + ' %'
    return score_data

def process_data_client(number_ID):
    
    id_client = data[['SK_ID_CURR','score','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT',
                    'DAYS_BIRTH','REGION_RATING_CLIENT','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN']].copy()
    id_client['score'] = 100 - (id_client['score']*100).round(2)
    id_client['score'] = id_client['score'].astype(int)
    id_client['score'] = id_client['score'].astype('str')
    id_client['score'] = id_client['score'] + ' %'
    id_client = id_client[id_client['SK_ID_CURR']==number_ID]
    id_score = list(id_client['score'].values)
    id_client.rename(columns={'score':'Chance de remboursement'}, inplace=True)
    
    return id_client[['SK_ID_CURR','Chance de remboursement']],id_client, id_score


def top_5_id(dataframe):
    data = dataframe.sort_values(by='score')
    return data.head(5)
    
#------------------------------------------------------------------------------------------------    
def gender_update(gender_output, data, col_name):
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
def childrens_update(number_child, data, col_name):
    return data[(data[col_name]<=number_child)]

#------------------------------------------------------------------------------------------------    
def house_update(house_owner, data, col_name):
    if house_owner == True:
        return data[(data[col_name]== 1)]
    else:
        return data[(data[col_name]== 0)]
    
#------------------------------------------------------------------------------------------------    
def car_update(car_owner, data, col_name):
    if car_owner == True:
        return data[(data[col_name]== 1)]
    else:
        return data[(data[col_name]== 0)]

#------------------------------------------------------------------------------------------------   
def credit_update(credit_amount, data, col_name):
    return data[(data[col_name]<=credit_amount[1]) & (data[col_name]>=credit_amount[0])]
#------------------------------------------------------------------------------------------------    
def update_data(filters):
    """Parameters
    --------
    filters: dictionary of filters outputs to filter the dataframe 
    Return
    --------
    Return updated dataframe
    """
     
    score_data = data[['SK_ID_CURR','score','CODE_GENDER','AMT_INCOME_TOTAL','AMT_CREDIT',
                    'DAYS_BIRTH','REGION_RATING_CLIENT','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN']].copy()
    
    score_slider = np.subtract((100), filters['score_range'])
    score_slider = np.divide(score_slider,(100))
    score_data = score_data[(score_data['score']<=score_slider[0]) & (score_data['score']>=score_slider[1])]
    score_data = gender_update(filters['gender'], score_data, col_name='CODE_GENDER')
    score_data = age_update(filters['age'], score_data, col_name='DAYS_BIRTH')
    score_data = childrens_update(filters['number_childerns'], score_data, col_name='CNT_CHILDREN')
    score_data = house_update(filters['house_owner'], score_data, col_name='FLAG_OWN_REALTY')
    score_data = car_update(filters['car_owner'], score_data, col_name='FLAG_OWN_CAR')
    score_data = credit_update(filters['credit_amount'], score_data, col_name='AMT_CREDIT')
    
    plot_data = score_data.head(1000)
    return plot_data
    
def scatter_plot(data):
    """Parameters
    --------
    data: updated dataframe 
    Return
    --------
    Return scatter plot
    """
    st.vega_lite_chart(data, {
            'width': 'container',
            'height': 400,
            'mark':'circle',
            'encoding':{
                'x':{
                'field':'DAYS_BIRTH',
                'type': 'quantitative'
                },
                'y':{
                'field':'AMT_CREDIT',
                'type':'quantitative'
                },
                'size':{
                'field':'AMT_INCOME_TOTAL',
                'type':'quantitative'
                },
                'color':{
                'field':'REGION_RATING_CLIENT',
                'type':'nominal'}
                }
            }, use_container_width=True)
    
#------------------------------------------------------------------------------------------------     