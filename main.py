import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# st.markdown(
#     """
#     <style>
#     .main {
#     backgroun-color:#F5F5F5;
#     }
#     </style>
#     unsafe_allow_html = True
#     """
# )

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data


header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Boston House Price Predictor ')
    st.text('In this project i look into the pricing of houses in Boston......')

with datasets:
    st.header('Boston House Price Dataset')
    st.text('I found this dataset in https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html')
    house_price = get_data('data/Boston.csv')
    st.write(house_price.head())
    st.subheader('Distribution of the Median Value of Owner Occupied Home in Boston Dataset')
    house_price_info = pd.DataFrame(house_price['medv'].value_counts()).head(35)
    st.bar_chart(house_price_info)

with features:
    st.header('The features that i created')
    st.markdown('* **First Feature :** I created this feature because of this... i calculated this feature using the following logic....!')
    st.markdown('* **Second Feature :** I created this feature because of this... i calculated this feature using the following logic....!')

with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the hyper-parameters of the model and how it changes the performance.')

    sel_cols, dis_col = st.columns(2)
    max_depth = sel_cols.slider('What should be the max. depth of the model?', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_cols.selectbox('How many trees should be there ?', options=[100,200,300,'No Limit'], index=0)
    sel_cols.text('Here is the list of Input features in my data.')
    sel_cols.table(house_price.columns)
    input_feature = sel_cols.text_input("Which feature should be used as the input feature?", 'medv')
    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    X = house_price[[input_feature]]
    y = house_price[['medv']]
    regr.fit(X,y)
    prediction = regr.predict(y)
    dis_col.subheader('Mean absolute error of the model is : ')
    dis_col.write(mean_absolute_error(y,prediction))
    dis_col.subheader('Mean squared error of the model is : ')
    dis_col.write(mean_squared_error(y, prediction))
    dis_col.subheader('R2 Score of the model is : ')
    dis_col.write(r2_score(y, prediction))





