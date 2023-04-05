#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Tuesday 28 Mar. 2023
# ==============================================================================


import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------- #

@st.cache
def anomaly_convert_df(df):
    return df.to_csv().encode('utf-8')


def anomaly_highlight(row):
    """Highlight a row in a dataframe according to a condition"""
    color = "#80003A" if row['anomaly'] == True else ''
    return [f'background-color:{color};'] * len(row)


def anomaly_detection(estimator: int, contamination: float, upload):
    """Detect anomalies into a dataframe and highlight it

    Parameters
    ----------
        estimator: int, required
        contamination: float, required
        upload: required
            CSV File format to analyze
    """

    df = pd.read_csv(upload)

    ## -- Get only column is not an Object and Replace NaN
    keep_col = [column for column in df if df[column].dtype != 'object']
    for col in df[keep_col]:
        imputer = SimpleImputer(strategy='median')
        df[[col]] = imputer.fit_transform(df[[col]])

    ## -- Apply the Isolation Forest Algorithm and count the total of anomalies
    model = IsolationForest(n_estimators=estimator, contamination=contamination,
                            max_samples='auto', random_state=42)

    model.fit(df[keep_col])
    df['anomaly'] = model.predict(df[keep_col]) == -1

    total_anomaly = 0
    total_anomaly += df['anomaly'].value_counts().get(-1, 0)
    st.subheader(f"We found {total_anomaly} Anomalies found in your CSV")

    ## -- Apply the Highlight, Delete the Anomaly Column and Display the DF
    ## -- With Column Anomaly
    df_styled = df.style.apply(anomaly_highlight, axis=1)

    with st.expander("Watch The Anomalies Find"):
        st.dataframe(df_styled, height=300)

    ## -- Download the CSV with the anomaly column
    csv = anomaly_convert_df(df)

    st.markdown("***")
    col1, col2, col3 = st.columns(3)

    col2.download_button(label="Download CSV with Anomaly Column",
                         data=csv,
                         file_name=f'anomaly_{upload.name}',
                         mime='text/csv',
    )

    ## -- Without column Anomaly but HTML Table
    # df_styled = df.style.apply(anomaly_highlight, axis=1)
    # df_styled_html = df_styled.hide("anomaly", axis=1).hide(axis=0).to_html()
    # st.write(df_styled_html, unsafe_allow_html=True)


################################################################################
################################# STREAMLIT ####################################

## -- METADATA WEB APP
st.set_page_config(page_title = "Traitement d'images",
                   page_icon = ":camera:",
                   layout = "wide"
)

## -- BACKGROUND
page_bg_img = f"""
  <style>
    .stApp {{
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
  </style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

## -- SIDEBAR
st.sidebar.write("## Upload and Configuration :gear:")
estimator = st.sidebar.slider('Set the estimator number', 1, 100)
contamination = st.sidebar.slider('Set the contamination rate', 0.1, 0.5)
uploaded_file = st.sidebar.file_uploader("Upload an CSV file", type=["csv"])

## -- MAIN CONTENT
if uploaded_file is not None:
    anomaly_detection(estimator, contamination, uploaded_file)
else:
    st.header("Detect anomalies in your data")
    st.write("Try uploading an CSV to watch if anomalies are present.")
    st.markdown("***")
    st.subheader("No File Uploaded Yet")
