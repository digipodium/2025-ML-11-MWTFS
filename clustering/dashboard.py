import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Market Basket Analysis", layout='wide')

st.title("Market Basket Analysis")
st.markdown("This app performs Market Basket Analysis using the Apriori algorithm.")


upload_file = st.file_uploader("Upload Groceries_dataset.csv", type=["csv"])

if upload_file:
    #read the uploaded file
    df = pd.read_csv(upload_file)
    st.success('File uploaded successfully!')

    #Show a preview of the uploaded data
    st.subheader("Preview of the uploaded data")
    st.write(df.head())

    #Check if required columns are present in the dataset
    if {'Member_number','Date','itemDescription'}.issubset(df.columns):
        df_grouped = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()

        #Convert grouped data into a list of transactions
        transactions = df_grouped['itemDescription'].tolist()

        #Apply one-hot encoding to the transaction list
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)    

        st.success("Transactions prepared and encoded successfully!")

        #Add sidebar sliders to set parameters
        st.sidebar.header("Set Parameters")
        min_support = st.sidebar.slider("Minimum Support", 0.001,0.02,0.002, step=0.001)
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.1, step=0.05)

        #Apply apriori algorithm to find frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_)