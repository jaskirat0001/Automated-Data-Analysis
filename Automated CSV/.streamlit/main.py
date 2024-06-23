from dotenv import load_dotenv
import os
import numpy as np
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib

 
matplotlib.use('TkAgg')


load_dotenv()
# key = "sk-proj-LoLlOWHlLsA4aGN6O8qLT3BlbkFJdZLq2l8SP4PtQryNCJHJ"
# API_KEY = os.environ[key]
# API_KEY = os.environ("sk-proj-LoLlOWHlLsA4aGN6O8qLT3BlbkFJdZLq2l8SP4PtQryNCJHJ")



os.environ["PANDASAI_API_KEY"] = "$2a$10$aGuqVaNoV7Gi4ytGCapOA.xsTnp2HdFc6aYuGLPpLZJOvcXpGcmla"

# llm = OpenAI(api_token = API_KEY)
# pandas_ai = PandasAI(llm)
llm  = OpenAI(api_token = "sk-proj-LoLlOWHlLsA4aGN6O8qLT3BlbkFJdZLq2l8SP4PtQryNCJHJ")


st.title("Prompt-driven ANalysis with pandasai App")

uploaded_file = st.file_uploader("Upload a csv file for analysis", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    pandas_ai = SmartDataframe(df)
    # st.write(df)
    st.write(df.head())
    
    prompt = st.text_input("Enter a prompt")
    if st.button("Generate"):
        if prompt:
            st.write("PandasAI is generating a response...")
            st.write(pandas_ai.chat(prompt))
        else:
            st.warning("Please enter a prompt")


