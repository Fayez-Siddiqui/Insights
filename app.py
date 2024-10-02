import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import traceback
from PIL import Image
import datetime
import time
from dotenv import load_dotenv
import os
import base64
from agents.openai_agent import PandasAgentOpenAI
import utils
import logging


# Configure logging
logging.basicConfig(filename='logs/csv_agent.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

data = "data"
plot = 'plot'
#os.makedirs(data, exist_ok=True)
#os.makedirs(plot, exist_ok=True)



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title='CSV Agent',  layout='wide')

#def data_uploader():
#    st.title("Data")
#    st.subheader("Upload CSV file for analysis")
#    # Upload csv file
#    uploaded_file = st.file_uploader("Choose csv file", type=["csv"])
#    if uploaded_file is not None:
#        utils.save_uploaded_file(uploaded_file)
#    else:
#        utils.remove_existing_files(data)
#        utils.remove_existing_files(plot)
#def file_checker():
#    file = []
#    for filename in os.listdir(data):
#        file_path = os.path.join(data, filename)
 #       file.append(file_path)

#    return file




st.title("Insight.AI")
st.write("**Chat with data**")

#selection = st.sidebar.radio("Go to", ["Data", "Analysis"])

#if selection == "Data":
#    data_uploader()
#elif selection == "Analysis":
#    sidebar_option = st.sidebar.selectbox(
#        "Select Model",
#        ("gpt-4o","gpt-4"))

temperature = st.sidebar.slider('Select Temperature', min_value=0.0, max_value=1.0, value =0.1, step = 0.1)

csv_agent = PandasAgentOpenAI(api_key=openai_api_key)
# Persistent state for storing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if "query" in content:
            st.markdown(content["query"])
        if "result_value" in content:
            st.markdown(content["result_value"])
        if "list" in content:
            st.markdown(content["list"])
        if "dict" in content:
            st.markdown(content["dict"])
        if "pandas_series" in content:
            st.dataframe(content["pandas_series"])
        if "printed_output" in content:
            st.markdown(content["printed_output"])
        if "error" in content:
            st.markdown(content["error"])
        if "dataframe" in content:
            st.dataframe(content["dataframe"], use_container_width=False)
        if "plot_data" in content:
            plot_data = content.get("plot_data", None)
            if plot_data is not None:
                plot_data = plot_data[0]
                image_data = base64.b64decode(plot_data)
                image = Image.open(io.BytesIO(image_data))

                st.image(image)



# Accept user input
if prompt := st.chat_input("Enter your question"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": {"query":prompt}})

if prompt:
    reply = csv_agent.run_agent(prompt)
    python_code = csv_agent.current_code
    logging.info(f"query: {prompt}; python_code_generated: {python_code}")
    #time.sleep(2)
    #print(reply)
    if "result_value" in reply:
        st.write(reply["result_value"])
    if "list" in reply:
        st.markdown(reply["list"])
    if "dict" in reply:
        st.markdown(reply["dict"])
    if "pandas_series" in reply:
        st.dataframe(reply["pandas_series"])
    if "printed_output" in reply:
        st.markdown(reply["printed_output"])
    if "error" in reply:
        st.markdown(reply["error"])
    if "plot_data" in reply:
        # Convert base64 string back to binary image data
        plot_data = reply.get("plot_data", None)
        if plot_data is not None:
            plot_data = plot_data[0]
            image_data = base64.b64decode(plot_data)
            image = Image.open(io.BytesIO(image_data))

            st.image(image)
    if "dataframe" in reply:
        st.dataframe(reply["dataframe"], use_container_width=False)

    if python_code != "":
         show_code = st.checkbox('Show Python Code')
         if show_code:
             st.markdown(python_code)

    st.session_state.messages.append({"role": "assistant", "content": reply})