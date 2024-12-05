import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)


vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)


tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)


x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm not sure how to respond to that."

counter = 0

def main():
    global counter
    
    
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
   
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    .stTextArea > div > div > textarea {
        background-color: #e6f3e6;
        color: black;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #e6f2ff;
    }
    </style>
    """, unsafe_allow_html=True)

    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Chatbot Assistant</h1>", unsafe_allow_html=True)

    
    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.radio("Navigation", menu)

    
    if choice == "Chat":  
        
        counter += 1
        user_input = st.chat_input("Type your message here...", key=f"user_input_{counter}")

        
        chat_container = st.container()

        if user_input:
            
            user_input_str = str(user_input)

            
            response = chatbot(user_input)

           
            with chat_container:
                
                st.chat_message("user").write(user_input_str)
                
               
                with st.chat_message("assistant"):
                    st.write(response)

           
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.balloons()
                st.write("Thank you for chatting with me. Have a great day!")

    
    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        
        
        if not os.path.exists('chat_log.csv'):
            st.warning("No conversation history found.")
            return

        
        df = pd.read_csv('chat_log.csv', encoding='utf-8')
        
        
        st.sidebar.header("Filter History")
        date_filter = st.sidebar.date_input("Filter by Date")
        
        
        if date_filter:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df[df['Timestamp'].dt.date == date_filter]
        
        
        st.dataframe(df, use_container_width=True)

    
    elif choice == "About":
        st.header("🤖 About the AI Chatbot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("gh-ullas.jpg", caption="AI Chatbot")
        
        with col2:
            st.write("""
            ### Project Overview
            This AI Chatbot uses Natural Language Processing (NLP) and Machine Learning to understand and respond to user inputs.

            #### Key Technologies:
            - Natural Language Processing
            - Scikit-learn
            - Logistic Regression
            - Streamlit
            """)
        
        
        with st.expander("Technical Details"):
            st.write("""
            - Uses TF-IDF Vectorization
            - Trained on predefined intents
            - Supports multiple conversation topics
            """)
        
        with st.expander("Future Improvements"):
            st.write("""
            - Add more sophisticated NLP techniques
            - Implement deep learning models
            - Expand intent coverage
            """)

if __name__ == '__main__':
    main()