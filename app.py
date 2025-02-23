import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from groq import Groq
from gtts import gTTS
from io import BytesIO

# Function to scrape the article
def scrape_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = "\\n".join([para.get_text() for para in paragraphs])
        return article_text
    else:
        return None

# Function to summarize and generate key points using Groq's LLaMA model
def summarize_article(article_text):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    message = {
        "role": "user",
        "content": f"Please summarize the following news article content, generate an outline, and highlight key points:\n\"{article_text}\""
    }

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # LLaMA 2-70B model
        messages=[message],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        stream=False
    )

    response = completion.choices[0].message.content
    return response

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# Streamlit UI
st.title("SnapNews: AI Insight Summaries")
st.subheader("Input a News Article URL")

# Step 1: Input URL
article_url = st.text_input("Enter the URL of a news article:")

if article_url:
    st.write("Scraping article content...")

    # Step 2: Scrape the article
    article_text = scrape_article(article_url)
    if article_text:
        st.write("Article content successfully scraped.")
        st.text_area("Extracted Article Text:", article_text, height=200)

        # Step 3: Summarize article
        st.write("Summarizing the article...")
        summary = summarize_article(article_text)

        # Display the summary
        st.subheader("Summary, Outline, and Key Points:")
        st.write(summary)

        # Step 4: Text to Speech
        st.write("Converting summary to speech...")
        audio_data = text_to_speech(summary)

        # Play the audio in Streamlit
        st.audio(audio_data, format='audio/mp3')

        # Option to download the summary audio
        st.download_button("Download Summary Audio", audio_data, file_name="summary.mp3")
    else:
        st.error("Failed to retrieve the article. Please check the URL.")
