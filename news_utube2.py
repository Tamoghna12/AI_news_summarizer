import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from typing import List, Dict, Any
import pandas as pd

client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# Function to get YouTube videos based on search criteria
def get_videos(query: str, max_results: int, api_key: str) -> List[Dict[str, Any]]:
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=max_results,
            order="viewCount"
        ).execute()

        videos = []
        for item in search_response['items']:
            if item['snippet']['liveBroadcastContent'] != 'live':  # Filter out live broadcasts
                video_id = item['id']['videoId']

                # Fetch video statistics
                video_response = youtube.videos().list(
                    part="statistics",
                    id=video_id
                ).execute()

                if video_response['items']:
                    statistics = video_response['items'][0]['statistics']
                    view_count = int(statistics.get('viewCount', 0))

                    if view_count > 10000:
                        videos.append({
                            'title': item['snippet']['title'],
                            'video_id': video_id,
                            'view_count': view_count,
                            'thumbnail': item['snippet']['thumbnails']['default']['url']
                        })

        return videos
    except HttpError as e:
        st.error(f"An error occurred: {e}")
        return []

# Function to extract video transcript
def get_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        st.error(f"Error getting transcript for video {video_id}: {str(e)}")
        return ""

def summarize_text(text: str, model: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                    "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=150
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        st.error(f"Error summarizing text: {str(e)}")
        return ""

# Streamlit app
def main():
    st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="wide")

    st.title("📰 AI News Summarizer from YouTube")
    st.write("Automatically fetch and summarize trending AI news and tools from YouTube.")

    # Sidebar for configurations
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Enter your YouTube API Key", type="password")

    if not api_key:
        st.sidebar.error("Please enter a valid API key")
        return

    query = st.sidebar.text_input("Enter the search query", "AI tools news")
    num_videos = st.sidebar.number_input("Number of videos", min_value=1, max_value=20, value=10)
    
    # Ollama model selection
    model = st.sidebar.selectbox("Choose Ollama model", ["llama3.1:8b", "mistral-nemo:12b-instruct-2407-q2_K", "phi3:latest", "wizardlm2:7b"])

    if st.sidebar.button("Fetch and Summarize Videos"):
        with st.spinner("Fetching videos..."):
            videos = get_videos(query, max_results=num_videos, api_key=api_key)

        if not videos:
            st.warning("No videos found")
            return

        st.success(f"Videos found: {len(videos)}")

        summaries = []
        progress_bar = st.progress(0)

        for i, video in enumerate(videos):
            st.subheader(f"Processing video: {video['title']}")
            st.image(video['thumbnail'], use_column_width=True)

            with st.spinner("Extracting transcript..."):
                transcript = get_transcript(video['video_id'])

            if transcript:
                with st.spinner("Summarizing..."):
                    summary = summarize_text(transcript, model)
                summaries.append({
                    "Title": video['title'],
                    "Video ID": video['video_id'],
                    "View Count": video['view_count'],
                    "Summary": summary
                })
                st.write(summary)

            progress_bar.progress((i + 1) / len(videos))

        # Create a DataFrame for display
        df = pd.DataFrame(summaries)
        st.subheader("Summary Table")
        st.dataframe(df)

        # Save the summaries to a single .txt file
        with open("summaries.txt", "w", encoding="utf-8") as f:
            for summary in summaries:
                f.write(f"Title: {summary['Title']}\n")
                f.write(f"Video ID: {summary['Video ID']}\n")
                f.write(f"View Count: {summary['View Count']}\n")
                f.write(f"Summary:\n{summary['Summary']}\n")
                f.write("\n" + "-"*50 + "\n\n")

        st.success("Summaries saved to summaries.txt")

        # Provide download link for the .txt file
        with open("summaries.txt", "r", encoding="utf-8") as file:
            st.download_button(
                label="Download Summaries TXT",
                data=file,
                file_name="summaries.txt",
                mime="text/plain"
            )


if __name__ == '__main__':
    main()

