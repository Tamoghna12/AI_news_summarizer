import streamlit as st
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
import ollama
from typing import List, Dict, Any
import pandas as pd

# Function to get YouTube videos based on search criteria

# Function to check if Ollama model is available


def check_ollama_model(model: str) -> bool:
    try:
        ollama.list()
        return model in [m['name'] for m in ollama.list()['models']]
    except Exception:
        return False


def get_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    try:
        videos_search = VideosSearch(query, limit=max_results)
        results = videos_search.result()['result']

        videos = []
        for video in results:
            try:
                view_count = int(video['viewCount']['text'].replace(
                    ',', '').replace(' views', ''))
                if view_count > 10000:
                    videos.append({
                        'title': video['title'],
                        'video_id': video['id'],
                        'view_count': view_count,
                        'thumbnail': video['thumbnails'][0]['url']
                    })
            except Exception as e:
                st.warning(f"Error processing video {video['id']}: {str(e)}")
        return videos
    except Exception as e:
        st.error(f"An error occurred during search: {str(e)}")
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

# Function to summarize text using Ollama


def summarize_text(text: str, model: str) -> str:
    try:
        response = ollama.generate(
            model=model, prompt=f"Summarize the following text:\n\n{text}")
        summary = response['response']
        return summary
    except Exception as e:
        st.error(f"Error summarizing text: {str(e)}")
        return ""

# Streamlit app


def main():
    st.set_page_config(page_title="AI News Summarizer",
                       page_icon="📰", layout="wide")
    st.title("AI News Summarizer from YouTube using Ollama")

    # Sidebar for configurations
    st.sidebar.header("Configuration")
    query = st.sidebar.text_input("Enter the search query", "AI tools news")
    num_videos = st.sidebar.number_input(
        "Number of videos", min_value=1, max_value=20, value=10)
    model = st.sidebar.selectbox("Choose Ollama model", [
                                 "llama3.1:8b", "mistral-nemo:12b-instruct-2407-q2_K", "phi3:latest", "wizardlm2:7b"])

    if st.sidebar.button("Fetch and Summarize Videos"):
        with st.spinner("Fetching videos..."):
            videos = get_videos(query, max_results=num_videos)

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
