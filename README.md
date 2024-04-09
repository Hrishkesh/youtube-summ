
## !! Need to update Readme as this repo uses the FASTAPI and UIPath to summarize the content

# YouTube Video Summarizer

This is a Python application that allows you to summarize the content of a YouTube video using OpenAI's GPT-3.5 language model and Langchain. The application get the transcription provided by YouTube, chunks the Transcription with Langchain and generates a summary in the language of the youtube video.

## Features
- Get the Transcription from Youtube
- Chunks the transcriptions with Langchain
- Summarizes transcribed text using OpenAI's GPT-3.5 model
- Built with Streamlit for an easy-to-use web interface

## Prerequisites

Before you begin, ensure you have installed the following:

- Python 3.6 or above
- [PyTube](https://pytube.io/en/latest/)
- [OpenAI](https://beta.openai.com/docs/developer-quickstart/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/)
- [Langchain](https://pypi.org/project/langchain/)

## Installation 

1. Clone this repository:
```bash
git clone https://github.com/DevRico003/youtube_summarizer
```
2. Change into the cloned repository:
```bash
cd youtube_summarizer
```
3. Install all necessary packages:
```bash
pip install -r requirements.txt
```
4. Create a `.env` directory in your home directory (or any directory of your choice), and create in the directory `.env` a file called `openai_api` and add your OpenAI API Key:
```bash
OPENAI_API_KEY=your_openai_api_key
```
5. Change the `env_path` variable in the Python script to match the path of your `.env` file.

## Usage


## Disclaimer

Please note that this application relies on the OpenAI GPT-3.5 language model, and its performance and results are subject to the capabilities of the model and the quality of the provided data. The generated summaries may not always be perfect and may require manual editing for accuracy.




