from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import uvicorn
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]  # Adjust this as necessary for your application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

class VideoRequest(BaseModel):
    youtube_url: str

def get_transcript(youtube_url):
  video_id = youtube_url.split("v=")[-1]
  transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

  # Try fetching the manual transcript
  try:
      transcript = transcript_list.find_manually_created_transcript()
      language_code = transcript.language_code  # Save the detected language
  except:
      # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
      try:
          generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
          transcript = generated_transcripts[0]
          language_code = transcript.language_code  # Save the detected language
      except:
          # If no auto-generated transcript is found, raise an exception
          raise Exception("No suitable transcript found.")

  full_transcript = " ".join([part['text'] for part in transcript.fetch()])
  return full_transcript, language_code  # Return both the transcript and detected language

def summarize_with_langchain_and_openai(transcript, language_code, model_name='gpt-3.5-turbo'):
    # Split the document if it's too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4]) # Adjust this as needed

    # Prepare the prompt for summarization
    system_prompt = 'I want you to act as a Life Coach that can create good summaries!'
    prompt = f'''Summarize the following text in {language_code}.
    Text: {text_to_summarize}

    Add a title to the summary in {language_code}. 
    Include an INTRODUCTION, BULLET POINTS if possible, and a CONCLUSION in {language_code}.'''

    # Start summarizing using OpenAI
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=1
    )
    
    return response['choices'][0]['message']['content']

# Optional: Define a root endpoint for health check
@app.get("/")
async def root():
    return {"message": "YouTube Video Summarizer API"}


@app.post("/summarize/")
async def summarize_video(request: VideoRequest):
    try:
        transcript, language_code = get_transcript(request.youtube_url)
        summary = summarize_with_langchain_and_openai(transcript, language_code)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)