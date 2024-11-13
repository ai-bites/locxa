import os
import platform
import streamlit as st
import subprocess
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.chat_message_histories.file import FileChatMessageHistory
from langchain_community.chat_models.ollama import ChatOllama
from langchain.chains.llm import LLMChain
from transformers import pipeline
from langchain.prompts import (
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
    )
from gtts import gTTS


class ChatBot:
    def __init__(self, model, temperature, max_tokens) -> None:
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_template = ChatPromptTemplate(
            input_variables=["content", "messages"],
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """
                    You're a Personal Assistant, named Locxa.
                    Give short and concise in about 50 words, but always provide a full sentence.
                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessagePromptTemplate.from_template("{content}"),
            ],
        )
        self.memory = ConversationBufferMemory(
            memory_key="messages",
            chat_memory=FileChatMessageHistory(file_path="memory.json"),
            return_messages=True,
            input_key="content",
        )

    def create_chain(self) -> LLMChain:
        chain = LLMChain(llm=self.llm, 
                 prompt=self.prompt_template, 
                 memory=self.memory)
        return chain 


def transcribe_audio(pipeline, audio_bytes):
    """Speech to text conversion"""
    ip_audo_file = "ip_audio.mp3"
    with open(ip_audo_file, "wb") as f:
        f.write(audio_bytes)
    
    transcript = None 
    if os.path.isfile(ip_audo_file):
        try:
            result = pipeline(ip_audo_file)
            transcript = result['text'].strip()
            return transcript
        except:
            print("Did not process speech very well!")
        
    return transcript


def text_to_speech(text):
    # Save the converted audio in a mp3 file named
    language = 'en'
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    
    # play the stored audio
    if platform.platform().startswith("mac"):
        os.system("afplay output.mp3")
    else:
        os.system("aplay output.mp3")