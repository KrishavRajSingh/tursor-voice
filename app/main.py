from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
import asyncio
from openai.helpers import LocalAudioPlayer
import random
from openai import AsyncOpenAI
import os
load_dotenv()

openai = AsyncOpenAI()


config = {"configurable": {"thread_id": str(random.randint(0, 1000000))}}
 

def main():
    with MongoDBSaver.from_conn_string(os.environ["MONGODB_URI"]) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        
        
        r = sr.Recognizer()

        with sr.Microphone() as source:
            # r.adjust_for_ambient_noise(source)
            r.pause_threshold = 1.5

            while True:
                try:
                    print("Say something!")
                    audio = r.listen(source)

                    print("Processing audio...")
                    sst = r.recognize_google(audio)

                    print("You Said:", sst)
                    for event in graph.stream({ "messages": [{"role": "user", "content": sst}] }, config, stream_mode="values"):
                        if "messages" in event:
                                last_message = event["messages"][-1]
                                last_message.pretty_print()
                                # print("AI:", last_message)
                                # Optional: Add TTS capability
                                if hasattr(last_message, "type") and last_message.type == "ai" and last_message.content:
                                    asyncio.run(speak(last_message.content))
                except sr.UnknownValueError:
                    print("Sorry, I did not understand that.")
                except sr.RequestError as e:    
                    print(f"Could not request results; {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")


async def speak(text: str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

main()

# if __name__ == "__main__":
#      asyncio.run(speak(text="This is a sample voice. Hi Piyush"))