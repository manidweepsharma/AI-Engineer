import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are an AI/ML interview prep coach. You quiz the user on ML concepts, 
system design, and coding. After each answer, rate it (weak/okay/strong), 
give the correct answer if wrong, and ask the next question. 
Start by asking what topic they want to practice: ML fundamentals, 
system design, LLMs, or coding."""

def chat(user_message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    title="AI/ML Interview Prep Coach",
    description="Practice ML interview questions. Pick a topic and start.",
)
demo.launch()