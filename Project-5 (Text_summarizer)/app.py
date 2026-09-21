from fastapi import FastAPI, Request
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pydantic import BaseModel
import torch
import re
from fastapi.templating import Jinja2Templates #UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# Initialize our FastApi  app
app = FastAPI(
    title="Text Summarizer",
    description="Text Summariztion using T5",
    version="1.0")

#Model and Tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#Device 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model.to(device)

# templating
template = Jinja2Templates(directory=".")

#Input Schema for dialogue
class DialogueInput(BaseModel):
    dialogue: str
    

def clean_data(text):
    text = re.sub(r"\r\n"," ",  text)  # remove next line etc.
    text = re.sub(r"\s+", " ", text)  # extra space
    text = re.sub(r"<.*?>", " ", text)  # HTML Tags
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


#API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return template.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )