from langchain_pymupdf4llm import PyMuPDF4LLMLoader
# from langchain_pymupdf4llm.pymupdf4llm_loader import 
from langchain_pymupdf4llm import PyMuPDF4LLMParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
load_dotenv()   

URL = os.getenv('URL')
API_KEY = os.getenv('OPENROUTER_API_KEY')
model = ChatOpenAI(
                model="amazon/nova-2-lite-v1:free",
                temperature=temperature,
                api_key=API_KEY, # type: ignore
                base_url=URL,
                # max_tokens=max_tokens
            )
PATH = '../data/Applied-Machine-Learning-and-AI-for-Engineers (2).pdf'

loader = PyMuPDF4LLMLoader(PATH)
doc = loader.load()


