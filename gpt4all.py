import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Pinecone

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import uvicorn
from fastapi import FastAPI, Response
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


app = FastAPI() 

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/search/")
async def search_query(query: str):
    embeddings = GPT4AllEmbeddings()
    vectorstore = Pinecone.from_existing_index(
        index_name='gpt4', embedding= embeddings
    )

    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model="./models/replit-code-v1_5-3b-newbpe-q4_0.gguf", backend="gptj", callbacks=callbacks, verbose=False)

    prompt = 'Give me an answer to the query. The answer must be less than 3 sentences.' + '\nQuery: '+ query + '\nAnswer:'
    
    print("Prompt is ------>", prompt)

    llm = GPT4All(model="./models/orca-mini-3b-gguf2-q4_0.gguf", backend="gptj", callbacks=callbacks, verbose=False)

    search_kwargs = {"k": 1}  # Adjust as needed

    # Create the RetrievalQA instance using Pinecone as the retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        verbose=False,
    )

    result = qa_chain.invoke(query)
    res = result['result']
    
    print(res)
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)