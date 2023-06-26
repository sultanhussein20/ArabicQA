import os
import sys

import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to cache & reuse the model to disk (for repeated queries on the same data)
PERSIST = False
query = 'ما هو اسمي'
query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  from langchain.indexes.vectorstore import VectorStoreIndexWrapper
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  # txt = 'اسمي حسن'
  # with open('data.txt', "w",  encoding='utf-32') as file:
  #       file.write(txt)

  loader = TextLoader('data.txt',  encoding='utf-32')
  # This code can also import folders, including various filetypes like PDFs using the DirectoryLoader.
  # loader = DirectoryLoader(".", glob="*.txt")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = RetrievalQA.from_chain_type(
  # llm=ChatOpenAI(model="gpt-3.5-turbo"),
  llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613"),  
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)
print(chain.run(query))
