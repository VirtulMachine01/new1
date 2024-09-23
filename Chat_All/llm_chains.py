from prompt_templates import memory_prompt_template
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Chroma
# import sys
# import pysqlite3
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_llm(model_path = config["model_path"]["small"], model_type = config["model_type"], model_config=config["model_config"]):
    llm = CTransformers(model = model_path, model_type= model_type, config = model_config)
    return llm

def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name = embeddings_path)

def create_embeddings2(embeddings_name = config["embedding_name"]):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embeddings_name,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}
        )
    return embeddings

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

# def create_llm_chain(llm, chat_prompt):
#     return chat_prompt | llm
def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

def load_normal_chain(chat_history):
    return chatChain(chat_history)


def load_vectorstore(index_path):
    embeddings = create_embeddings2()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
# def load_vectordb(embeddings):
#     persistent_client = chromadb.PersistentClient("chroma_db")
#     langchain_chroma = Chroma(
#         client=persistent_client,
#         collection_name="pdfs",
#         embedding_function=embeddings
#     )

#     return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return PDFChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever = vector_db.as_retriever())

class PDFChatChain:
    def __init__(self, chat_history,):
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectorstore("faiss_db/faiss_index.index")
        llm = create_llm()
        self.llm_chain = load_retrieval_chain(llm, self.memory, self.vector_db)
    def run(self, user_input):
        return self.llm_chain.invoke({"query": user_input, "history": self.memory.chat_memory.messages})
    

class chatChain:

    def __init__(self, chat_history,):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        # self.llm_chain = create_llm_chain(llm, chat_prompt)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)
    
    def run(self, user_input):
        return self.llm_chain.invoke({"human_input": user_input, "history": self.memory.chat_memory.messages})
        # return self.llm_chain.run(human_input=user_input, history=self.memory.chat_memory.messages, stop=["Human:"])