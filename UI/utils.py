from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PinconeDB
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI
import streamlit as st
import os
from os.path import join
from dotenv import load_dotenv
import json
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage




def del_environment_variables():
    if 'OPENAI_API_KEY' in os.environ:
        # delete current environment variables
        del os.environ["OPENAI_API_KEY"]
        del os.environ["ANTHROPIC_API_KEY"]
        del os.environ["GEMINI_API_KEY"]
        del os.environ["TOGETHER_API_KEY"]
        del os.environ["PINECONE_API_KEY"]

    # Get the absolute path of the current script file
    script_path = os.path.abspath(os.getcwd() + os.sep + os.pardir)
    dotenv_path = join(script_path, '.env')
    load_dotenv(dotenv_path)

del_environment_variables()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")
customer_index_key = os.getenv("PINECONE_API_KEY")

pinecone_db = PinconeDB(api_key=os.environ.get("PINECONE_API_KEY"))
# Get the company index
pc_company_idx = pinecone_db.Index(host="https://customers-ttxf9sp.svc.aped-4627-b74a.pinecone.io")
client = OpenAI(api_key=openai_api_key)

def find_match_with_lang(input, index_vectorstore: PineconeVectorStore):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    # Assume input is a string of a numbered list of queries, separated by newlines
    queries = input.split("\n")
    # Strip numbers and whitespace and newlines
    queries = [q.strip() for q in queries if q.strip()]
    embedded_queries = embeddings_model.embed_documents(queries)
    distinct_results = []
    for query in embedded_queries:
        results = index_vectorstore.similarity_search_by_vector_with_score(embedding=query, k=2)
        for result in results:
            if result[0] not in distinct_results:
                distinct_results.append(result[0])
    print(distinct_results)
    return distinct_results

def find_match_with_pinecone(input, pincone_idx :PinconeDB.Index):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    # Assume input is a string of a numbered list of queries, separated by newlines
    queries = input.split("\n")
    # Strip numbers and whitespace and newlines
    queries = [q.strip() for q in queries if q.strip()]
    embedded_queries = embeddings_model.embed_documents(queries)
    distinct_results = []
    for query in embedded_queries:
        results = pincone_idx.query(vector=query, top_k=2, namespace="Company-Vectors",include_metadata=True)
        for result in results:
            if result not in distinct_results:
                distinct_results.append(result)

    print(distinct_results)
    return distinct_results

def openAI_query_refiner(product_profile, client: OpenAI):
    completion = client.chat.completions.create(
    messages = fill_template(product_profile),
    model="gpt-3.5-turbo-1106",
    max_tokens=250,
    temperature=0.6
    )
    completion = dict(completion)
    choices = completion.get('choices')

    return choices[0].message.content

def claude_query_refiner(product_profile):
    chat = ChatAnthropic(model='claude-3-opus-20240229', api_key=anthropic_api_key, temperature=.4)
    template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention the industry, company size, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}
        """
    
    human = f"Generate 3 buyer profiles for {product_profile['Product_name']}. Put results in a numbered list. Do not call them anything, just list the paragraphs. Append each result with a newline character."
    messages = [
            SystemMessage(content=template),
            HumanMessage(content=human),
        ]
    return chat.invoke(messages).content


def mistral_query_refiner(product_profile):
    chat =  ChatOpenAI(api_key=together_api_key, base_url="https://api.together.xyz/v1",
                model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=.4)
    template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention the industry, company size, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}
        """
    
    human = f"Generate 3 buyer profiles for {product_profile['Product_name']}. Put results in a numbered list. Do not call them anything, just list the paragraphs. Append each result with a newline character."
    messages = [
            SystemMessage(content=template),
            HumanMessage(content=human),
        ]
    return chat.invoke(messages).content

def llama_query_refiner(product_profile):
    chat =  ChatGroq(temperature=.5, model_name="llama3-70b-8192")
    template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention the industry, company size, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}
        """
    
    human = f"Generate 3 buyer profiles for {product_profile['Product_name']}. Put results in a numbered list. Do not call them anything, just list the paragraphs. Append each result with a newline character."
    messages = [
            SystemMessage(content=template),
            HumanMessage(content=human),
        ]
    print(chat.invoke(messages).content)
    return chat.invoke(messages).content


def fill_template(product_profile):
     template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention the industry, company size, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}
        """
     return [
        {"role": "system", "content": template},
        {"role": "user", "content": f"Generate 3 buyer profiles for {product_profile['Product_name']}. Put results in a numbered list. Do not call them anything, just list the paragraphs. Append each result with a newline character."}
    ]

