import random
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
import snowflake.connector





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
        results = index_vectorstore.similarity_search_by_vector_with_score(embedding=query, k=3)
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
    st.session_state.openai_context = ''
    completion = client.chat.completions.create(
    messages = fill_template(product_profile),
    model="gpt-3.5-turbo-1106",
    max_tokens=250,
    temperature=0.6
    )
    completion = dict(completion)
    choices = completion.get('choices')

    response = choices[0].message.content
    st.session_state.openai_context += response
    return response

def claude_query_refiner(product_profile):
    st.session_state.anthropic_context = ''
    chat = ChatAnthropic(model='claude-3-opus-20240229', api_key=anthropic_api_key, temperature=.4)
    template = f"""
       Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention generic information about the industry, company size range, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.

        MAKE SURE YOUR RESPONSES ARE RELEVANT TO THE PRODUCT AND NOT GENERIC.

        MAKE SURE YOUR RESPONSES ARE DISTINCT FROM PREVIOUS RESPONSES AND DO NOT REPEAT.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}

        PREVIOUS RESPONSES:
        ({st.session_state.anthropic_context})
        """
    
    human = f"Generate 3 buyer profiles for {product_profile['Product_name']}. UNDER NO CIRCUMSTANCES SHOULD YOU REPEAT ANY OF THE RESPONSES MENTIONED HERE ({st.session_state.anthropic_context}). Do not mention a specific product's name. Put results in a numbered list. Do not call them anything, just list the paragraphs. Do not say anything else, just return the list. Append each result with a newline character."
    messages = [
            SystemMessage(content=template),
            HumanMessage(content=human),
        ]
    response = chat.invoke(messages).content
    st.session_state.anthropic_context += response
    return response


def mistral_query_refiner(product_profile):
    st.session_state.mixtral_context = ''
    chat =  ChatOpenAI(api_key=together_api_key, base_url="https://api.together.xyz/v1",
                model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=.4)
    template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention generic information about the industry, company size range, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.

        MAKE SURE YOUR RESPONSES ARE RELEVANT TO THE PRODUCT AND NOT GENERIC.

        MAKE SURE YOUR RESPONSES ARE DISTINCT FROM PREVIOUS RESPONSES AND DO NOT REPEAT.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}

        PREVIOUS RESPONSES:
        ({st.session_state.mixtral_context})
        """
    
    human = f"Generate 3 buyer profiles for {product_profile['Product_name']}. UNDER NO CIRCUMSTANCES SHOULD YOU REPEAT ANY OF THE RESPONSES MENTIONED HERE ({st.session_state.mixtral_context}). Do not mention a specific product's name. Put results in a numbered list. Do not call them anything, just list the paragraphs. Do not say anything else, just return the list. Append each result with a newline character."
    messages = [
            SystemMessage(content=template),
            HumanMessage(content=human),
        ]
    response = chat.invoke(messages).content
    st.session_state.mixtral_context += response
    return response

def llama_query_refiner(product_profile):
    st.session_state.llama_context = ''
    chat =  ChatGroq(temperature=.6, model_name="llama3-70b-8192")
    template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention generic information about the industry, company size range, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.

        MAKE SURE YOUR RESPONSES ARE RELEVANT TO THE PRODUCT AND NOT GENERIC.

        MAKE SURE YOUR RESPONSES ARE DISTINCT FROM PREVIOUS RESPONSES AND DO NOT REPEAT.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}

        PREVIOUS RESPONSES:
        ({st.session_state.llama_context})
        """
    
    human = f"Generate 3 buyer profiles for {product_profile['Product_name']}. UNDER NO CIRCUMSTANCES SHOULD YOU REPEAT ANY OF THE RESPONSES MENTIONED HERE ({st.session_state.llama_context}). Do not mention a specific product's name. Put results in a numbered list. Do not call them anything, just list the paragraphs. Do not say anything else, just return the list. Append each result with a newline character."
    messages = [
            SystemMessage(content=template),
            HumanMessage(content=human),
        ]
    response = chat.invoke(messages).content
    st.session_state.llama_context += response
    return response


def fill_template(product_profile):
     template = f"""
        Your task is to generate 3 different company personas for the given products. Each persona should be an short sentence that describes a company that would be interested in buying the product.
        Each query MUST tackle the question from a different viewpoint, we want to get a variety of RELEVANT search results.

        The goal is to cover distinct company profiles. The profiles should be short paragraph descriptions and mention generic information about the industry, company size range, the company's main focus.

        You can speculate certain geographic locations that may be interested in the product.

        Be very specific in your responses and separate each persona with a new line.

        MAKE SURE YOUR RESPONSES ARE RELEVANT TO THE PRODUCT AND NOT GENERIC.

        MAKE SURE YOUR RESPONSES ARE DISTINCT FROM PREVIOUS RESPONSES AND DO NOT REPEAT.
        
        PRODUCT DATA:
        Product Name: {product_profile['Product_name']}
        Product Description: {product_profile['Description']}
        Product Features: {product_profile['Features']}
        Sample Pitch: {product_profile['Sales pitch']}
        Product Profile: :{product_profile['Product persona']}
        Product Price: {product_profile['Price']}

        PREVIOUS RESPONSES (UNDER NO CIRCUMSTANCES SHOULD YOU REPEAT ANY OF THE RESPONSES - YOUR NEW RESPONSES SHOULD MENTION NEW INDUSTRIES, COMPANY SIZES, GEOGRAPHIC LOCATIONS, ETC.):
        ({st.session_state.openai_context})
        """
     return [
        {"role": "system", "content": template},
        {"role": "user", "content": f"Generate 3 buyer profiles for {product_profile['Product_name']}. UNDER NO CIRCUMSTANCES SHOULD YOU REPEAT ANY OF THE RESPONSES MENTIONED HERE ({st.session_state.openai_context}). Do not mention a specific product's name. Put results in a numbered list. Do not call them anything, just list the paragraphs. Do not say anything else, just return the list. Append each result with a newline character."}
    ]


# OpenAI call
def parsepdf(product, client: OpenAI):
    
    # Establish Snowflake connection
    if 'conn' not in st.session_state:
        account = 'unb07582.us-east-1'
        user = os.environ["user"]
        password =  os.environ["password"]
        database = 'TRANSFORM_DB'
        schema = 'TRANSFORM_SCHEMA_2'
        warehouse = 'TRANSFORM_WH'
        st.session_state.conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            database=database,
            schema=schema,
            warehouse=warehouse
        )
    q = f'''
    You are a text processing agent working with Software Product documents.

    You can use the source text to understand the product and extract specified values from the source text.
    Return answer as JSON object with following fields:
    - "Company_name" <string>
    - "Product_name" <string>
    - "Price" <string>
    - "Description" <string> Write a lengthy description of the product based on the source text.
    - "Features" <string> Understand the features of the product and write all the features mentioned in the source text.
    - "Sales pitch" <string> Write a short sales pitch to sell the product to wide range of customers.
    - "Product persona" <string> Write a short but accurate persona about the product based on your understanding from the source text and your understanding of the company from your own knowledge base. 
    - "Product link" <string> Find the product link from the source text. If you don't find it, Keep it null. 

    You can infer any relevant data about the company and product based on previous training for filling the Description and features column. You can also the source for more context
    ========
    {product}
    ========
    '''

    # OpenAI call
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": q}]
    )
    c = completion.choices[0].message.content

    # Parse OpenAI response
    data = json.loads(c)
        
    # Converting Dictionary to JSON
    product_info_json = json.dumps({
        "Company_name": data["Company_name"],
        "Product_name": data["Product_name"],
        "Price": data["Price"],
        "Description": data["Description"],
        "Features": data["Features"],
        "Sales pitch": data["Sales pitch"],
        "Product persona": data["Product persona"],
        "Product_link": data["Product link"]
    })

    cur = st.session_state.conn.cursor()

    sql_product_catalog = """
        INSERT INTO product_catalog (
            Company_name,
            Product_name,
            Product_profile
        ) VALUES (%s, %s, %s)
        """

    cur.execute(sql_product_catalog, (
        data["Company_name"],
        data["Product_name"],
        # Cast the JSON as an object
        product_info_json
    ))

    st.session_state.conn.commit()
    cur.close()
    return data