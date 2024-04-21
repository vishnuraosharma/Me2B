from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
import streamlit as st
import Product_Model.product as p
import Product_Model.product_catalog as pc
import Product_Model.product_audience as pa
import json
import pickle
import os
from os.path import join
from dotenv import load_dotenv
from openai import OpenAI
import csv
import time
import csv
from snowflake.snowpark import Session
import pandas as pd
import requests
from pinecone import Pinecone as PinconeDB
from langchain_pinecone import Pinecone, PineconeVectorStore
#from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import logging
from langchain_openai import OpenAIEmbeddings
from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils import *
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
import plotly.express as px


st.set_page_config(page_title="Me2B", page_icon=":bee:", layout="wide")
 

def upload_to_pinecone(json_data):
    print('vectorizing')
    st.session_state.count = 0
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    st.session_state.embeddings = model.encode([json_data])
    print(st.session_state.embeddings)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("product-descriptions")
    index.upsert(
                vectors= [{ "id": f"{st.session_state.count}", 
                            "values":st.session_state.embeddings[0],
                            "metadata":{"product_name": "Automated Agent QA"
                                        }
                            }],
                namespace="pd1"
                )
    print("uploaded")
    st.session_state.count = st.session_state.count + 1
    del st.session_state.embeddings
    refresh_product_data()


def refresh_product_data():
    if 'product_ids' in st.session_state:
        del st.session_state.product_ids
    if 'product_data_dict' in st.session_state:
        del st.session_state.product_data_dict

def refresh_current_chat(curr_model):
    if curr_model == "GPT 3.5 Turbo":
        del st.session_state.msgs_open_ai
        if 'langchain_messages_open_ai' in st.session_state:
            del st.session_state.langchain_messages_open_ai
        if 'open_ai_matches' in st.session_state:
            del st.session_state.open_ai_matches
        if 'open_ai_refined_queries' in st.session_state:
            del st.session_state.open_ai_refined_queries

    elif curr_model == "LLAMA3":
        del st.session_state.msgs_llama3
        if 'langchain_messages_llama3' in st.session_state:
            del st.session_state.langchain_messages_llama3
        if 'llama3_matches' in st.session_state:
            del st.session_state.llama3_matches
        if 'llama3_refined_queries' in st.session_state:
            del st.session_state.llama3_refined_queries

    elif curr_model == "Claude 3.0":
        del st.session_state.msgs_claude
        if 'langchain_messages_claude' in st.session_state:
            del st.session_state.langchain_messages_claude
        if 'claude_ai_matches' in st.session_state:
            del st.session_state.claude_ai_matches
        if 'claude_refined_queries' in st.session_state:
            del st.session_state.claude_refined_queries

    elif curr_model == "Mixtrial":
        del st.session_state.msgs_mistral
        if 'langchain_messages_mistral' in st.session_state:
            del st.session_state.langchain_messages_mistral
        if 'mistral_ai_matches' in st.session_state:
            del st.session_state.mistral_ai_matches
        if 'mistral_refined_queries' in st.session_state:
            del st.session_state.mistral_refined_queries

    # New product selected, clear all chat messages
    else:
        del st.session_state.msgs_open_ai
        if 'langchain_messages_open_ai' in st.session_state:
            del st.session_state.langchain_messages_open_ai
        if 'open_ai_matches' in st.session_state:
            del st.session_state.open_ai_matches
        if 'open_ai_refined_queries' in st.session_state:
            del st.session_state.open_ai_refined_queries

        del st.session_state.msgs_mistral
        if 'langchain_messages_mistral' in st.session_state:
            del st.session_state.langchain_messages_mistral
        if 'mistral_ai_matches' in st.session_state:
            del st.session_state.mistral_ai_matches
        if 'mistral_refined_queries' in st.session_state:
            del st.session_state.mistral_refined_queries

        del st.session_state.msgs_claude
        if 'langchain_messages_claude' in st.session_state:
            del st.session_state.langchain_messages_claude
        if 'claude_ai_matches' in st.session_state:
            del st.session_state.claude_ai_matches
        if 'claude_refined_queries' in st.session_state:
            del st.session_state.claude_refined_queries
        
        del st.session_state.msgs_llama3
        if 'langchain_messages_llama3' in st.session_state:
            del st.session_state.langchain_messages_llama3
        if 'llama3_matches' in st.session_state:
            del st.session_state.llama3_matches
        if 'llama3_refined_queries' in st.session_state:
            del st.session_state.llama3_refined_queries
        

def refresh_all_chats_and_delete_selected_product():
    del st.session_state.msgs_open_ai
    if 'langchain_messages_open_ai' in st.session_state:
        del st.session_state.langchain_messages_open_ai
    if 'open_ai_matches' in st.session_state:
        del st.session_state.open_ai_matches
    if 'open_ai_refined_queries' in st.session_state:
        del st.session_state.open_ai_refined_queries

    del st.session_state.msgs_mistral
    if 'langchain_messages_mistral' in st.session_state:
        del st.session_state.langchain_messages_mistral
    if 'mistral_ai_matches' in st.session_state:
        del st.session_state.mistral_ai_matches
    if 'mistral_refined_queries' in st.session_state:
        del st.session_state.mistral_refined_queries

    del st.session_state.msgs_claude
    if 'langchain_messages_claude' in st.session_state:
        del st.session_state.langchain_messages_claude
    if 'claude_ai_matches' in st.session_state:
        del st.session_state.claude_ai_matches
    if 'claude_refined_queries' in st.session_state:
        del st.session_state.claude_refined_queries
    
    del st.session_state.msgs_llama3
    if 'langchain_messages_llama3' in st.session_state:
        del st.session_state.langchain_messages_llama3
    if 'llama3_matches' in st.session_state:
        del st.session_state.llama3_matches
    if 'llama3_refined_queries' in st.session_state:
        del st.session_state.llama3_refined_queries
    
    if 'product_id' in st.session_state:
        del st.session_state.product_id
    if 'product_profile_json' in st.session_state:
        del st.session_state.product_profile_json
    

def get_product_names_from_vector_dict(vector_dict):
    return [dict["metadata"]["product_name"] for id, dict in vector_dict.items() if dict["metadata"]["product_name"]]

def create_chain(vectorstore, openai_api_key):
    pass

def get_embeddings(text):
    response = st.session_state.client.Embed.create(
        input=[text],
        model="text-embedding-3-small"  # Choose the model that best fits your use case
    )
    return response['data'][0]['embedding']

def del_environment_variables():
    if 'OPENAI_API_KEY' in os.environ:
        # delete current environment variables
        del os.environ["OPENAI_API_KEY"]
        del os.environ["ANTHROPIC_API_KEY"]
        del os.environ["GEMINI_API_KEY"]
        del os.environ["TOGETHER_API_KEY"]
        del os.environ["PINECONE_API_KEY"]
        del os.environ["SNOWFLAKE_ACCOUNT"]
        del os.environ["SNOWFLAKE_USER"]
        del os.environ["SNOWFLAKE_WAREHOUSE"]
        del os.environ["TEST_DB"]
        del os.environ["SNOWFLAKE_PASSWORD"]
        

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
groq_api_key = os.getenv("GROQ_API_KEY")

st.session_state.client = OpenAI(api_key=openai_api_key)

####################
## PINCONE
####################
# Connect to the company index
if 'pc_company' not in st.session_state:
    st.session_state.pinecone_db = PinconeDB(api_key=os.environ.get("PINECONE_API_KEY"))
    # Get the company index
    st.session_state.pc_company_idx = st.session_state.pinecone_db.Index(host="https://customers-ttxf9sp.svc.aped-4627-b74a.pinecone.io")
    st.session_state.vectorstore = PineconeVectorStore(index_name="customers",
                embedding=OpenAIEmbeddings(),
                namespace="Company-Vector",
                text_key="description",
            )

####################
## SNOWFLAKE
####################
# Get the credentials from .env
connection_parameters = {
"account"    : os.getenv('SNOWFLAKE_ACCOUNT'),
"user"       : os.getenv('SNOWFLAKE_USER'),
"warehouse"  : os.getenv('SNOWFLAKE_WAREHOUSE'),
"database"   : os.getenv('TEST_DB'),
"password"   : os.getenv('SNOWFLAKE_PASSWORD')
}

if 'product_data_tbl' not in st.session_state:
    # fire up an instance of a snowflake connection
    st.session_state.new_session = Session.builder.configs(connection_parameters).create()

    product_data = st.session_state.new_session.table("TRANSFORM_DB.TRANSFORM_SCHEMA_2.SAMPLE_PRODUCT_DATA")

    # Convert the Snowflake dataframe to a pandas dataframe
    st.session_state.product_data_tbl = product_data.toPandas()

    # Rename the columns
    st.session_state.product_data_tbl.columns = ['ID','Company Name','Product Name', 'Product Profile']
    st.session_state.product_names = st.session_state.product_data_tbl["Product Name"].unique()
    
    st.session_state.color_dict = {
    'light_yellow': '#f7ebcd',
    'mustard':'#febc2e',
    'dark_yellow':'#eba301',
    'chocolate':'#78591b'
}

    company_table = st.session_state.new_session.table("TRANSFORM_DB.TRANSFORM_SCHEMA_2.companiesdedupeddata")

    # Convert the Snowflake dataframe to a pandas dataframe
    st.session_state.company_data_tbl = company_table.toPandas()

    # Rename the columns
    st.session_state.company_data_tbl.columns = ['WEBSITE','HANDLE','NAME','TYPE','INDUSTRY','SIZE','FOUNDED','COUNTRY_CODE','CITY','STATE','UID','SCRAPPEDDATE','SCRAPPEDDATA','ROWNUM']
    
    #drop WEBSITE, SCRAPPEDDATE,SCRAPPEDDATA, ROWNUM
    st.session_state.company_data_tbl.drop(columns=['WEBSITE','SCRAPPEDDATE','SCRAPPEDDATA','ROWNUM','HANDLE', 'UID'], inplace=True)

    # Round the FOUNDED column to the nearest year
    st.session_state.company_data_tbl['FOUNDED'] = pd.to_numeric(st.session_state.company_data_tbl['FOUNDED'], errors='coerce').fillna(0).astype(int)

    # replace 0 with NaN
    st.session_state.company_data_tbl['FOUNDED'] = st.session_state.company_data_tbl['FOUNDED'].replace(0, pd.NA)

    # Uppercase Type and industry columns
    st.session_state.company_data_tbl['TYPE'] = st.session_state.company_data_tbl['TYPE'].str.upper()
    st.session_state.company_data_tbl['INDUSTRY'] = st.session_state.company_data_tbl['INDUSTRY'].str.upper()



talk, view, upload = st.tabs(["Me2B Bot", "Me2B Buyer Analytics", "Update Product Catalog"])
if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = """You are an B2B business development representative bot called me2b. You are an expert on making matches between product data and customers.  
                        You have two main functions:
                        1. Based the potential company matches, you find the best matches between a product and a few potential customers. You can articulate why a company is a good lead for a product and what the potential value is for the company. 
                        2. You are also a master at writing an excellent, concise cold email or linkedin message to outreach to a company. You know how to make a great pitch and are best known for your ability to make a match between a product and a potential customer and writing concisely.

                        First, read through this product profile ('{description}'), these key features ('{features}').
                        Next, read this description of a fictional company persona that might want to use this product ({potential_customers}). 
                        Finally, evaluate potential company matches below and highlight product features that are a good match and come up with a few reasons why the product is a good fit for the company. Comapny persona matches will arrive in the next system message.
                        For each match, write a few sentences about why the product is a good fit for the company.
                        Pick your favorite match and explain why it is the best fit in one or two sentences at the end of the message.
                        Ask the user if he wants to write a cold email to the company and which role the email should be addressed to.

                        ONLY WHEN PROMPTED you should write an email to pitch the product to the user-specified company. Your first message should not include any email content.

                        When prompted, write a cold email to pitch the product to the company. Your email should include the following and be a few concise sentences: 
                            Mention potential competitors. Mention the product description, and give specific reasons for why the product is a good fit for the company.
                            Use the information to write a cold email to pitch the product. Please mention a specific integration in this list in your message.
                            Here's some extra color for your message ('{pitch}')                    
                        
                        DO NOT INCLUDE MADE UP INFORMATION. ONLY USE INFORMATION PROVIDED IN THE PROMPT OR THE POTENTIAL COMPANY MATCHES.
                            """
with talk:

    curr_product = st.sidebar.selectbox("Select one of your products", st.session_state.product_names, key="curr_prod", on_change=refresh_all_chats_and_delete_selected_product)
    if 'product_id' not in st.session_state:
        st.session_state.product_id = st.session_state.product_data_tbl[st.session_state.product_data_tbl["Product Name"] == curr_product]["ID"].values[0]
        st.session_state.product_profile_json = json.loads(dict(st.session_state.product_data_tbl[st.session_state.product_data_tbl["Product Name"] == curr_product]["Product Profile"])[st.session_state.product_id])

        print(st.session_state.product_profile_json)
    
    st.info("Choose a model to generate potential buyer match profiles and chat with about them.")

    curr_model = st.sidebar.selectbox("Select a model", ["GPT 3.5 Turbo", "Claude 3.0", "Mixtrial", "LLAMA3"], key="curr_model")
    # Set up memory
    st.session_state.msgs_open_ai = StreamlitChatMessageHistory(key="langchain_messages_open_ai")
    st.session_state.msgs_claude = StreamlitChatMessageHistory(key="langchain_messages_claude")
    st.session_state.msgs_mistral = StreamlitChatMessageHistory(key="langchain_messages_mistral")
    st.session_state.msgs_llama3 = StreamlitChatMessageHistory(key="langchain_messages_llama3")
    if len(st.session_state.msgs_open_ai.messages) == 0:
        st.session_state.msgs_open_ai.add_ai_message("Ready to find a new customer?")
    if len(st.session_state.msgs_claude.messages) == 0:
        st.session_state.msgs_claude.add_user_message("...")
        st.session_state.msgs_claude.add_ai_message("Ready to find a new customer?")
    if len(st.session_state.msgs_mistral.messages) == 0:
        st.session_state.msgs_mistral.add_ai_message("Ready to find a new customer?")
    if len(st.session_state.msgs_llama3.messages) == 0:
        st.session_state.msgs_llama3.add_ai_message("Ready to find a new customer?")

    
    # view_messages = st.expander("View the message contents in session state")

    ###########################
    ### OpenAI
    ###########################
    if curr_model == "GPT 3.5 Turbo":
        if 'open_ai_refined_queries' not in st.session_state or 'open_ai_matches' not in st.session_state:
            with st.spinner("Finding matches..."):
                st.session_state.open_ai_refined_queries = openAI_query_refiner( product_profile= st.session_state.product_profile_json, client= st.session_state.client)
                st.sidebar.subheader("Product Profiles")
                st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                \nHere are a few examples. """)
                st.sidebar.write(st.session_state.open_ai_refined_queries)
                st.session_state.open_ai_matches = find_match_with_lang(st.session_state.open_ai_refined_queries, st.session_state.vectorstore)
                #st.session_state.open_ai_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
        else:
            st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHere are a few examples. """)            
            st.sidebar.write(st.session_state.open_ai_refined_queries)


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.prompt_template),
                "system", f"Here are the potential company persona matches: {st.session_state.open_ai_matches}",
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | ChatOpenAI(api_key=openai_api_key)
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: st.session_state.msgs_open_ai,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Render current messages from StreamlitChatMessageHistory
        for msg in st.session_state.msgs_open_ai.messages:
            st.chat_message(msg.type).write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if prompt := st.chat_input():
            st.chat_message("human").write(prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": prompt,
                                                 "features": st.session_state.product_profile_json["Features"],
                                                 "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                 }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()

        # # Draw the messages at the end, so newly generated ones show up immediately
        # with view_messages:
        #     view_messages.json(st.session_state.langchain_messages_open_ai)
    
    ###########################      
    ### Claude
    ###########################
    
    elif curr_model == "Claude 3.0":
        if 'claude_matches' not in st.session_state or 'claude_refined_queries' not in st.session_state:
            with st.spinner("Finding matches..."):
                st.session_state.claude_refined_queries = claude_query_refiner( product_profile= st.session_state.product_profile_json)
                st.sidebar.subheader("Product Profiles")
                st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHere are a few examples. """)
                st.sidebar.write(st.session_state.claude_refined_queries)
                st.session_state.claude_matches = find_match_with_lang(st.session_state.claude_refined_queries, st.session_state.vectorstore)
                #st.session_state.open_ai_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
        else:
            st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHere are a few examples. """)
            st.sidebar.write(st.session_state.claude_refined_queries)


        claude_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.prompt_template),
                "system", f"Here are the potential company persona matches: {st.session_state.claude_matches}",
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = claude_prompt | ChatAnthropic(model='claude-3-opus-20240229', api_key=anthropic_api_key)
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: st.session_state.msgs_claude,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Render current messages from StreamlitChatMessageHistory
        for msg in st.session_state.msgs_claude.messages:
            st.chat_message(msg.type).write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if claude_prompt := st.chat_input():
            st.chat_message("human").write(claude_prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": claude_prompt,
                                                 "features": st.session_state.product_profile_json["Features"],
                                                 "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                 }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()


        # # Draw the messages at the end, so newly generated ones show up immediately
        # with view_messages:
        #     view_messages.json(st.session_state.langchain_messages_claude)
    
    ###########################
    ### mistral
    ###########################
    elif curr_model == "Mixtrial":
        if 'mistrial_refined_queries' not in st.session_state or 'mistrial_matches' not in st.session_state:
            with st.spinner("Finding matches..."):
                st.session_state.mistrial_refined_queries = mistral_query_refiner( product_profile= st.session_state.product_profile_json)
                st.sidebar.subheader("Product Profiles")
                st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                \nHere are a few examples. """)
                st.sidebar.write(st.session_state.mistrial_refined_queries)
                st.session_state.mistrial_matches = find_match_with_lang(st.session_state.mistrial_refined_queries, st.session_state.vectorstore)
                #st.session_state.mistrial_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
        else:
            st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHere are a few examples. """)            
            st.sidebar.write(st.session_state.mistrial_refined_queries)

        mixtrial_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.prompt_template),
                "system", f"Here are the potential company persona matches: {st.session_state.mistrial_matches}",
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = mixtrial_prompt | ChatOpenAI(api_key=together_api_key, base_url="https://api.together.xyz/v1",
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",)
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: st.session_state.msgs_mistral,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Render current messages from StreamlitChatMessageHistory
        for msg in st.session_state.msgs_mistral.messages:
            st.chat_message(msg.type).write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if mixtrial_prompt := st.chat_input():
            st.chat_message("human").write(mixtrial_prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": mixtrial_prompt,
                                                 "features": st.session_state.product_profile_json["Features"],
                                                 "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                 }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()
        
        ###########################      
        ### LLaMa3
        ###########################
    elif curr_model == "LLAMA3":
        if 'llama3_matches' not in st.session_state or 'llama3_refined_queries' not in st.session_state:
            with st.spinner("Finding matches..."):
                st.session_state.llama3_refined_queries = mistral_query_refiner( product_profile= st.session_state.product_profile_json)
                st.sidebar.subheader("Product Profiles")
                st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                \nHere are a few examples. """)
                st.sidebar.write(st.session_state.llama3_refined_queries)
                st.session_state.llama3_matches = find_match_with_lang(st.session_state.llama3_refined_queries, st.session_state.vectorstore)
                #st.session_state.mistrial_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
        else:
            st.sidebar.info("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHere are a few examples. """)            
            st.sidebar.write(st.session_state.llama3_refined_queries)

        llama_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.prompt_template),
                "system", f"Here are the potential company persona matches: {st.session_state.llama3_matches}",
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        
        chain = llama_prompt | ChatGroq(temperature=0, model_name="llama3-70b-8192")
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: st.session_state.msgs_llama3,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Render current messages from StreamlitChatMessageHistory
        for msg in st.session_state.msgs_llama3.messages:
            st.chat_message(msg.type).write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if llama_prompt := st.chat_input():
            st.chat_message("human").write(llama_prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": llama_prompt,
                                                "features": st.session_state.product_profile_json["Features"],
                                                "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()
   
    restart_chat = st.button("Restart Chat")
    if restart_chat:
        refresh_current_chat(st.session_state.curr_model)
        st.rerun()


with view:
    # show 'NAME','TYPE','INDUSTRY','SIZE','FOUNDED','COUNTRY_CODE','CITY','STATE'
    st.dataframe(st.session_state.company_data_tbl[['NAME','TYPE','INDUSTRY','SIZE','FOUNDED','COUNTRY_CODE','CITY','STATE']])


    type, size = st.columns(2)
    industry, founding = st.columns(2)

    with type:

        # Visualization 1: Bar chart for company types
        type_counts = st.session_state.company_data_tbl['TYPE'].value_counts().reset_index()
        type_counts.columns = ['TYPE', 'COUNT']
        fig1 = px.bar(type_counts, x='TYPE', y='COUNT', title="Company Types")
        #change the color of the bar chart
        fig1.update_traces(marker_color=st.session_state.color_dict['dark_yellow'])
        st.plotly_chart(fig1, use_container_width=True)

    with size:
        # Visualization 2: Distribution of company sizes
        fig2 = px.histogram(st.session_state.company_data_tbl, x='SIZE', title="Company x Size Distribution", nbins=20)
        # update y-axis label
        fig2.update_yaxes(title_text='COUNT')
        fig2.update_traces(marker_color=st.session_state.color_dict['dark_yellow'])
        st.plotly_chart(fig2, use_container_width=True)

    with industry:
        # Visualization 3: Distribution of companies by industry
        industry_counts = st.session_state.company_data_tbl['INDUSTRY'].value_counts().reset_index().head(10)
        industry_counts.columns = ['INDUSTRY', 'COUNT']
        fig3 = px.bar(industry_counts, x='INDUSTRY', y='COUNT', title="Company x Industry Distribution")
        fig3.update_traces(marker_color=st.session_state.color_dict['dark_yellow'])
        st.plotly_chart(fig3, use_container_width=True)

    with founding:
        # Visualization 4: Scatter plot of company founding year vs. size
        founding_counts = st.session_state.company_data_tbl['FOUNDED'].value_counts().reset_index().head(10)
        founding_counts.columns = ['FOUNDED', 'COUNT']
        fig4 = px.bar(founding_counts, x='FOUNDED', y='COUNT', title="Company x Founding Year Distribution")
        fig4.update_traces(marker_color=st.session_state.color_dict['dark_yellow'])
        st.plotly_chart(fig4, use_container_width=True)


with upload:
    
    st.title('Upload Product Data')
    st.table(st.session_state.product_data_tbl)

    # Create a sample SAAS product and add it to the product catalog
    p_audience = pa.ProductAudience("Companies with a strong digital presence that pride themselves in Customer Service.", "United States", {'min': 20, 'max':50000}, ["Salesforce", "Zendesk", "Intercom", "Gorgias", "Khoros", "Gladly", "Five9"], 
                                    {'min': 10000000, 'max':100000000}, ["B2B", "Customer Service", "e-Commerce","BPO", "SaaS", "Retail", "Healthcare", "Telecom", "Financial Services","Logistics"])
    product_cat = pc.ProductCatalog()
    saas_product = p.Product("Automated Agent QA", 
                            """Automated Agent QA helps customer service teams automate their QA process by processing conversations with AI to understand if agents did a good job sticking to policy and resolving customer issues across all contacts. 
                            It uses AI to analyze customer service interactions over chat, email, or voice and provide insights to provide direct feedback to agents improve customer service. Automated QA automates the time-consuming parts of the agent performance review process, so you have more time for higher value work like conversational analysis and agent coaching. 
                            """, 
                            "Conversation Intelligence", vertical = "Customer Service", competitors=["Maestro QA", "Playvox", "Klaus"],
                            integrations= ["Salesforce", "Zendesk", "Intercom", "Gorgias", "Khoros", "Gladly", "Five9"],      
                            features=["Automate interaction selection using AI to highlight and score the most impactful customer conversations", "Agent Performance Review"
                            , "Conduct more reviews of customer interactions without having to listen to entire phone calls or comb through transcripts",
                            "Easily pinpoint relevant parts of the conversation using the Loris Sentiment Graph to focus QA analyst reviews", "Conversational Analysis", """Automatically complete agent scorecards for effective communication skills, helping QA analysts do more reviews on more agents with less effort. Easily tailor your workflows, scorecards, and assignments based on what’s important to your customer experience. Simplify everything QA – combining customer interactions, scorecards, agent coaching, and disputes into one platform""", 
                            """Use AI to create objective scoring across all your agents – and across all your channels – for more uniform evaluation. Spot trends and insights at the agent, team, and organizational level Use conversational insights to remove customer friction, improve your products, and develop new offerings""",
                            "Agent Coaching"], audience=p_audience)

    product_cat.add_product(saas_product)
    #Serialize the product catalog to a json string
    product_cat_json = json.dumps(product_cat, default=lambda o: o.__dict__, indent=2)
    with st.form("my_form"):
        st.write("Inside the form")
        name = st.text_input("Product Name", value="Automated Agent QA")
        product_desc = st.text_area("Product Details", value=product_cat_json)


        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            
            # Convert product_desc to JSON format
            json_data = {
                "product_desc": product_desc
            }
            # Upload JSON data to Pinecone
            upload_to_pinecone(json_data)