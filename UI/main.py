from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
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
import plotly.express as px
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from PyPDF2 import PdfReader


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

    product_data = st.session_state.new_session.table("TRANSFORM_DB.TRANSFORM_SCHEMA_2.PRODUCT_CATALOG")

    # Convert the Snowflake dataframe to a pandas dataframe
    st.session_state.product_data_tbl = product_data.toPandas()

    # Rename the columns
    st.session_state.product_data_tbl.columns = ['ID','Company Name','Product Name', 'Product Profile']
    # Convert the product profile to a dictionary for each row
    st.session_state.product_data_tbl['Product Profile'] = st.session_state.product_data_tbl['Product Profile'].apply(lambda x: json.loads(x))
    st.session_state.product_names = st.session_state.product_data_tbl["Product Name"].unique()
    
if 'company_data_tbl' not in st.session_state:
    
    st.session_state.color_dict = {
    'light_yellow': '#f7ebcd',
    'mustard':'#febc2e',
    'dark_yellow':'#eba301',
    'chocolate':'#78591b'
    }

    
    st.session_state.company_data_tbl = st.session_state.new_session.table("TRANSFORM_DB.TRANSFORM_SCHEMA_2.companiesdedupeddata")
    st.session_state.company_data_tbl = st.session_state.company_data_tbl.toPandas()
    st.session_state.persona_data_tbl = st.session_state.new_session.table("TRANSFORM_DB.TRANSFORM_SCHEMA_2.personas")
    st.session_state.persona_data_tbl = st.session_state.persona_data_tbl .toPandas()
    
    # Rename the columns
    st.session_state.company_data_tbl.columns = ['WEBSITE','HANDLE','NAME','TYPE','INDUSTRY','SIZE','FOUNDED','COUNTRY_CODE','CITY','STATE','UID','SCRAPPEDDATE','SCRAPPEDDATA','ROWNUM']
    
    #drop WEBSITE, SCRAPPEDDATE,SCRAPPEDDATA, ROWNUM
    st.session_state.company_data_tbl.drop(columns=['WEBSITE','SCRAPPEDDATE','SCRAPPEDDATA','ROWNUM','HANDLE'], inplace=True)

    # Round the FOUNDED column to the nearest year
    st.session_state.company_data_tbl['FOUNDED'] = pd.to_numeric(st.session_state.company_data_tbl['FOUNDED'], errors='coerce').fillna(2005).astype(int)

    # replace 0 with NaN
    st.session_state.company_data_tbl['SIZE'] = st.session_state.company_data_tbl['SIZE'].fillna('Unknown')
    st.session_state.company_data_tbl['COUNTRY_CODE'] = st.session_state.company_data_tbl['COUNTRY_CODE'].fillna('Unknown')
    st.session_state.company_data_tbl['CITY'] = st.session_state.company_data_tbl['CITY'].fillna('Unknown')
    st.session_state.company_data_tbl['STATE'] = st.session_state.company_data_tbl['STATE'].fillna('Unknown')
    st.session_state.company_data_tbl['TYPE'] = st.session_state.company_data_tbl['TYPE'].fillna('Unknown')
    st.session_state.company_data_tbl['INDUSTRY'] = st.session_state.company_data_tbl['INDUSTRY'].fillna('Unknown')

    # Rename country codes column to country
    st.session_state.company_data_tbl.rename(columns={'COUNTRY_CODE':'COUNTRY'}, inplace=True)
    # Convert the Snowflake dataframe to a pandas dataframe

    





    #st.session_state.company_data_tbl['FOUNDED'] = st.session_stat e.company_data_tbl['FOUNDED'].replace(0, pd.NA)

    # Uppercase Type and industry columns
    st.session_state.company_data_tbl['TYPE'] = st.session_state.company_data_tbl['TYPE'].str.upper()
    st.session_state.company_data_tbl['INDUSTRY'] = st.session_state.company_data_tbl['INDUSTRY'].str.upper()

talk, view, upload = st.tabs(["Me2B Bot", "Lead Dashboard", "Manage Products"])
if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = """You are an B2B business development representative bot called me2b. You are an expert on making matches between product data and customers.  
                        You have two main functions:
                        1. Based the potential company matches, you find the best matches between a product and a few potential customers. You can articulate why a company is a good lead for a product and what the potential value is for the company. 
                        2. You are also a master at writing an excellent, concise cold email or linkedin message to outreach to a company. You know how to make a great pitch and are best known for your ability to make a match between a product and a potential customer and writing concisely.

                        First, read through this product profile ('{description}'), these key features ('{features}') and this price ({price}).
                        Next, read this description of a fictional company persona that might want to use this product ({potential_customers}). 
                        Finally, evaluate potential company matches below and highlight product features that are a good match and come up with a few reasons why the product is a good fit for the company. Comapny persona matches will arrive in the next system message.
                        Make it seem like you found the potential company matches and are presenting them to the user.
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
        st.session_state.product_profile_json = list(dict(st.session_state.product_data_tbl[st.session_state.product_data_tbl["Product Name"] == curr_product]["Product Profile"]).values())[0]

        print(st.session_state.product_profile_json)
    

    curr_model = st.sidebar.selectbox("Select a model", ["GPT 3.5 Turbo", "Claude 3.0", "Mixtrial 7B", "LLAMA3 80B"], key="curr_model", help="Select a model to power company match generation and the resulting chat.")
    # Set up memory
    st.session_state.msgs_open_ai = StreamlitChatMessageHistory(key="langchain_messages_open_ai")
    st.session_state.msgs_claude = StreamlitChatMessageHistory(key="langchain_messages_claude")
    st.session_state.msgs_mistral = StreamlitChatMessageHistory(key="langchain_messages_mistral")
    st.session_state.msgs_llama3 = StreamlitChatMessageHistory(key="langchain_messages_llama3")
    if len(st.session_state.msgs_open_ai.messages) == 0:
        st.session_state.msgs_open_ai.add_ai_message(f"Hi! I'm **Me2B Bot**, powered by *GPT 3.5 Turbo*. Are you ready to find a new **{st.session_state.curr_prod}** customer?")
    if len(st.session_state.msgs_claude.messages) == 0:
        st.session_state.msgs_claude.add_user_message("...")
        st.session_state.msgs_claude.add_ai_message(f"Hi! I'm **Me2B Bot**, powered by *Claude 3.0*. Are you ready to find a new **{st.session_state.curr_prod}** customer?")
    if len(st.session_state.msgs_mistral.messages) == 0:
        st.session_state.msgs_mistral.add_ai_message(f"Hi! I'm **Me2B Bot**, powered by *Mixtral 7B*. Are you ready to find a new **{st.session_state.curr_prod}** customer?")
    if len(st.session_state.msgs_llama3.messages) == 0:
        st.session_state.msgs_llama3.add_ai_message(f"Hi! I'm **Me2B Bot**, powered by *Llama3*. Are you ready to find a new **{st.session_state.curr_prod}** customer?")

    
    # view_messages = st.expander("View the message contents in session state")
    st.title("Me2B Bot")
    ###########################
    ### OpenAI
    ###########################
    if curr_model == "GPT 3.5 Turbo":
        if 'open_ai_refined_queries' not in st.session_state or 'open_ai_matches' not in st.session_state:
            with st.spinner(f"Generating Buyer Profiles for {st.session_state.curr_prod}..."):
                st.session_state.open_ai_refined_queries = openAI_query_refiner(product_profile= st.session_state.product_profile_json, client= st.session_state.client)
                
                st.sidebar.subheader("Generated Potential Buyer Profiles")
                st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")  
            with st.spinner(f"Finding Matches for {st.session_state.curr_prod}..."):
                st.sidebar.write(st.session_state.open_ai_refined_queries)
                st.session_state.open_ai_matches = find_match_with_lang(st.session_state.open_ai_refined_queries, st.session_state.vectorstore)
                st.rerun()
                #st.session_state.open_ai_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
        else:
            st.sidebar.subheader("Generated Potential Buyer Profiles")
            st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")        
            st.sidebar.write(st.session_state.open_ai_refined_queries)
            if st.sidebar.button("Buzz"):
                del st.session_state.open_ai_refined_queries
                del st.session_state.open_ai_matches
                refresh_current_chat(st.session_state.curr_model)
                st.rerun()

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
            if msg.type == "ai":
                st.chat_message(msg.type,avatar=("me2b.png")).write(msg.content)
            else:
                st.chat_message(msg.type,avatar="👤").write(msg.content)

        # If user inputs a new prompt, generate and draw a new response 
        if prompt := st.chat_input():
            st.chat_message("human",avatar="👤").write(prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": prompt,
                                                 "features": st.session_state.product_profile_json["Features"],
                                                 "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                "price": st.session_state.product_profile_json["Price"],
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
            with st.spinner(f"Generating Buyer Profiles for {st.session_state.curr_prod}..."):
                st.session_state.claude_refined_queries = claude_query_refiner( product_profile= st.session_state.product_profile_json)
                st.sidebar.subheader("Generated Potential Buyer Profiles")
                st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")  
                st.sidebar.write(st.session_state.claude_refined_queries)
            with st.spinner(f"Finding Matches for {st.session_state.curr_prod}..."):
                st.session_state.claude_matches = find_match_with_lang(st.session_state.claude_refined_queries, st.session_state.vectorstore)
                #st.session_state.open_ai_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
                st.rerun()
        else:
            st.sidebar.subheader("Generated Potential Buyer Profiles")
            st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")  
            st.sidebar.write(st.session_state.claude_refined_queries)
            if st.sidebar.button("Buzz"):
                del st.session_state.claude_refined_queries
                del st.session_state.claude_matches 
                refresh_current_chat(st.session_state.curr_model)
                st.rerun()

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
            if msg.type == "ai":
                st.chat_message(msg.type,avatar=("me2b.png")).write(msg.content)
            else:
                st.chat_message(msg.type,avatar="👤").write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if claude_prompt := st.chat_input():
            st.chat_message("human",avatar="👤").write(claude_prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": claude_prompt,
                                                 "features": st.session_state.product_profile_json["Features"],
                                                 "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                "price": st.session_state.product_profile_json["Price"],
                                                 }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()


        # # Draw the messages at the end, so newly generated ones show up immediately
        # with view_messages:
        #     view_messages.json(st.session_state.langchain_messages_claude)
    
    ###########################
    ### mistral
    ###########################
    elif curr_model == "Mixtrial 7B":
        if 'mistrial_refined_queries' not in st.session_state or 'mistrial_matches' not in st.session_state:
            with st.spinner(f"Generating Buyer Profiles for {st.session_state.curr_prod}..."):
                st.session_state.mistrial_refined_queries = mistral_query_refiner( product_profile= st.session_state.product_profile_json)
                st.sidebar.subheader("Generated Potential Buyer Profiles")
                st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")  
                st.sidebar.write(st.session_state.mistrial_refined_queries)
            with st.spinner(f"Finding Matches for {st.session_state.curr_prod}..."):
                st.session_state.mistrial_matches = find_match_with_lang(st.session_state.mistrial_refined_queries, st.session_state.vectorstore)
                #st.session_state.mistrial_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
                st.rerun()
        else:
            st.sidebar.subheader("Generated Potential Buyer Profiles")
            st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")          
            st.sidebar.write(st.session_state.mistrial_refined_queries)
            if st.sidebar.button("Buzz"):
                del st.session_state.mistrial_refined_queries
                del st.session_state.mistrial_matches 
                refresh_current_chat(st.session_state.curr_model)
                st.rerun()

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
            if msg.type == "ai":
                st.chat_message(msg.type,avatar=("me2b.png")).write(msg.content)
            else:
                st.chat_message(msg.type,avatar="👤").write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if mixtrial_prompt := st.chat_input():
            st.chat_message("human",avatar="👤").write(mixtrial_prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": mixtrial_prompt,
                                                 "features": st.session_state.product_profile_json["Features"],
                                                 "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                "price": st.session_state.product_profile_json["Price"],
                                                 }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()
        
        ###########################      
        ### LLaMa3
        ###########################
    elif curr_model == "LLAMA3 80B":
        if 'llama3_matches' not in st.session_state or 'llama3_refined_queries' not in st.session_state:
            with st.spinner(f"Generating Buyer Profiles for {st.session_state.curr_prod}..."):
                st.session_state.llama3_refined_queries = llama_query_refiner(product_profile= st.session_state.product_profile_json)
                st.sidebar.subheader("Generated Potential Buyer Profiles")
                st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")  
                st.sidebar.write(st.session_state.llama3_refined_queries)
            with st.spinner(f"Finding Matches for {st.session_state.curr_prod}..."):
                st.session_state.llama3_matches = find_match_with_lang(st.session_state.llama3_refined_queries, st.session_state.vectorstore)
                #st.session_state.mistrial_matches = find_match_with_pinecone(refined_queries, st.session_state.pc_company_idx)
                st.rerun()
        else:
            st.sidebar.subheader("Generated Potential Buyer Profiles")
            st.sidebar.warning("""To match your product to a company, we use your product data to generate a variety of potential buyer profiles.\n
                                            \nHit the **buzz** button to create new ones.""")      
            st.sidebar.write(st.session_state.llama3_refined_queries)
            if st.sidebar.button("Buzz"):
                del st.session_state.llama3_refined_queries 
                del st.session_state.llama3_matches 
                refresh_current_chat(st.session_state.curr_model)
                st.rerun()

        llama_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", st.session_state.prompt_template),
                "system", f"Here are the potential company persona matches: {st.session_state.llama3_matches}",
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        
        chain = llama_prompt | ChatGroq(api_key= groq_api_key , temperature=0, model_name="llama3-70b-8192")
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: st.session_state.msgs_llama3,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Render current messages from StreamlitChatMessageHistory
        for msg in st.session_state.msgs_llama3.messages:
            if msg.type == "ai":
                st.chat_message(msg.type,avatar=("me2b.png")).write(msg.content)
            else:
                st.chat_message(msg.type,avatar="👤").write(msg.content)

        # If user inputs a new prompt, generate and draw a new response
        if llama_prompt := st.chat_input():
            st.chat_message("human",avatar="👤").write(llama_prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"question": llama_prompt,
                                                "features": st.session_state.product_profile_json["Features"],
                                                "description": st.session_state.product_profile_json["Description"],
                                                "pitch":  st.session_state.product_profile_json["Sales pitch"],
                                                "potential_customers": st.session_state.product_profile_json["Product persona"],
                                                "price": st.session_state.product_profile_json["Price"],
                                                }, config)
            st.chat_message("ai").write(response.content)
            st.rerun()
   
    restart_chat = st.button("Restart Chat")
    if restart_chat:
        refresh_current_chat(st.session_state.curr_model)
        st.rerun()


with view:
    
    def pass_new_match_to_curr_chat(new_matches):
        if st.session_state.curr_model == "GPT 3.5 Turbo":
            st.session_state.msgs_open_ai.add_user_message(f"""I found company data from the Me2B Lead Dashboard - help me assess them as new matches for {st.session_state.curr_prod}\n
                                                        
                                                          \n {new_matches}""" )
            st.session_state.msgs_open_ai.add_ai_message("Let's chat about them - ask me a question!")
        elif st.session_state.curr_model == "Claude 3.0":
            st.session_state.msgs_claude.add_user_message(f"""I found company data from the Me2B Lead Dashboard - help me assess them as new matches for {st.session_state.curr_prod}\n
                                                        
                                                          \n {new_matches}""" )
            st.session_state.msgs_claude.add_ai_message("Let's chat about them - ask me a question!")            
        elif st.session_state.curr_model == "Mixtrial 7B":
            st.session_state.msgs_mistral.add_user_message(f"""I found company data from the Me2B Lead Dashboard - help me assess them as new matches for {st.session_state.curr_prod}\n
                                                        
                                                          \n {new_matches}""" )
            st.session_state.msgs_mistral.add_ai_message("Let's chat about them - ask me a question!")            
        elif st.session_state.curr_model == "LLAMA3 80B":
            st.session_state.msgs_llama3.add_user_message(f"""I found company data from the Me2B Lead Dashboard - help me assess them as new matches for {st.session_state.curr_prod}\n
                                                        
                                                          \n {new_matches}""" )
            st.session_state.msgs_llama3.add_ai_message("Let's chat about them - ask me a question!")            
        st.rerun()


        
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        # if df has a column named "SELECT", drop it
        if "SELECT" in df.columns:
            df = df.drop("SELECT", axis=1)

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container(height=0)

        with modification_container:
            to_filter_columns = st.multiselect("", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = int(df[column].min())
                    _max = int(df[column].max())
                    step = 1
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]

        return df
    
    def dataframe_with_selections(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "SELECT", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = st.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"SELECT": st.column_config.CheckboxColumn(required=True)},
            disabled=df.columns, use_container_width=True
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.SELECT]
        return selected_rows.drop('SELECT', axis=1)
    
    dashboard = st.container()
    dashboard.title("Lead Dashboard")
    dashboard.warning("Analyze Me2B's entire pool of company data to **find the right leads for your product.**")
    st.divider()
    st.header("Filter & Share with Me2B Bot")
    st.warning("Filter by company attributes.  \n\nShare leads that match your ICP with *Me2B Bot* to **generate the right content for the right companies** :bee:.")
    filtered_df = (filter_dataframe(st.session_state.company_data_tbl[['NAME','TYPE','INDUSTRY','SIZE','FOUNDED','COUNTRY','CITY','STATE', 'UID']]))

    selected_companies = dataframe_with_selections(filtered_df)

    with dashboard:
        type, size = st.columns(2)
        industry, founding = st.columns(2)

        with type:

            # Visualization 1: Bar chart for company types
            type_counts = filtered_df['TYPE'].value_counts().reset_index()
            type_counts.columns = ['TYPE', 'COUNT']
            fig1 = px.bar(type_counts, x='TYPE', y='COUNT', title="Distribution by Company Type")
            #change the color of the bar chart
            fig1.update_traces(marker_color=st.session_state.color_dict['light_yellow'])
            st.plotly_chart(fig1, use_container_width=True)

        with size:
            # Visualization 2: Distribution of company sizes
            fig2 = px.bar(filtered_df, x='SIZE', title="Distribution by Company Size")
            # update y-axis label
            fig2.update_yaxes(title_text='COUNT')
            fig2.update_traces(marker_color=st.session_state.color_dict['light_yellow'])
            st.plotly_chart(fig2, use_container_width=True)

        with industry:
            # Visualization 3: Distribution of companies by industry
            industry_counts = filtered_df['INDUSTRY'].value_counts().reset_index().head(10)
            industry_counts.columns = ['INDUSTRY', 'COUNT']
            fig3 = px.bar(industry_counts, x='INDUSTRY', y='COUNT', title=" Distribution by Industry")
            fig3.update_traces(marker_color=st.session_state.color_dict['light_yellow'])
            st.plotly_chart(fig3, use_container_width=True)

        with founding:
            # Visualization 4: Scatter plot of company founding year vs. size
            founding_counts = filtered_df['FOUNDED'].value_counts().reset_index().head(10)
            founding_counts.columns = ['FOUNDED', 'COUNT']
            fig4 = px.bar(founding_counts, x='FOUNDED', y='COUNT', title="Distribution by Founding Year")
            fig4.update_traces(marker_color=st.session_state.color_dict['light_yellow'])
            st.plotly_chart(fig4, use_container_width=True)
    # Create a pie chart for the distribution of country
    
        # Visualization 5: Distribution of companies by country
        country_counts = filtered_df['COUNTRY'].value_counts().reset_index().head(10)
        country_counts.columns = ['COUNTRY', 'COUNT']
        fig5 = px.pie(country_counts, values='COUNT', names='COUNTRY', title="Distribution by Country", hole=.65)
        fig5.update_traces(marker_colors=px.colors.sequential.YlOrBr_r)
        st.plotly_chart(fig5, use_container_width=True)


    if st.button("Share with Me2B Bot :bee:"):

        if selected_companies.empty:
            st.error("Please select companies to share with Me2B Bot :bee:")
            st.stop()

        # Get the list of UIDs of the selected companies
        selected_uids = selected_companies['UID'].tolist()

        # Lookup the company names from the UIDs
        selected_companies = st.session_state.company_data_tbl[st.session_state.company_data_tbl['UID'].isin(selected_uids)]
        
        # Get the personas of the selected companies from the persona table and add them to a string. Separate each persona with a newline character
        selected_personas = st.session_state.persona_data_tbl[st.session_state.persona_data_tbl['UID'].isin(selected_uids)]
        # Get the persona column from the selected personas
        selected_personas = selected_personas['PERSONA'].str.cat(sep='\n\n\n')
        st.success("Company data shared with Me2B Bot")
        time.sleep(1)
        pass_new_match_to_curr_chat(selected_personas)
        st.rerun()
    

def stream_json(json_data):
    
    
    for row in json_data:
        if row == 'Product_name':
            row = 'Product name'
        if row == 'Company_name':
            row = 'Company name'
        yield ('**'+row + '**:' +'\n')
        if row == 'Product name':
            row = 'Product_name'
        if row == 'Company name':
            row = 'Company_name' 
        if json_data[row] == None:
            yield('Not found')
        else:
            for token in json_data[row].split():
                yield(' '+token)
                time.sleep(0.1)
        yield('\n\n')
    
        
        

with upload:
    
    st.title('Upload Product Data')
    st.warning("Here's a peak at products that have been already been loaded :bee:. \n\nUpload a PDF to **add a new product** to your colony.")
    st.dataframe(st.session_state.product_data_tbl[['Company Name','Product Name']], use_container_width=True, hide_index=True)

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    files = st.file_uploader(
        "Upload a PDF file",
        type="pdf",
        key=st.session_state["file_uploader_key"],
    )

    if files:
        # Read PDF file
        reader = PdfReader(files)
        num_pages = len(reader.pages)
        st.divider()

        # Extract text from each page and write to a file
        with st.spinner('Extracting key data from your Product PDF...'):
            with open('product.txt', 'w', encoding='utf-8') as p:
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    p.write(f"{text}\n\n")
        
            # Read extracted text
            with open('product.txt', 'r', encoding='utf-8') as p:
                product = p.read()
                
            try:
                data = parsepdf(product,client= st.session_state.client)
                os.remove('product.txt')
                st.subheader(f"Here's the honey we squeezed out of your {num_pages}-page PDF :honey_pot:")
                
            except Exception as e:
                st.error(f"Error parsing PDF: {e}")

            st.write_stream(stream_json(data))
            st.session_state.flag = False
            st.success("Product data uploaded successfully!")
            time.sleep(7)
            st.session_state.flag = True

    if 'flag' in st.session_state:
        st.session_state["file_uploader_key"] += 1
        del st.session_state["flag"]
        del st.session_state.product_data_tbl
        st.rerun()

        
    

            
            


        
