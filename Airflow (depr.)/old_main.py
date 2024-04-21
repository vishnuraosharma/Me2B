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
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer




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


# Serialize the product catalog to a json string
product_cat_json = json.dumps(product_cat, default=lambda o: o.__dict__, indent=2)

# Get the absolute path of the current script file
script_path = os.path.abspath(os.getcwd() + os.sep + os.pardir)

# Construct the path to the .env file
dotenv_path = join(script_path, '.env')
load_dotenv(dotenv_path)

# Get the credentials from .env
connection_parameters = {
"account"    : os.getenv('SNOWFLAKE_ACCOUNT'),
"user"       : os.getenv('SNOWFLAKE_USER'),
"warehouse"  : os.getenv('SF_WAREHOUSE'),
"database"   : os.getenv('TEST_DB'),
"password"   : os.getenv('SNOWFLAKE_PASSWORD')
}

if 'fakedata' not in st.session_state:
    # fire up an instance of a snowflake connection
    st.session_state.new_session = Session.builder.configs(connection_parameters).create()

    # Read docs at https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes#label-snowpark-python-dataframe-construct for more
    df_table = st.session_state.new_session.table("TEST_DB.SYNTHETIC_DATA.SAMPLE_MATCHES_LORIS")

    # Convert the Snowflake dataframe to a pandas dataframe
    st.session_state.fakedata = df_table.toPandas()

    # Rename the columns
    st.session_state.fakedata.columns = ['Company', 'Product Match','Match Strength', 'Vertical', 'Competitors', 'Integrations']
    st.session_state.row = 1
        # drop the first row
    st.session_state.fakedata = st.session_state.fakedata.drop(0)



def start_new_thread():
    """
    Reset the chat history to start a new thread.
    This function clears the existing messages and adds a new initial message from the assistant.
    """
    st.session_state.start_chat = True
    thread = st.session_state.client.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.messages = []
    

talk, view, upload = st.tabs(["Me2B Bot", "View Leads", "Upload Product Catalog"])
with talk:
    if 'fakedata' not in st.session_state:
        st.session_state.fakedata = pd.DataFrame(csv.reader(open('faker/fake_matches.csv', 'r')), columns = ['Company', 'Product Match','Match Strength', 'Vertical', 'Competitors', 'Integrations'])
        st.session_state.row = 1
        # drop the first row
        st.session_state.fakedata = st.session_state.fakedata.drop(0)
    if 'company' not in st.session_state:
        if 'productmatch' not in st.session_state:
            if 'vertical' not in st.session_state:
                if 'competitors' not in st.session_state:
                    if 'integrations' not in st.session_state:
                        # read the faker data from csv as pandas df
                        print("first row")
                        st.session_state.company = (st.session_state.fakedata)['Company'][st.session_state.row]
                        st.session_state.productmatch = (st.session_state.fakedata)['Product Match'][st.session_state.row]
                        st.session_state.vertical = (st.session_state.fakedata)['Vertical'][st.session_state.row]
                        st.session_state.competitors = (st.session_state.fakedata)['Competitors'][st.session_state.row].split(',')
                        st.session_state.integrations = (st.session_state.fakedata)['Integrations'][st.session_state.row].split(',')
                        st.session_state.match_strength = int((st.session_state.fakedata)['Match Strength'][st.session_state.row])

    # Populate the left column with information
    st.sidebar.title('Lead Profile')
    # Add more widgets to show the details like ID, Agent Name, etc.
    st.sidebar.subheader(f'Company Match')
    st.sidebar.write(st.session_state.company)
    st.sidebar.empty()
    st.sidebar.subheader(f'Product Match')
    st.sidebar.write(st.session_state.productmatch)
    st.sidebar.empty()
    st.sidebar.subheader(f'Match Strength')
    st.sidebar.slider('', 1, 5, st.session_state.match_strength)
    st.sidebar.empty()
    st.sidebar.subheader(f'Vertical')
    st.sidebar.write(st.session_state.vertical)
    st.sidebar.empty()
    st.sidebar.subheader(f'Competitors')
    st.sidebar.write(st.session_state.competitors)
    st.sidebar.empty()
    st.sidebar.subheader(f'Integrations')
    st.sidebar.write(st.session_state.integrations)


    # Add a next button
    if st.sidebar.button(label='Next'):
        start_new_thread()
        st.write(st.session_state.row)
        st.session_state.row = st.session_state.row + 1
        # Read faker data from csv in faker folder
        st.session_state.company = (st.session_state.fakedata)['Company'][st.session_state.row]
        st.session_state.productmatch = (st.session_state.fakedata)['Product Match'][st.session_state.row]
        st.session_state.vertical = (st.session_state.fakedata)['Vertical'][st.session_state.row]
        st.session_state.competitors = (st.session_state.fakedata)['Competitors'][st.session_state.row].split(',')
        st.session_state.integrations = (st.session_state.fakedata)['Integrations'][st.session_state.row].split(',')
        st.session_state.match_strength = int((st.session_state.fakedata)['Match Strength'][st.session_state.row])
        st.rerun()

    # Add a previous button
    if st.sidebar.button(label='Previous'):
        start_new_thread()
        if st.session_state.row > 0:
            st.session_state.row = st.session_state.row - 1
        # Read faker data from csv in faker folder
        st.session_state.company = (st.session_state.fakedata)['Company'][st.session_state.row]
        st.session_state.productmatch = (st.session_state.fakedata)['Product Match'][st.session_state.row]
        st.session_state.vertical = (st.session_state.fakedata)['Vertical'][st.session_state.row]
        st.session_state.competitors = (st.session_state.fakedata)['Competitors'][st.session_state.row].split(',')
        st.session_state.integrations = (st.session_state.fakedata)['Integrations'][st.session_state.row].split(',')
        st.session_state.match_strength = int((st.session_state.fakedata)['Match Strength'][st.session_state.row])
        st.rerun()


    # Get the absolute path of the current script file
    script_path = os.path.abspath(os.getcwd() + os.sep + os.pardir)

    print (script_path)
    # Construct the path to the .env file
    dotenv_path = join(script_path, '.env')
    load_dotenv(dotenv_path)
    #Set OpenAI API key from Streamlit secrets
    # Initialize OpenAI client with your API key
    if 'client' not in st.session_state:
        st.session_state.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    


    assistant_id = "asst_CcvPt3JAQEe2jUsIITLK3aJo"

    if "start_chat" not in st.session_state:
        st.session_state.start_chat = True
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None


    st.title("Me2B Bot")
    st.write("Your helpful outreach bot.")


    if st.session_state.start_chat:
        if "openai_model" not in st.session_state:
            st.session_state.openai_model = "gpt-4-1106-preview"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Who would you like to contact?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.client.beta.threads.messages.create(
                    thread_id=st.session_state.thread_id,
                    role="user",
                    content=prompt
                )
            
            run = st.session_state.client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_id,
                instructions=f"""You are an B2B business development representative bot. You are an expert on all of Loris products and services 
                                in your knowledge base and know how to make an excellent, concise cold email or linkedin message to outreach to a company.

                                Read through Loris's product profile {product_cat_json}. Mention Loris' competitors {st.session_state.competitors} in your response and why Loris is a better solution.

                                Use the information to write a cold email or linkedin message to the company {st.session_state.company} to pitch a product related to {st.session_state.productmatch}. Please mention a specific integration in this list {st.session_state.integrations[0]} in your message.

                                For each prospect, you will think of 3 angles to message someone at the prospect's company based on the ICP below. One for a low-level employee, one for middle-management, and one for the c-suite. Based on user input, you will craft the message.
                                """
            )

            while run.status != 'completed':
                time.sleep(1)
                run = st.session_state.client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id
                )
            messages = st.session_state.client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id
            )

            # Process and display assistant messages
            assistant_messages_for_run = [
                message for message in messages 
                if message.run_id == run.id and message.role == "assistant"
            ]
            for message in assistant_messages_for_run:
                st.session_state.messages.append({"role": "assistant", "content": message.content[0].text.value})
                with st.chat_message("assistant"):
                    st.markdown(message.content[0].text.value)

    
    if st.button(label="Restart Chat"):
        st.session_state.start_chat = True
        thread = st.session_state.client.beta.threads.create()
        st.session_state.thread_id = thread.id
        st.session_state.messages = []
        st.rerun()

st.write(st.session_state)

with view:
    st.title('View Leads')
    st.dataframe(
        st.session_state.fakedata,
        column_config={
            "Company": "Company",
            "Match Strength": st.column_config.NumberColumn(
                "Match Strength",
                format="%d ⭐",
            ),
        },
        hide_index=True,
    )

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

with upload:
    
    st.title('Upload Product Data')
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


   

# {
#   "products": [
#     {
#       "id": 1,
#       "name": "Automated Agent QA",
#       "vertical": "Customer Service",
#       "description": "Automated Agent QA helps customer service teams automate their QA process by processing conversations with AI to understand if agents did a good job sticking to policy and resolving customer issues across all contacts. \n                         It uses AI to analyze customer service interactions over chat, email, or voice and provide insights to provide direct feedback to agents improve customer service. Automated QA automates the time-consuming parts of the agent performance review process, so you have more time for higher value work like conversational analysis and agent coaching. \n                         ",
#       "category": "Conversation Intelligence",
#       "competitor_products": [
#         "Maestro QA",
#         "Playvox",
#         "Klaus"
#       ],
#       "integrations": [
#         "Salesforce",
#         "Zendesk",
#         "Intercom",
#         "Gorgias",
#         "Khoros",
#         "Gladly",
#         "Five9"
#       ],
#       "features": [
#         "Automate interaction selection using AI to highlight and score the most impactful customer conversations",
#         "Agent Performance Review",
#         "Conduct more reviews of customer interactions without having to listen to entire phone calls or comb through transcripts",
#         "Easily pinpoint relevant parts of the conversation using the Loris Sentiment Graph to focus QA analyst reviews",
#         "Conversational Analysis",
#         "Automatically complete agent scorecards for effective communication skills, helping QA analysts do more reviews on more agents with less effort. Easily tailor your workflows, scorecards, and assignments based on what\u2019s important to your customer experience. Simplify everything QA \u2013 combining customer interactions, scorecards, agent coaching, and disputes into one platform",
#         "Use AI to create objective scoring across all your agents \u2013 and across all your channels \u2013 for more uniform evaluation. Spot trends and insights at the agent, team, and organizational level Use conversational insights to remove customer friction, improve your products, and develop new offerings",
#         "Agent Coaching"
#       ],
#       "audience": {
#         "description": "Companies with a strong digital presence that pride themselves in Customer Service.",
#         "region": "United States",
#         "teamsize": {
#           "min": 20,
#           "max": 50000
#         },
#         "tech_stack": [
#           "Salesforce",
#           "Zendesk",
#           "Intercom",
#           "Gorgias",
#           "Khoros",
#           "Gladly",
#           "Five9"
#         ],
#         "annual_revenue": {
#           "min": 10000000,
#           "max": 100000000
#         },
#         "verticals": [
#           "B2B",
#           "Customer Service",
#           "e-Commerce",
#           "BPO",
#           "SaaS",
#           "Retail",
#           "Healthcare",
#           "Telecom",
#           "Financial Services",
#           "Logistics"
#         ]
#       }
#     }
#   ]
# }

    
    
    
        
#         # Extract produc