from airflow import DAG
from airflow.contrib.operators.snowflake_operator import SnowflakeOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import requests
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
import langchain
import os

#Set up your Apify API token and OpenAI API key
os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"
os.environ["APIFY_API_TOKEN"] = "Your Apify API token"

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'snowflake_company_tasks',
    default_args=default_args,
    description='Perform tasks on new company additions in Snowflake',
    schedule_interval=timedelta(minutes=30),  # Adjust as per your requirement
)

# Function to fetch data from BuiltWith's Domain API
def fetch_technographics(website):
    # Perform API call to BuiltWith's Domain API
    # Assuming you have an API key stored securely
    api_key = 'YOUR_API_KEY'
    url = f'https://api.builtwith.com/v14/api.json?KEY={api_key}&LOOKUP={website}'
    response = requests.get(url)
    data = response.json()
    technographics = data.get('Results', [])
    return technographics

# Function to scrape website for basic description
def scrape_website(website):
    apify = ApifyWrapper()
    
    # Call the Actor to obtain text from the crawled webpages
    #Run the Website Content Crawler on a website, wait for it to finish, and save
    #its results into a LangChain document loader:
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={
            "startUrls": [{"url": website}]
        },
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )

    # Create a vector store based on the crawled data
    index = VectorstoreIndexCreator().from_loaders([loader])

    # Query the vector store
    query = "give a basic description of this company"
    result = index.query(query)
    
    return result

# Task to fetch technographics
fetch_technographics_task = PythonOperator(
    task_id='fetch_technographics',
    python_callable=fetch_technographics,
    provide_context=True,
    dag=dag,
)

# Task to scrape website for basic description
scrape_website_task = PythonOperator(
    task_id='scrape_website',
    python_callable=scrape_website,
    provide_context=True,
    dag=dag,
)

# Sensor to listen for new data in Snowflake table
snowflake_sensor = SnowflakeOperator(
    task_id='snowflake_sensor',
    snowflake_conn_id='snowflake_default',  # Snowflake connection ID configured in Airflow
    sql="SELECT COUNT(*) FROM COMPANIES WHERE created_at > (CURRENT_TIMESTAMP() - INTERVAL '5' MINUTE)",  # Check for new data added in the last 5 minutes
    mode='poke',
    timeout=600,  # Timeout after 10 minutes
    dag=dag,
)

# Define Snowflake operators to insert data into respective tables
insert_technographics_task = SnowflakeOperator(
    task_id='insert_technographics',
    sql="INSERT INTO company_technographics (website, found_technographics, createDate) VALUES (%s, %s, %s)",
    snowflake_conn_id='snowflake_default',
    parameters=("{{ task_instance.xcom_pull(task_ids='fetch_technographics') }}"),
    autocommit=True,
    dag=dag,
)

insert_ai_descriptions_task = SnowflakeOperator(
    task_id='insert_ai_descriptions',
    sql="INSERT INTO company_ai_descriptions (website, basic_description, createDate) VALUES (%s, %s, %s)",
    snowflake_conn_id='snowflake_default',
    parameters=("{{ task_instance.xcom_pull(task_ids='scrape_website') }}"),
    autocommit=True,
    dag=dag,
)

# Define task dependencies
snowflake_sensor >> fetch_technographics_task >> insert_technographics_task
snowflake_sensor >> scrape_website_task >> insert_ai_descriptions_task
