from airflow import DAG
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from datetime import datetime, timedelta
import requests
from airflow.operators.python import PythonOperator  # Add this line to import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
 
dag = DAG(
    'fetch_company_data_and_insert_into_snowflake',
    default_args=default_args,
    description='A DAG to fetch data about companies from an API based on their domain and insert into Snowflake',
    schedule_interval=timedelta(days=1),
)
 
def fetch_data_from_api(domain):
    # Replace this with your actual API endpoint
    url = f'https://api.example.com/company/{domain}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
 
def insert_into_snowflake(**kwargs):
    ti = kwargs['ti']
    domains = ti.xcom_pull(task_ids='fetch_records')
    for domain in domains:
        company_data = fetch_data_from_api(domain)
        if company_data:
            # Insert fetched data into Snowflake table
            sql = f"INSERT INTO enriched_domains (domain, data) VALUES ('{domain}', {json.dumps(company_data)})"
            snowflake_hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')  # Assuming you have a connection setup in Airflow
            snowflake_hook.run(sql)
        else:
            print(f"Failed to fetch data for domain: {domain}")
 
with dag:
    # Assuming you already have a connection setup in Airflow to Snowflake
    fetch_records_task = SnowflakeOperator(
        task_id='fetch_records',
        sql='SELECT domain FROM company WHERE created_at >= DATEADD(DAY, -1, CURRENT_DATE())',
        snowflake_conn_id='snowflake_conn',  # Assuming you have a connection setup in Airflow
        autocommit=True,
    )
 
    create_table_task = SnowflakeOperator(
        task_id='create_table',
        sql='CREATE TABLE IF NOT EXISTS enriched_domains (domain STRING, data VARIANT)',
        snowflake_conn_id='snowflake_conn',
        autocommit=True,
    )
 
    insert_into_snowflake_task = PythonOperator(
        task_id='insert_into_snowflake',
        python_callable=insert_into_snowflake,
    )
 
    fetch_records_task >> create_table_task >> insert_into_snowflake_task