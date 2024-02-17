import streamlit as st
import Product_Model.product as p
import Product_Model.product_catalog as pc
import Product_Model.product_audience as pa
import json
import pickle


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

print (product_cat_json)
# Beautify the json string


# Dump the json of the product catalog
st.write(product_cat_json)


# def add_product(product_name):
#     # Add the product to the database or perform any other necessary actions
#     st.write(f"Added product: {product_name}")

# def remove_product(product_name):
#     # Remove the product from the database or perform any other necessary actions
#     st.write(f"Removed product: {product_name}")

# st.title("Product Management App")

# # Sidebar
# st.sidebar.header("Actions")
# action = st.sidebar.selectbox("Select an action", ["Add Product", "Remove Product"])

# if action == "Add Product":
#     product_name = st.text_input("Enter product name")
#     if st.button("Add"):
#         add_product(product_name)

# elif action == "Remove Product":
#     product_name = st.text_input("Enter product name")
#     if st.button("Remove"):
#         remove_product(product_name)

# # Display all products in st.table
# st.header("Product Catalog")
# product_data = []
# for product in product_cat.products:
#     product_data.append([product.id, product.name, product.price, product.description, product.category])
# st.table(product_data)