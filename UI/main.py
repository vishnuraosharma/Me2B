import streamlit as st
import Product_Model.product as p
import Product_Model.product_catalog as pc

# Create a sample SAAS product and add it to the product catalog
product_cat = pc.ProductCatalog()
saas_product = p.Product("My SaaS Product", 100, "A sample SaaS product", "SaaS", [], [], [])
product_cat.add_product(saas_product)

def add_product(product_name):
    # Add the product to the database or perform any other necessary actions
    st.write(f"Added product: {product_name}")

def remove_product(product_name):
    # Remove the product from the database or perform any other necessary actions
    st.write(f"Removed product: {product_name}")

st.title("Product Management App")

# Sidebar
st.sidebar.header("Actions")
action = st.sidebar.selectbox("Select an action", ["Add Product", "Remove Product"])

if action == "Add Product":
    product_name = st.text_input("Enter product name")
    if st.button("Add"):
        add_product(product_name)

elif action == "Remove Product":
    product_name = st.text_input("Enter product name")
    if st.button("Remove"):
        remove_product(product_name)

# Display all products in st.table
st.header("Product Catalog")
product_data = []
for product in product_cat.products:
    product_data.append([product.id, product.name, product.price, product.description, product.category])
st.table(product_data)