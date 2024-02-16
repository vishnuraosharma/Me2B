import Product_Model.product_audience as pa

class Product:

    id_counter = 1

    def __init__(self, name,  description, category, vertical, competitors, integrations, features, audience):
        self.id = Product.id_counter
        self.name = name
        self.vertical = vertical
        self.description = description
        self.category = category
        self.competitor_products = competitors
        self.integrations = integrations
        self.features = features
        self.audience = audience

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_description(self):
        return self.description

    def set_description(self, description):
        self.description = description

    def get_category(self):
        return self.category

    def set_category(self, category):
        self.category = category

    def __str__(self):
        return f"Product: {self.name}\nPrice: {self.price}\nDescription: {self.description}\nCategory: {self.category}"
