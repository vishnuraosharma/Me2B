import Product_Model.product

class ProductCatalog:
    def __init__(self):
        self.products = []

    def add_product(self, product):
        self.products.append(product)

    def delete_product(self, product_id):
        for product in self.products:
            if product.id == product_id:
                self.products.remove(product)
                break

    def get_product_by_id(self, product_id):
        for product in self.products:
            if product.id == product_id:
                return product

        return None