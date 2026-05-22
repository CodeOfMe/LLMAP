import json

def search_product(query, category=None):
    products = {
        "phone_case": {"name": "Clear Plastic Phone Case", "price": 49, "stock": 23},
        "bluetooth_speaker": {"name": "Portable Bluetooth Speaker", "price": 299, "stock": 100},
    }
    for key, product in products.items():
        if query in key:
            return product
    return {"error": "No matching product found"}

if __name__ == "__main__":
    result1 = search_product("phone_case")
    print(f"# search_product(\"phone_case\") local result:\n{json.dumps(result1, ensure_ascii=False)}")
    result2 = search_product("noise_cancelling_headphones")
    print(f"\n# search_product(\"noise_cancelling_headphones\") local result:\n{json.dumps(result2, ensure_ascii=False)}")