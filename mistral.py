from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
import joblib
from geopy.distance import geodesic
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the HUGGINGFACE_API_KEY
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
login(token = huggingface_api_key)

app = FastAPI()

# Load models
xgb_model = joblib.load("xgboost_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")

# Load Mistral-7B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
genai_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# MySQL connection
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "4102",
    "database": "pot"
}

class UserLocation(BaseModel):
    latitude: float
    longitude: float
    n_clusters: int
    min_distance_km: float

@app.post("/analyze-impact")
async def analyze_impact(user_loc: UserLocation):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT id, name, latitude, longitude, capacity, orders_served, traffic_density
    FROM store
    """
    cursor.execute(query)
    stores = cursor.fetchall()
    conn.close()

    nearby_stores = []
    for store in stores:
        distance = geodesic((user_loc.latitude, user_loc.longitude), 
                            (store['latitude'], store['longitude'])).km
        if distance <= user_loc.min_distance_km:
            store['distance_km'] = distance
            nearby_stores.append(store)

    # Predict delivery times for nearby stores
    predictions = []
    for store in nearby_stores:
        features = [[store['distance_km'], store['orders_served'], store['traffic_density'], store['capacity']]]
        predicted_time = xgb_model.predict(features)[0]
        predictions.append({"store_id": store['id'], "predicted_time": predicted_time})

    # Use clustering to suggest new store locations
    store_coords = [(store['latitude'], store['longitude']) for store in nearby_stores]
    kmeans_model.set_params(n_clusters=user_loc.n_clusters)
    kmeans_model.fit(store_coords)
    new_store_coords = kmeans_model.cluster_centers_.tolist()

    # Generate business insights with Mistral
    insights_prompt = f"""
    A user clicked at coordinates ({user_loc.latitude}, {user_loc.longitude}).
    The 3 nearest stores would take the following times: {predictions}.
    If we open new stores at {new_store_coords}, how would this improve delivery times, store loads, and customer satisfaction?
    Provide a business-level summary.
    """

    insight_response = genai_pipeline(insights_prompt, max_length=500, num_return_sequences=1)
    business_insights = insight_response[0]['generated_text']

    return {
        "nearby_stores": nearby_stores,
        "predicted_delivery_times": predictions,
        "suggested_new_store_locations": new_store_coords,
        "business_insights": business_insights
    }

# With this setup, you can run Mistral-7B locally (if you have enough resources) or through an API.
# Let me know if you want me to guide you on deploying the whole system! 🚀