from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector
import joblib
from geopy.distance import geodesic
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

# Login to Hugging Face Hub
if huggingface_api_key:
    login(token=huggingface_api_key)
else:
    raise ValueError("Missing Hugging Face API Key!")

app = FastAPI()

# Load models
try:
    xgb_model = joblib.load("xgboost_model.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", load_in_8bit=True)
    genai_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# MySQL connection config
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "4102",
    "database": "pot"
}

# Request model class
class UserLocation(BaseModel):
    latitude: float
    longitude: float
    n_clusters: int
    min_distance_km: float

# API endpoint
@app.post("/analyze-impact")
async def analyze_impact(user_loc: UserLocation):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Fetch store data
        query = """
        SELECT id, name, latitude, longitude, capacity, orders_served, traffic_density
        FROM store
        """
        cursor.execute(query)
        stores = cursor.fetchall()
    
    except mysql.connector.Error as err:
        return {"error": f"Database error: {err}"}
    finally:
        cursor.close()
        conn.close()

    # Find nearby stores
    nearby_stores = [
        {**store, 'distance_km': geodesic((user_loc.latitude, user_loc.longitude), (store['latitude'], store['longitude'])).km}
        for store in stores
        if geodesic((user_loc.latitude, user_loc.longitude), (store['latitude'], store['longitude'])).km <= user_loc.min_distance_km
    ]

    # Predict delivery times
    predictions = [
        {"store_id": store['id'], "predicted_time": xgb_model.predict([[store['distance_km'], store['orders_served'], store['traffic_density'], store['capacity']]])[0]}
        for store in nearby_stores
    ]

    # Cluster analysis for new store locations
    store_coords = [(store['latitude'], store['longitude']) for store in nearby_stores]
    if store_coords:
        kmeans_model.set_params(n_clusters=user_loc.n_clusters)
        kmeans_model.fit(store_coords)
        new_store_coords = kmeans_model.cluster_centers_.tolist()
    else:
        new_store_coords = []

    # Generate business insights with Mistral
    insights_prompt = f"""
    A user clicked at coordinates ({user_loc.latitude}, {user_loc.longitude}).
    The nearest stores have predicted delivery times: {predictions}.
    If new stores are opened at {new_store_coords}, how would this impact delivery times, store loads, and customer satisfaction?
    Provide a detailed business-level summary.
    """
    
    try:
        insight_response = genai_pipeline(insights_prompt, max_length=500, num_return_sequences=1)
        business_insights = insight_response[0]['generated_text']
    except Exception as e:
        business_insights = f"Insight generation failed: {e}"

    return {
        "nearby_stores": nearby_stores,
        "predicted_delivery_times": predictions,
        "suggested_new_store_locations": new_store_coords,
        "business_insights": business_insights
    }

