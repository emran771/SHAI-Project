import streamlit as st
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading


model = joblib.load('E:\\Projects\\sales\\model\\linear_model.pkl')
preprocessor = joblib.load('E:\\Projects\\sales\\model\\preprocessor.pkl')


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SalesPredictionInput(BaseModel):
    units_sold: int
    unit_cost: float
    item_type: str
    sales_channel: str
    order_priority: str


@app.post("/predict")
async def predict_sales(input_data: SalesPredictionInput):

    sample_input = {
        "Order ID": 12345, 
        "Units Sold": input_data.units_sold,
        "Unit Cost": input_data.unit_cost,
        "Total Revenue": input_data.units_sold * input_data.unit_cost,  
        "Total Cost": 0.8 * (input_data.units_sold * input_data.unit_cost),
        "Total Profit": (input_data.units_sold * input_data.unit_cost) - (0.8 * input_data.units_sold * input_data.unit_cost),
        "Order Processing Time": 15,  
        "Order Year": 2022, "Order Month": 5, "Order Day": 12,  
        "Ship Year": 2022, "Ship Month": 5, "Ship Day": 16,  
        "Region": "Europe",  
        "Country": "Germany",  
        "Item Type": input_data.item_type,
        "Sales Channel": input_data.sales_channel,
        "Order Priority": input_data.order_priority
    }

    
    sample_df = pd.DataFrame([sample_input])
    sample_processed = preprocessor.transform(sample_df)

    # Predict with the model
    prediction = model.predict(sample_processed)
    return {"predicted_sales_value": prediction[0]}


server_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), daemon=True)
server_thread.start()


st.title('Sales Prediction Application')

# User inputs for Streamlit UI
units_sold = st.slider('Units Sold', min_value=1, max_value=10000, value=1000, step=10)
unit_cost = st.slider('Unit Cost', min_value=1.0, max_value=500.0, value=50.0, step=0.5)
item_type = st.selectbox('Item Type', options=['Office Supplies', 'Meat', 'Cereal', 'Cosmetics'])
sales_channel = st.selectbox('Sales Channel', options=['Offline', 'Online'])
order_priority = st.selectbox('Order Priority', options=['C', 'H', 'L', 'M'])


sample_input = {
    "Order ID": 12345,  
    "Units Sold": units_sold,
    "Unit Cost": unit_cost,
    "Total Revenue": units_sold * unit_cost,  
    "Total Cost": 0.8 * (units_sold * unit_cost),  
    "Total Profit": (units_sold * unit_cost) - (0.8 * units_sold * unit_cost),
    "Order Processing Time": 15,  
    "Order Year": 2022, "Order Month": 5, "Order Day": 12,  
    "Ship Year": 2022, "Ship Month": 5, "Ship Day": 16,  
    "Region": "Europe",  
    "Country": "Germany",  
    "Item Type": item_type,
    "Sales Channel": sales_channel,
    "Order Priority": order_priority
}


sample_df = pd.DataFrame([sample_input])


sample_processed = preprocessor.transform(sample_df)

# Predict with the model
prediction = model.predict(sample_processed)

# Display the prediction
st.write('Predicted Sales Value:', prediction[0])
