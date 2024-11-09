import os
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
from tabulate import tabulate  
import streamlit as st


load_dotenv()


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def connect_to_db():
    return psycopg2.connect(
        host="localhost",
        database="sales_data_db",
        user="postgres",
        password=os.getenv("DB_PASSWORD")
    )


def question_to_sql(question):
    prompt = f"Convert the following business question to SQL for a table named 'sales_data' with columns: Region, Country, Item_Type, Sales_Channel, Order_Priority, Order_Date, Order_ID, Ship_Date, Units_Sold, Unit_Price, Unit_Cost, Total_Revenue, Total_Cost, Total_Profit. Return only the SQL code without explanations.\n\nQuestion: {question}"
    
    response = client.chat.completions.create(
        model="gpt-4",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )
    
    sql_query = response.choices[0].message.content.strip()
    

    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[-1].split("```")[0].strip()
    
    return sql_query


def execute_query(sql_query):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except Exception as e:
        return str(e)


def main():
    st.title("SQL Query Bot for Sales Data")
    st.write("Ask a question about the sales data, and the bot will generate and execute an SQL query for you.")


    question = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if question:
            sql_query = question_to_sql(question)
            st.write("Generated SQL query:")
            st.code(sql_query)

            # Execute the query and display results
            results = execute_query(sql_query)
            if isinstance(results, list) and results:
                st.write("Results:")
                st.table(results)
            else:
                st.write("No results or an error occurred.")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
