import os
import re
import json
import time
from urllib import response
import numpy as np
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

# LangChain and FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import torch


# -------------------- Environment Setup -------------------- #
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

engine = create_engine(DATABASE_URL) if DATABASE_URL else None




# -------------------- Claude API -------------------- #
import time

def call_claude(prompt, model="claude-3.5-sonnet", temperature=0.7, max_tokens=1024, log_usage=True):
    url = "https://quchnti6xu7yzw7hfzt5yjqtvi0kafsq.lambda-url.eu-central-1.on.aws/"

    if not ANTHROPIC_API_KEY:
        return "Error: Missing ANTHROPIC_API_KEY."

    payload = {
        "api_key": ANTHROPIC_API_KEY,
        "prompt": prompt,
        "model_id": model,
        "model_params": {
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }

    try:
        start = time.time()
        res = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        duration = time.time() - start

        res.raise_for_status()
        result = res.json()
        response_text = result.get("response", {}).get("content", [{}])[0].get("text", "No text found")

        if log_usage:
            input_tokens = len(prompt.split())
            output_tokens = len(response_text.split())
            approx_cost = input_tokens * 15e-6 + output_tokens * 75e-6

            print("\ Claude Performance Metrics")
            print(f"  Time taken: {round(duration, 3)}s")
            print(f" Input tokens: {input_tokens}")
            print(f" Output tokens: {output_tokens}")
            print(f" Estimated Cost: ${round(approx_cost, 6)}\n")

        return response_text

    except Exception as e:
        return f"Claude error: {e}"

# Static system prompt for LLM role
# -------------------- Vector Store Setup -------------------- #
# ---- Global Dictionary for Column Data Types ---- #
column_data_types = {}

# ---- Get Schema Columns ---- #
def get_table_columns(table="orders"):
    global column_data_types
    if not engine:
        return []

    try:
        columns = inspect(engine).get_columns(table)
        column_data_types = {col["name"]: str(col["type"]) for col in columns}
        return list(column_data_types.keys())
    except Exception as e:
        print(f"Error fetching columns: {e}")
        return []

# ---- Get Sample Rows ---- #
def get_sample_data(table="orders", limit=3):
    if not engine:
        return None
    try:
        df = pd.read_sql(text(f"SELECT * FROM {table} LIMIT {limit};"), engine)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error fetching sample data: {e}")
        return None

# ---- Check Parentheses ---- #
def has_balanced_brackets(query: str) -> bool:
    stack = []
    for char in query:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return False
            stack.pop()
    return not stack

# ---- Fix Parentheses ---- #
def fix_unbalanced_parentheses(query: str) -> str:
    sql = query.strip()
    max_iter = 5

    i = 0
    while sql.count('(') < sql.count(')') and i < max_iter:
        sql = re.sub(r'\)\s*$', '', sql)
        i += 1

    i = 0
    while sql.count('(') > sql.count(')') and i < max_iter:
        sql = re.sub(r'^\s*\(', '', sql)
        i += 1

    sql = re.sub(r'\(\s*\)', '', sql)
    return sql


# Build SQL prompt with type hint for LLM
def enhance_sql_query(sql_query):
    if not has_balanced_brackets(sql_query):
        sql_query = fix_unbalanced_parentheses(sql_query)

    if not column_data_types:
        return sql_query

    # Fix TO_TIMESTAMP usage based on actual column data types
    sql_query = fix_to_timestamp_usage(sql_query, column_data_types)

    # Your existing logic for casting numeric/text columns
    for col, dtype in column_data_types.items():
        col_escaped = re.escape(col)

        if ('sales' in col or 'profit' in col or 'amount' in col) and 'text' in dtype:
            pattern = rf'\b{col_escaped}\b'
            sql_query = re.sub(pattern, f"{col}::numeric", sql_query)

    return sql_query

# ---- Build Prompt for LLM ---- #
def build_sql_prompt(question, columns, sample_data=None):
    key_entities = """
Order & Sales Info:
Order Id, Order Date, Order Status, Order Profit Per Order, Sales, Order Item Total, Order Item Quantity, Order Item Product Price, Order Item Discount, Order Item Discount Rate, Benefit per order, Sales per customer

Shipping Info:
Days for shipping (real), Days for shipment (scheduled), Delivery Status, Late_delivery_risk, Shipping date, Shipping Mode

 Customer Info:
Customer Id, Customer Fname, Customer Lname, Customer Email, Customer City, Customer Country, Customer Segment, Customer State, Customer Zipcode

 Product Info:
Product Card Id, Product Category Id, Product Name, Product Price, Product Status, Category Name, Product Description (empty), Department Name

 Geographic Info:
Latitude, Longitude, Order Region, Order State, Order City, Market

Payment & Type Info:
Type (e.g., DEBIT, CASH, etc.)
"""

    prompt = f"""
You are a senior data analyst and PostgreSQL expert. Given the table `orders`, write a valid, syntactically correct SQL query that answers the question below.

Use the following key entities and columns to guide your SQL generation:

{key_entities}

Guidelines:
- Assume some columns like dates or amounts might be stored as text and cast them explicitly (e.g., `::date`, `::numeric`).
- Use Common Table Expressions (WITH) for intermediate calculations.
- Use window functions properly without nesting aggregates inside aggregates.
- Use aliases, clean formatting, and only return the SQL query wrapped in triple backticks with 'sql' tag.
- Ensure all parentheses are balanced — no extra or missing brackets in CTEs, functions, etc.
- If a column stores dates as text in the format "MM/DD/YYYY HH24:MI", cast it as:
TO_TIMESTAMP(column_name, 'MM/DD/YYYY HH24:MI')
Schema Columns: {', '.join(columns)}
{f"Sample Rows: {sample_data}" if sample_data else ""}

User Question:
{question}

Important: **Return ONLY the SQL query, NO explanation, NO comments, NO extra text.**
SQL query only:


"""
    return prompt.strip()

# ---- Extract SQL from LLM Response ---- #
def extract_sql(text: str) -> str:
    cleaned = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"(SELECT[\s\S]+|WITH[\s\S]+)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned

def fix_to_timestamp_usage(sql_query: str, column_data_types: dict) -> str:
    """
    Fix SQL query to apply TO_TIMESTAMP only on string columns representing dates.
    Remove TO_TIMESTAMP from columns already of timestamp or date type.

    Args:
        sql_query: The raw SQL query string.
        column_data_types: Dict mapping column names to their data types.

    Returns:
        Updated SQL query with correct TO_TIMESTAMP usage.
    """
    # First, find all TO_TIMESTAMP calls
    # Pattern example: TO_TIMESTAMP(column_name, 'format')
    pattern = re.compile(r"TO_TIMESTAMP\(\s*([a-zA-Z0-9_]+)\s*,\s*'[^']+'\s*\)", re.IGNORECASE)

    def replace_func(match):
        col = match.group(1)
        dtype = column_data_types.get(col.lower(), '').lower()
        # If column is timestamp or date, remove TO_TIMESTAMP, keep column name as is
        if 'timestamp' in dtype or 'date' in dtype:
            return col
        # Otherwise, keep as is (string columns)
        return match.group(0)

    # Apply replacement to SQL query
    fixed_sql = pattern.sub(replace_func, sql_query)

    return fixed_sql

def fix_sql_timestamp_casts_with_llm(sql_query: str, column_data_types: dict) -> str | None:
    prompt = f"""
You are a PostgreSQL expert. 

Given the following SQL query and table schema with column data types, please:

1. Only apply TO_TIMESTAMP(column, format) on columns where the column data type is a string/text type containing date/time data.
2. For columns that are already of type timestamp or date, use them as-is without TO_TIMESTAMP().
3. Fix any casting errors or incorrect function usage that could cause PostgreSQL errors such as:
   - function to_timestamp(timestamp without time zone, unknown) does not exist
   - date/time field value out of range
4. Output a corrected and fully working SQL query that runs without errors.

Table columns and their data types: 
{column_data_types}

Original SQL query:
{sql_query}
"""
    response = call_claude(prompt)
    if not response:
        print(" No response from LLM")
        return None

    fixed_sql = extract_sql(response)
    return fixed_sql.strip() if fixed_sql else None


# Run SQL safely
def run_sql(sql_query):
    if not engine:
        st.error("Database connection not configured.")
        return pd.DataFrame()
    try:
        return pd.read_sql(text(sql_query), engine)
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return pd.DataFrame()
def auto_cast_ilike_columns(sql_query):
    pattern = r"(\b[\w\.]+\b)\s+(ILIKE|LIKE)\s+(['\"%][^'\"]*['\"%])"
    def replacer(match):
        col, op, val = match.groups()
        # Avoid double casting
        if '::text' in col:
            return match.group(0)
        return f"{col}::text {op} {val}"
    return re.sub(pattern, replacer, sql_query, flags=re.IGNORECASE)
# Retry logic for running enhanced SQL
def run_sql_with_retry(sql_query):
    df = run_sql(sql_query)
    if df.empty:
        # Step 1: Try auto-casting ILIKE/LIKE columns
        cast_fixed_sql = auto_cast_ilike_columns(sql_query)
        if cast_fixed_sql != sql_query:
            print("Retrying with auto-casted ILIKE columns...")
            df = run_sql(cast_fixed_sql)
            if not df.empty:
                return df

        # Step 2: Try your existing enhancement
        enhanced = enhance_sql_query(sql_query)
        if enhanced != sql_query:
            print("Retrying with enhanced SQL...")
            df = run_sql(enhanced)
            if not df.empty:
                return df

        # Step 3: Try fixing timestamp casts with LLM
        print("Retrying with LLM-fixed SQL casts...")
        llm_fixed_sql = fix_sql_timestamp_casts_with_llm(sql_query, column_data_types)
        if llm_fixed_sql and llm_fixed_sql != sql_query:
            df = run_sql(llm_fixed_sql)
            if not df.empty:
                print("Success with LLM-fixed SQL!")
                return df

        print("All retries failed.")
    return df


# -------------------- Document QA Setup -------------------- #
def load_documents(pdf_dir="data/docs"):
    documents = []
    if not os.path.exists(pdf_dir):
        return []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

def store_index(chunks, index_path="faiss_index"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_path)

def load_index(index_path="faiss_index"):
    if not os.path.exists(index_path):
        return None
    try:
        vectorstore = FAISS.load_local(
            index_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def document_qa(question, faiss_index, k=5):
    if not faiss_index:
        return "FAISS index not loaded."
    docs = faiss_index.similarity_search(question, k=k)
    combined_context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""You are a helpful assistant. Based on the following context, answer the question clearly and accurately.

Context:
{combined_context}

Question: {question}
Answer:"""


    return call_claude(prompt)

##-----hybrid query-----##

def build_hybrid_prompt(question, relevant_docs, sql_result_df):
    return f"""
You are an intelligent assistant skilled in data analysis and document understanding. You have access to both structured data (SQL query results) and unstructured data (documents). Your goal is to reason like a human analyst by combining insights from both.

### STEP 1: SQL ANALYSIS
Start by analyzing the SQL result below to extract key patterns, numbers, trends, or facts relevant to the question.

SQL RESULT:
{sql_result_df.to_markdown(index=False)}

### STEP 2: DOCUMENT INSIGHTS
Use the document context below to complement or validate the information from the SQL result. Look for definitions, policies, explanations, or any text that helps interpret the data.

DOCUMENT CONTEXT:
{chr(10).join(relevant_docs)}

### STEP 3: CHAIN-OF-THOUGHT REASONING
Think step-by-step to answer the user’s question using both:
- Data insights from the SQL output
- Interpretive or background knowledge from the documents

Show your reasoning before giving the answer.

### USER QUESTION
{question}

IMPORTANT: Provide a clear, concise answer based on the combined reasoning above.
IMPORTANT: Do not return any SQL code, just the final answer.
IMPORTANT:Dont return your reasoning, just the final answer.
IPORTANT:DOn't return the Document context, just the final answer.
### FINAL ANSWER
Provide a clear and concise answer based on the combined reasoning above.
"""


def retrieve_relevant_docs(query, faiss_index, k=3):
    # faiss_index is a LangChain FAISS object
    docs = faiss_index.similarity_search(query, k=k)
    # docs is a list of Document objects, extract text content
    return [doc.page_content for doc in docs]

# ---- Main Function to Generate Optimized SQL ---- #
def generate_sql_query_from_question(question: str) -> str | None:
    try:
        print("Fetching schema and samples...")
        columns = get_table_columns()
        if not columns:
            print(" No columns found in 'orders' table.")
            return None

        sample_data = get_sample_data()

        print(" Building prompt...")
        prompt = build_sql_prompt(question, columns, sample_data)

        print(" Sending to Claude...")
        response = call_claude(prompt)

        if not response:
            print(" Claude returned no response.")
            return None

        print("Extracting SQL...")
        sql_query = extract_sql(response)
        if not sql_query:
            print(" No SQL extracted.")
            return None

        sql_query = sql_query.strip()
        if not (sql_query.lower().startswith("select") or sql_query.lower().startswith("with")):
            print("SQL doesn't start with SELECT/WITH.")
            return None

        print("🛠️ Enhancing SQL...")
        enhanced_sql = enhance_sql_query(sql_query)

        if not has_balanced_brackets(enhanced_sql):
            print("Brackets unbalanced. Fixing...")
            enhanced_sql = fix_unbalanced_parentheses(enhanced_sql)

        print(" SQL generation complete.")
        return enhanced_sql

    except Exception as e:
        print(f" Unexpected error: {e}")
        return None

def fix_duplicate_aliases(sql_query):
    # Detect duplicate aliases using regex pattern like "AS h" appearing more than once
    aliases = re.findall(r'\bAS\s+(\w+)', sql_query, re.IGNORECASE)
    seen = set()
    alias_map = {}
    for alias in aliases:
        if alias.lower() in seen:
            # Generate a unique alias
            new_alias = alias + "_1"
            alias_map[alias] = new_alias
        else:
            seen.add(alias.lower())

    # Replace old aliases with new ones
    for old, new in alias_map.items():
        sql_query = re.sub(rf'\b{old}\b', new, sql_query)

    return sql_query
def extract_policy_logic(question, docs):
    """
    Use Claude to extract rule-based logic from documents based on the user's question.
    This helps handle definitions, policies, or thresholds needed before generating SQL.
    """
    policy_prompt = f"""
You are a policy analyst helping translate user questions into logic that can be queried with data.

DOCUMENT CONTEXT:
{chr(10).join(docs)}

USER QUESTION:
{question}

TASK:
If the question involves any rules, classifications, thresholds, or definitions from the documents, extract that logic clearly.

Example Output:
- "Hazardous materials" are defined as items with code 'HAZMAT-001', 'HAZMAT-002'
- "No-movers" are items with zero sales in the past 180 days
- Ethical suppliers must have score >= 85 in supplier_audit_score

Only return criteria in plain English or pseudocode.
"""
    return call_claude(policy_prompt)

# -------------------- Config & Load FAISS -------------------- #
st.set_page_config(page_title="Claude AI Agent", layout="wide")
st.title("🧠 Unified Claude AI Agent")

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

if not os.path.exists("faiss_index"):
    with st.spinner("📂 Building FAISS index from PDFs..."):
        chunks = load_documents()
        if chunks:
            store_index(chunks)
        else:
            st.warning("No PDF documents found in the 'pdfs' folder.")
faiss_index = load_index()

# -------------------- Chain-of-Thought Classifier -------------------- #
def classify_query_with_claude(question):
    """
    Use Claude with CoT reasoning to classify query.
    It returns both the full reasoning and the final type (sql/document/hybrid).
    """
    cot_prompt = f"""
You are a helpful AI agent classifying user questions into one of the following three types:

1. SQL – when the question involves filtering, counting, or querying structured data from a database.
2. Document – when the question asks about definitions, classifications, policies, or textual info from documents.
3. Hybrid – when the question involves both: policy logic from documents AND structured filtering from databases.

Think step-by-step (Chain of Thought) to reason about the intent. Then, on the **last line only**, write:
Answer: SQL
or
Answer: Document
or
Answer: Hybrid

Example format:

Thought: This question asks about late deliveries. It might need policy info from a document, and filter orders. That’s a hybrid.
Answer: Hybrid

Now classify this question:

"{question}"
"""

    response = call_claude(cot_prompt)

    # Try to extract the last line (Answer: XXX)
    final_type = "unknown"
    for line in reversed(response.strip().splitlines()):
        if line.strip().lower().startswith("answer:"):
            final_type = line.split(":")[1].strip().lower()
            break

    return final_type, response.strip()
def build_dynamic_database_schema_with_samples(engine):
    tables = ["orders", "products", "customers"]
    schema_lines = ["You can only use the following tables and columns:\n"]

    for table in tables:
        try:
            # Get columns
            inspector = inspect(engine)
            columns = inspector.get_columns(table)
            column_names = [col["name"] for col in columns]
            schema_lines.append(f"- {table}({', '.join(column_names)})")

            # Get sample rows
            df = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 3;"), engine)
            if not df.empty:
                schema_lines.append(f"  Example rows from '{table}':")
                for idx, row in df.iterrows():
                    sample = ", ".join([f"{k}={v}" for k, v in row.items()])
                    schema_lines.append(f"    - {sample}")
            else:
                schema_lines.append(f"  No data found in '{table}'.")

        except Exception as e:
            schema_lines.append(f"- {table}( Error: {e})")

    # Final notes for prompt grounding
    schema_lines.append("\n Use only the tables and columns listed above.")
    schema_lines.append(" Do NOT use 'product_info' — it does NOT exist.")
    schema_lines.append(" Generate only valid PostgreSQL syntax. No explanations or natural language.")

    return "\n".join(schema_lines)


# -------------------- Chat-Based Unified Agent UI -------------------- #
st.title("🤖 Smart AI Agent")
st.markdown("Ask anything about your data or documents. Hybrid reasoning supported.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_query = st.chat_input("Ask a question (about database, document, or both)")

# Add user query to chat
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

# Show chat history so far
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Now respond if new user query is added
if user_query:
    with st.spinner("🧠 Thinking..."):
        DATABASE_SCHEMA = build_dynamic_database_schema_with_samples(engine)

        try:
            # Step 1: Classify query
            qtype, cot_reasoning = classify_query_with_claude(user_query)

            if qtype == "sql":
                sql_query = generate_sql_query_from_question(user_query)
                if sql_query:
                    df = run_sql_with_retry(sql_query)
                    result_text = df.to_markdown() if not df.empty else "No results returned."
                    st.session_state.chat_history.append({"role": "assistant", "content": result_text})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "❌ Failed to generate SQL query."})

            elif qtype == "document":
                doc_answer = document_qa(user_query, faiss_index)
                st.session_state.chat_history.append({"role": "assistant", "content": doc_answer})

            elif qtype == "hybrid":
                try:
                    docs = retrieve_relevant_docs(user_query, faiss_index, k=5)
                    criteria = extract_policy_logic(user_query, docs)

                    enhanced_q = f"""Question: {user_query}
Policy logic to follow: {criteria}
Available database schema:
{DATABASE_SCHEMA}
Generate a SQL query to answer the question. Use only available tables and valid columns.
Note: Use PostgreSQL syntax. When rounding to a specific number of decimal places, cast the expression to NUMERIC before applying ROUND, like this:
ROUND((expression)::numeric, decimal_places)
Avoid using ROUND() with double precision and two arguments without casting, as it causes errors."""

                    sql_query = generate_sql_query_from_question(enhanced_q)

                    if not sql_query:
                        st.session_state.chat_history.append({"role": "assistant", "content": "Failed to generate SQL query."})
                    else:
                        if "product_info" in sql_query:
                            sql_query = sql_query.replace("product_info", "products")

                        sql_query = fix_duplicate_aliases(sql_query)

                        sql_result_df = pd.read_sql(text(sql_query), engine)

                        hybrid_prompt = build_hybrid_prompt(user_query, docs, sql_result_df)
                        answer = call_claude(hybrid_prompt)

                        result_text = answer + "\n\n\n**SQL Table Result Preview**\n" + sql_result_df.head(5).to_markdown()
                        st.session_state.chat_history.append({"role": "assistant", "content": result_text})

                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f" Hybrid reasoning failed: {e}"})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f" Error: {e}"})

    # Show latest message from assistant
    with st.chat_message("assistant"):
        st.markdown(st.session_state.chat_history[-1]["content"])