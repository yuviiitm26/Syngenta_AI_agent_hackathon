🔍 Hybrid AI Agent with Claude 3.5 – SQL + Document Reasoning
This project implements an intelligent Hybrid AI Agent capable of answering natural language queries by reasoning over both:

📊 Structured data from a SQL database, and

📄 Unstructured policy documents using retrieval-augmented generation (RAG).

It uses Claude 3.5 Sonnet for language understanding and Chain-of-Thought (CoT) prompting to route and resolve queries through the appropriate pipeline.

🚀 Key Features
🔎 Query Type Classification
Automatically classifies user queries into SQL, Document, or Hybrid categories using CoT-based reasoning.

🧠 Hybrid Reasoning
Combines document understanding (via FAISS vector store) and database results to answer complex queries grounded in policy definitions.

📁 Document QA with RAG
Uses FAISS to retrieve top-k relevant documents and prompts Claude to answer based on context.

🗄️ SQL Generation + Execution
Converts natural language to executable SQL queries with schema grounding, executes on PostgreSQL, and returns result.

🔁 SQL Retry Mechanism
Detects hallucinated SQL errors (e.g., non-existent columns), patches and retries automatically.

📊 Real-Time Performance Metrics
Logs query response time, estimated Claude token usage, and cost for every interaction.

🛠️ Tech Stack
LLM: Claude 3.5 Sonnet via custom API proxy

Vector Store: FAISS

Database: PostgreSQL

Backend: Python

frontend: Streamlit UI for live querying

Monitoring: Console logs (timing, cost, tokens)

📈 Metrics Tracked
⏱️ Response Time

🧾 Token Usage (Input + Output)

💰 Cost Estimation

🔍 Document Retrieval Latency

🛢️ SQL Execution Time

🔄 Hybrid Integration Time

🧪 Example Use Cases
"What are the return policies for damaged items?" → Document QA

"Show orders with delivery delay over 7 days" → SQL Query

"List employees classified under 'non-compliant' by the latest policy" → Hybrid Reasoning

