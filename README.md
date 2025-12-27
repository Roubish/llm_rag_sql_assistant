# llm_rag_sql_assistant

**Project:** Industrial Safety & Surveillance → SQL Intelligent Assistant (Gemini) through RAG Application  
**Technologies:** Python, MySQL, PyMySQL, LangChain, Gemini-2.5, dotenv, Rich, Pandas, PIL  
**Duration:** Dec 2025 – Present  

---

## Overview

`llm_rag_sql_assistant` is a **conversational AI assistant** that allows users to query MySQL databases using **natural language**. Leveraging **Gemini LLM** and **RAG (Retrieval-Augmented Generation)** principles, it safely converts natural language questions into optimized SQL queries and provides rich output in the terminal.

The assistant supports multi-turn conversations, intelligent SQL generation, automatic error correction, and advanced query capabilities, making it ideal for **industrial safety and surveillance applications** or any scenario requiring fast insights from databases.

---

## Features

- Convert natural language queries into **safe SQL** for MySQL.
- **SQL safety checks** to prevent destructive commands (`DROP`, `TRUNCATE`, `DELETE`, `UPDATE`).
- **Auto-LIMIT** for large query results.
- Support for advanced queries:
  - Retrieve last N records or latest data
  - COUNT queries and aggregation
  - Date range filtering from natural language input
- **LLM-driven error correction** for SQL queries on database errors.
- **Rich CLI interface**:
  - Table visualization using Rich
  - CSV export for query results
  - Image handling from database blobs, file paths, or URLs
- **Multi-turn conversational support** with session memory and follow-up queries.
- **Prompt engineering** for structured and syntactically correct SQL output.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/llm_rag_sql_assistant.git
cd llm_rag_sql_assistant
