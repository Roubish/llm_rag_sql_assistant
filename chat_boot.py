"""
Natural Language → SQL → MySQL results

Usage:
  - Put your Google API key in the environment variable: GOOGLE_API_KEY
  - Put your MySQL credentials in env vars or set them in the script (not recommended)
  - Install dependencies: pip install pymysql python-dotenv tabulate pandas pillow langchain_google_genai langchain_core

This script:
  - Loads DB schema (table/columns) from MySQL
  - Uses Gemini (via langchain_google_genai.ChatGoogleGenerativeAI) to generate SQL
  - Validates SQL for safety
  - Executes SQL and formats results for the user
  - Supports simple preprocessing of user questions (e.g., "last 5", "how many")

NOTE: Do NOT hardcode real API keys in files committed to public repos. Use environment variables or a secure vault.
"""

import os
import re
import sys
import time
from typing import TypedDict, Optional
from typing_extensions import Annotated

import pymysql
import pandas as pd
from tabulate import tabulate
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------- Configuration --------------------------
# Read secrets from environment variables
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Load environment variables from .env file
load_dotenv()

# Now GOOGLE_API_KEY should be available
api_key = os.getenv("GOOGLE_API_KEY")


os.environ["GOOGLE_API_KEY"] = api_key
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "ghost")
DB_NAME = os.environ.get("DB_NAME", "mydb")

# if not GOOGLE_API_KEY:
#     raise RuntimeError("Set GOOGLE_API_KEY in environment before running")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in environment!")

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# -------------------------- Helpers: DB --------------------------------

def get_connection():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)


def fetch_table_schema():
    """Return a dict: {table_name: [col1, col2, ...]}"""
    schema = {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            tables = [t[0] for t in cur.fetchall()]
            for table in tables:
                cur.execute(f"DESCRIBE `{table}`;")
                cols = [row[0] for row in cur.fetchall()]
                schema[table] = cols
    finally:
        conn.close()
    return schema


def detect_alert_timestamp_column(schema: dict, alerts_table: str = "alerts") -> Optional[str]:
    """Try to find a reasonable timestamp column name in alerts table."""
    candidates = ["alert_time", "created_at", "timestamp", "time", "created", "date", "alert_timestamp"]
    cols = schema.get(alerts_table, [])
    for c in candidates:
        if c in cols:
            return c
    # fallback: try columns that contain 'time' or 'date'
    for c in cols:
        if "time" in c or "date" in c:
            return c
    return None


def run_sql_query(query: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            headers = [desc[0] for desc in cur.description] if cur.description else []
        return headers, results
    finally:
        conn.close()

# -------------------------- LLM prompt & generation --------------------------

query_prompt_template = PromptTemplate(
    input_variables=["dialect", "table_info", "input"],
    template=(
        "You are a SQL expert.\n"
        "Dialect: {dialect}\n"
        "Tables and columns: {table_info}\n\n"
        "Rules:\n"
        "1) Use the exact table and column names provided in the Tables and columns section.\n"
        "2) If the user asks for the latest or last N records, ORDER BY the alerts timestamp column (if present) DESC.\n"
        "3) If the user asks for counts (how many), return a COUNT(*) query.\n"
        "4) Do NOT emit destructive statements (DROP, DELETE, UPDATE without WHERE, ALTER, TRUNCATE).\n"
        "5) Return only a syntactically valid SQL query and nothing else. Do not add explanation text.\n\n"
        "Question: {input}"
    ),
)


# Define structured output type for the LLM
class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]


def generate_sql_from_question(schema: dict, question: str, max_retries: int = 2) -> str:
    """Ask Gemini to generate SQL. Returns SQL string or raises RuntimeError."""
    # Build a compact table info string
    table_info_parts = []
    for t, cols in schema.items():
        table_info_parts.append(f"{t}({', '.join(cols)})")
    table_info = "; ".join(table_info_parts)

    prompt = query_prompt_template.invoke({
        "dialect": "MySQL",
        "table_info": table_info,
        "input": question,
    })

    structured_llm = llm.with_structured_output(QueryOutput)

    last_err = None
    for attempt in range(max_retries + 1):
        result = structured_llm.invoke(prompt)
        sql = result.get("query")
        if not sql:
            raise RuntimeError("LLM did not return a SQL query")

        # Basic safety validation
        if is_sql_safe(sql):
            return sql
        else:
            last_err = "Generated SQL failed safety checks"
            # Provide feedback to LLM and retry
            prompt += "\n\nThe previous SQL was unsafe. Please regenerate a safe SQL query following the rules."
    raise RuntimeError(last_err or "Failed to generate safe SQL")

# -------------------------- SQL safety checks --------------------------

FORBIDDEN_PATTERNS = [r"\bDROP\b", r"\bTRUNCATE\b", r"\bALTER\b", r"\bUPDATE\b", r"\bDELETE\b"]


def is_sql_safe(sql: str) -> bool:
    s = sql.upper()
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, s):
            return False
    # disallow multiple statements separated by ';' (basic)
    if sql.strip().count(";") > 1:
        return False
    return True

# -------------------------- Post-processing & formatting --------------------------

def format_results(headers, rows):
    if not rows:
        return "No results returned."
    df = pd.DataFrame(rows, columns=headers)
    # Use tabulate for pretty printing
    return tabulate(df, headers="keys", tablefmt="psql", showindex=False)

# -------------------------- Image display helper --------------------------

def try_show_alert_image(headers, rows, image_column_candidates=("image", "image_url", "snapshot", "file_path", "file", "photo")):
    # Find an image column if present
    if not rows or not headers:
        return False
    lc = [h.lower() for h in headers]
    for candidate in image_column_candidates:
        if candidate in lc:
            idx = lc.index(candidate)
            val = rows[0][idx]
            if val is None:
                print("Image column is NULL for this alert.")
                return True
            # If it's bytes (BLOB), display
            if isinstance(val, (bytes, bytearray)):
                try:
                    img = Image.open(BytesIO(val))
                    img.show()
                    return True
                except Exception as e:
                    print("Failed to open image blob:", e)
                    return True
            # If it's a filesystem path or URL
            if isinstance(val, str):
                # local file
                if os.path.exists(val):
                    try:
                        img = Image.open(val)
                        img.show()
                        return True
                    except Exception as e:
                        print("Failed to open image file:", e)
                        return True
                # If it's a URL, we can't fetch here (no internet guaranteed). Print it.
                print("Image located at:", val)
                return True
    return False

# -------------------------- Interactive / Main flow --------------------------

def main_loop():
    print("Loading DB schema...")
    schema = fetch_table_schema()
    print("Found tables:", ", ".join(schema.keys()))

    # Try detect timestamp column for alerts table
    ts_col = detect_alert_timestamp_column(schema, alerts_table="alerts")
    if ts_col:
        print(f"Detected alert timestamp column: {ts_col}")
    else:
        print("No obvious timestamp column detected in 'alerts' table. If you ask for 'latest' results, specify the timestamp column name.")

    print("Ready. Ask questions about alerts (type 'exit' to quit). Examples:\n - Show last 10 Fire_extinguisher_is_not_available alerts\n - How many alerts of type Dock in last 7 days?\n - List distinct alert_type values")

    while True:
        try:
            user_q = input("\nYour question: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        if not user_q:
            continue
        if user_q.strip().lower() in ("exit", "quit"):
            break

        # Preprocess: if user asked for 'last N' and timestamp detected, hint the model
        preprocess_extra = ""
        m = re.search(r"last\s+(\d+)", user_q, re.IGNORECASE)
        if m and ts_col:
            preprocess_extra = f" Order by {ts_col} desc and limit {m.group(1)}."

        # If user asks generically for 'latest' but no ts column found, ask user to clarify
        if re.search(r"latest|last\s+\d+|most recent", user_q, re.IGNORECASE) and not ts_col:
            print("I couldn't detect a timestamp column to sort by. Please say which column to use (e.g., alert_time or created_at).")
            continue

        final_question = user_q + preprocess_extra

        try:
            sql = generate_sql_from_question(schema, final_question)
            print("Generated SQL:", sql)
        except Exception as e:
            print("Failed to generate SQL:", e)
            continue

        # Run SQL and show results
        try:
            headers, rows = run_sql_query(sql)
        except Exception as e:
            print("Error executing SQL:", e)
            # Optionally feed the DB error back to the LLM for correction (not implemented here)
            continue

        print(format_results(headers, rows))

        # Try show image if present
        shown = try_show_alert_image(headers, rows)
        if shown:
            print("Tried to display the alert image (or printed its location).")


if __name__ == '__main__':
    main_loop()
