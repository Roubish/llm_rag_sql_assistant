#!/usr/bin/env python3
"""
Upgraded Natural Language -> SQL -> MySQL assistant (Gemini)
Features:
 - .env support
 - smarter NL parsing: last N, how many, date ranges
 - safety checks, auto-LIMIT
 - retry with DB error feedback to LLM
 - CSV export
 - pretty terminal output (rich)
 - image handling (BLOB/local path/URL)
 - session memory for multi-turn dialog
"""

import os
import re
import csv
import time
import tempfile
import traceback
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, List

import pymysql
import pandas as pd
import dateparser
from dotenv import load_dotenv
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated
from typing import TypedDict

# ------------------------- configuration/load .env -------------------------
load_dotenv()  # loads .env into environment

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ghost")
DB_NAME = os.getenv("DB_NAME", "mydb")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment (.env?)")

# set for any libs that read it
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------------- utilities & globals -------------------------
console = Console()
SCHEMA_CACHE: Optional[Dict[str, List[str]]] = None
LAST_QUERY_CONTEXT: Dict[str, Any] = {}  # store last SQL, last result for session

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ------------------------- DB helpers -------------------------
def get_connection():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)

def fetch_table_schema() -> Dict[str, List[str]]:
    global SCHEMA_CACHE
    if SCHEMA_CACHE:
        return SCHEMA_CACHE
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
    SCHEMA_CACHE = schema
    return schema

def run_sql_query(query: str, fetchmany: Optional[int] = None) -> Tuple[List[str], List[Tuple]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            headers = [desc[0] for desc in cur.description] if cur.description else []
        return headers, rows
    finally:
        conn.close()

# ------------------------- SQL safety & normalization -------------------------
FORBIDDEN_PATTERNS = [r"\bDROP\b", r"\bTRUNCATE\b", r"\bALTER\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bCREATE\s+USER\b"]

def is_sql_safe(sql: str) -> bool:
    s = sql.upper()
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, s):
            return False
    # disallow multiple statements (rudimentary)
    if sql.strip().count(";") > 1:
        return False
    return True

def ensure_select_limit(sql: str, default_limit: int = 200) -> str:
    # add LIMIT if SELECT and no LIMIT present
    if sql.strip().lower().startswith("select") and "limit" not in sql.lower():
        return sql.rstrip().rstrip(";") + f" LIMIT {default_limit};"
    return sql

# ------------------------- LLM prompt & structured output -------------------------
query_prompt_template = PromptTemplate(
    input_variables=["dialect", "table_info", "input", "notes"],
    template=(
        "You are a SQL expert.\n"
        "Dialect: {dialect}\n"
        "Tables and columns: {table_info}\n\n"
        "Notes/Rules (follow precisely):\n"
        " - Use only the table and column names provided above.\n"
        " - If the user asks for 'last N' or 'latest', order by a logical timestamp column (e.g., alert_time) DESC.\n"
        " - If user asks 'how many' / 'count', return a COUNT(*) query.\n"
        " - Do NOT emit destructive statements (DROP/TRUNCATE/ALTER/DELETE/UPDATE/CREATE USER).\n"
        " - Return only a single syntactically valid SQL query and nothing else (no explanation).\n"
        " - If you cannot confidently map a column or table, ask for clarification instead of inventing names.\n\n"
        "Extra notes: {notes}\n\n"
        "Question: {input}"
    )
)

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def build_table_info_string(schema: Dict[str, List[str]]) -> str:
    parts = []
    for t, cols in schema.items():
        parts.append(f"{t}({', '.join(cols)})")
    return "; ".join(parts)

def ask_llm_for_sql(schema: Dict[str, List[str]], question: str, notes: str = "") -> str:
    table_info = build_table_info_string(schema)
    prompt = query_prompt_template.invoke({
        "dialect": "MySQL",
        "table_info": table_info,
        "input": question,
        "notes": notes
    })
    structured = llm.with_structured_output(QueryOutput)
    result = structured.invoke(prompt)
    sql = result.get("query")
    return sql

# ------------------------- NLP helpers -------------------------
def parse_last_n(question: str) -> Optional[int]:
    m = re.search(r"last\s+(\d+)", question, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"most recent\s+(\d+)", question, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return None

def wants_count(question: str) -> bool:
    return bool(re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", question, re.IGNORECASE))

# below not required
def parse_date_range(question: str) -> Optional[str]:
    # Try to detect simple phrases and convert to SQL-friendly conditions
    # Examples: "last week", "last 7 days", "yesterday", "between 2024-01-01 and 2024-02-01"
    q = question.lower()
    if "last week" in q:
        return "alert_time >= NOW() - INTERVAL 7 DAY"
    m = re.search(r"last\s+(\d+)\s+days", q)
    if m:
        days = int(m.group(1))
        return f"alert_time >= NOW() - INTERVAL {days} DAY"
    if "yesterday" in q:
        return "DATE(alert_time) = CURDATE() - INTERVAL 1 DAY"
    # between ... and ...
    m2 = re.search(r"between\s+([0-9\-\/]+)\s+and\s+([0-9\-\/]+)", q)
    if m2:
        dt1 = dateparser.parse(m2.group(1))
        dt2 = dateparser.parse(m2.group(2))
        if dt1 and dt2:
            return f"alert_time BETWEEN '{dt1.strftime('%Y-%m-%d')}' AND '{dt2.strftime('%Y-%m-%d')}'"
    return None

# ------------------------- postprocess & display -------------------------
def display_table(headers: List[str], rows: List[Tuple], max_rows: int = 100):
    if not rows:
        console.print("[yellow]No results returned.[/yellow]")
        return
    table = Table(show_lines=False)
    for h in headers:
        table.add_column(str(h))
    display_rows = rows if len(rows) <= max_rows else rows[:max_rows]
    for r in display_rows:
        table.add_row(*[str(c) if c is not None else "" for c in r])
    console.print(table)
    if len(rows) > max_rows:
        console.print(f"[dim]Showing {max_rows}/{len(rows)} rows. Use export to save full results.[/dim]")

def export_to_csv(headers: List[str], rows: List[Tuple], filename: Optional[str] = None) -> str:
    if not filename:
        filename = f"query_results_{int(time.time())}.csv"
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(filename, index=False)
    return filename

# not required only image link will be useful
def try_show_image(headers: List[str], rows: List[Tuple]) -> None:
    if not rows or not headers:
        return
    lc = [h.lower() for h in headers]
    candidates = ["image", "image_url", "snapshot", "file_path", "file", "photo", "img"]
    for c in candidates:
        if c in lc:
            idx = lc.index(c)
            val = rows[0][idx]
            if val is None:
                console.print("[yellow]Image column present but value is NULL for latest row.[/yellow]")
                return
            # bytes/blob
            if isinstance(val, (bytes, bytearray)):
                try:
                    img = Image.open(BytesIO(val))
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(tmp.name)
                    console.print(f"[green]Saved blob image to {tmp.name}[/green]")
                    return
                except Exception as e:
                    console.print("[red]Failed to render image blob:[/red]", e)
                    return
            # string path or URL
            if isinstance(val, str):
                if os.path.exists(val):
                    console.print(f"[green]Opening local image: {val}[/green]")
                    try:
                        Image.open(val).show()
                    except Exception:
                        console.print("[red]Failed to open image with PIL[/red]")
                else:
                    console.print(f"[blue]Image reference (path or URL): {val}[/blue]")
                return

# ------------------------- main conversational loop -------------------------
def main_loop():
    console.print("[bold green]NL → SQL assistant (upgraded)[/bold green]")
    schema = fetch_table_schema()
    console.print(f"Found tables: [cyan]{', '.join(schema.keys())}[/cyan]")

    ts_col = None
    if "alerts" in schema:
        ts_col = detect_timestamp_col(schema["alerts"])
        if ts_col:
            console.print(f"[green]Detected timestamp column in alerts:[/green] {ts_col}")
        else:
            console.print("[yellow]No obvious timestamp column detected in 'alerts'. When asking for latest records, include a time column or specify which to use.[/yellow]")

    console.print("Ready. Examples:\n - Show last 10 Fire_extinguisher_is_not_available alerts\n - How many Dock alerts in last 7 days?\n - List distinct alert_type values\nType 'exit' to quit.")

    while True:
        try:
            question = Prompt.ask("\nYour question")
        except KeyboardInterrupt:
            console.print("\n[bold]Exiting[/bold]")
            break
        q = (question or "").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # Preprocess
        notes = ""
        last_n = parse_last_n(q)
        if last_n and ts_col:
            notes += f"User asked for last {last_n} rows; prefer ordering by {ts_col} DESC and LIMIT {last_n}. "
        date_cond = parse_date_range(q)
        if date_cond:
            notes += f"Timestamp filter: {date_cond}. "

        wants_cnt = wants_count(q)
        if wants_cnt:
            notes += "User expects a COUNT(*) answer. "

        # Build question passed to LLM: give explicit hint re: alerts timestamp column if exists
        if "alerts" in schema and ts_col:
            notes += f"The alerts table timestamp column is {ts_col}. "

        # Ask LLM for SQL
        try:
            sql = ask_llm_for_sql(schema, q, notes=notes)
            if not sql:
                console.print("[red]LLM returned empty SQL.[/red]")
                continue
        except Exception as e:
            console.print("[red]Error calling LLM:[/red]", str(e))
            continue

        # Basic safety
        if not is_sql_safe(sql):
            console.print("[red]Generated SQL failed safety checks and will not be executed.[/red]")
            console.print(sql)
            continue

        # If SELECT and no LIMIT and user didn't explicitly ask for ALL, add a default limit
        sql_to_run = ensure_select_limit(sql, default_limit=500)

        # Execute with retry: if DB error, feed back error+SQL to LLM once to fix
        attempt = 0
        max_attempts = 2
        while attempt <= max_attempts:
            attempt += 1
            try:
                headers, rows = run_sql_query(sql_to_run)
                LAST_QUERY_CONTEXT['last_sql'] = sql_to_run
                LAST_QUERY_CONTEXT['last_headers'] = headers
                LAST_QUERY_CONTEXT['last_rows'] = rows
                console.print("[green]Query executed successfully.[/green]")
                break
            except Exception as db_e:
                console.print("[red]DB error:[/red]", str(db_e))
                # If attempts remain, ask LLM to correct the SQL using the DB error
                if attempt <= max_attempts:
                    err_msg = str(db_e)
                    fix_notes = f"The SQL produced an error when executed: {err_msg}. Please correct the SQL to run on MySQL using the provided schema and rules. Return only the corrected SQL."
                    try:
                        sql = ask_llm_for_sql(schema, q, notes=notes + " " + fix_notes)
                        if not sql:
                            console.print("[red]LLM did not return a correction.[/red]")
                            break
                        if not is_sql_safe(sql):
                            console.print("[red]Corrected SQL failed safety check.[/red]")
                            break
                        sql_to_run = ensure_select_limit(sql, default_limit=500)
                        console.print("[yellow]Retrying with corrected SQL:[/yellow]", sql_to_run)
                        continue
                    except Exception as e2:
                        console.print("[red]Failed to get correction from LLM:[/red]", str(e2))
                        break
                else:
                    console.print("[red]Max retries reached. Aborting.[/red]")
                    break

        # If we have results, display
        if LAST_QUERY_CONTEXT.get('last_rows') is None:
            console.print("[red]No results to show.[/red]")
            continue

        headers = LAST_QUERY_CONTEXT['last_headers']
        rows = LAST_QUERY_CONTEXT['last_rows']

        # If user wanted count and the SQL was COUNT, we can nicely print it
        if wants_cnt and len(rows) == 1 and len(rows[0]) >= 1:
            console.print(f"[bold]Count:[/bold] {rows[0][0]}")
        else:
            display_table(headers, rows, max_rows=200)

        # Try to show image if any
        try_show_image(headers, rows)

        # Offer export options
        if Confirm.ask("Export results to CSV?"):
            fname = export_to_csv(headers, rows)
            console.print(f"[green]Exported to {fname}[/green]")

        # Allow follow up question memory
        if Confirm.ask("Do you want to ask a follow-up about these results (use 'filter by ...' or 'only show ...')?"):
            follow = Prompt.ask("Follow-up question")
            # A simple follow-up processing: combine last SQL as subquery and apply filter if present
            # For safety keep it simple: if follow contains 'only' or 'filter', let LLM generate new SQL referencing the alerts table
            combined_q = f"Base question: {q}. Follow-up: {follow}. Use the same schema: {build_table_info_string(schema)}"
            # ask LLM to create a new SQL
            try:
                new_sql = ask_llm_for_sql(schema, combined_q, notes="Follow-up; ensure safe SQL.")
                new_sql = ensure_select_limit(new_sql)
                console.print("[yellow]Generated follow-up SQL:[/yellow]", new_sql)
                h2, r2 = run_sql_query(new_sql)
                display_table(h2, r2)
            except Exception as e:
                console.print("[red]Failed to produce or run follow-up SQL:[/red]", str(e))

# ------------------------- helpers not yet defined (detect_timestamp) -------------------------
def detect_timestamp_col(alerts_cols: List[str]) -> Optional[str]:
    # choose sensible timestamp-like column from list
    candidates = ["alert_time", "created_at", "timestamp", "time", "created", "date", "alert_timestamp", "ts"]
    for c in candidates:
        if c in alerts_cols:
            return c
    for c in alerts_cols:
        if "time" in c or "date" in c or "ts" in c:
            return c
    return None

# ------------------------- run main -------------------------
if __name__ == "__main__":
    try:
        main_loop()
    except Exception:
        console.print("[red]Fatal error running assistant:[/red]")
        traceback.print_exc()
