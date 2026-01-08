import streamlit as st
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine
import pandas as pd

from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# ============================================================
# App UI
# ============================================================
st.set_page_config(page_title="Chat with SQL Database", page_icon="🦜")
st.title("🦜 Chat with SQL Database")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = [
    "Use SQLite Database (student.db)",
    "Connect to MySQL Database",
]

selected_opt = st.sidebar.radio("Choose Database", radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input("Groq API Key", type="password")


# ============================================================
# LLM Configuration
# ============================================================
llm = None
if api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
else:
    st.info("🔑 Enter Groq API Key to start chatting.")


# ============================================================
# Database Configuration
# ============================================================
@st.cache_resource(ttl=2 * 60 * 60)
def configure_db(
    db_uri,
    mysql_host=None,
    mysql_user=None,
    mysql_password=None,
    mysql_db=None,
):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(
            f"file:{dbfilepath}?mode=ro",
            uri=True,
        )
        engine = create_engine("sqlite:///", creator=creator)
        return SQLDatabase(engine), engine
    else:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )
        return SQLDatabase(engine), engine


if db_uri == MYSQL:
    db, engine = configure_db(
        db_uri,
        mysql_host,
        mysql_user,
        mysql_password,
        mysql_db,
    )
else:
    db, engine = configure_db(db_uri)


# ============================================================
# Build Database Schema for Prompt
# ============================================================
def get_db_schema_text(engine, db_uri, tables):
    schema_lines = []

    for table in tables:
        try:
            if db_uri == LOCALDB:
                df = pd.read_sql_query(
                    f"PRAGMA table_info('{table}')", engine
                )
                cols = df["name"].tolist()
            else:
                df = pd.read_sql_query(
                    f"SHOW COLUMNS FROM {table}", engine
                )
                cols = df["Field"].tolist()

            if cols:
                schema_lines.append(f"{table}({', '.join(cols)})")

        except Exception:
            pass

    return "\n".join(schema_lines)


# ============================================================
# Display Database Schema (Clean Table UI)
# ============================================================
st.subheader("📊 Database Schema")

try:
    tables = db.get_usable_table_names()

    if not tables:
        st.warning("No tables found in database.")
    else:
        for table in tables:
            with st.expander(f"📁 Table: {table}", expanded=False):

                if db_uri == LOCALDB:
                    schema_df = pd.read_sql_query(
                        f"PRAGMA table_info('{table}')", engine
                    )
                    schema_df = schema_df.rename(columns={
                        "name": "Column",
                        "type": "Data Type",
                        "notnull": "Not Null",
                        "pk": "Primary Key"
                    })[["Column", "Data Type", "Not Null", "Primary Key"]]

                else:
                    schema_df = pd.read_sql_query(
                        f"SHOW COLUMNS FROM {table}", engine
                    )
                    schema_df = schema_df.rename(columns={
                        "Field": "Column",
                        "Type": "Data Type",
                        "Null": "Nullable",
                        "Key": "Key"
                    })[["Column", "Data Type", "Nullable", "Key"]]

                st.dataframe(schema_df, use_container_width=True)

except Exception as e:
    st.error("Unable to load database schema.")
    st.exception(e)


# ============================================================
# Prompt Template
# ============================================================
SQL_PROMPT = PromptTemplate(
    input_variables=["question", "schema"],
    template="""
You are a SQL expert.
Use ONLY the column names exactly as provided in the schema.
Generate ONLY a valid SQL SELECT query.
No explanation. No markdown. No comments.

Schema:
{schema}

User question:
{question}

SQL:
"""
)


# ============================================================
# Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "results" not in st.session_state:
    st.session_state.results = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.results = []


# ============================================================
# Render Previous Chat
# ============================================================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg.get("has_table"):
        df = st.session_state.results[msg["table_index"]]
        st.dataframe(df, use_container_width=True)


# ============================================================
# Chat Input
# ============================================================
user_query = st.chat_input("Ask a question about your data")

if user_query:

    if not api_key:
        st.info("🔑 Please enter Groq API Key.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        try:
            schema_text = get_db_schema_text(engine, db_uri, tables)

            prompt = SQL_PROMPT.format(
                question=user_query,
                schema=schema_text,
            )

            sql = llm.invoke(prompt).content.strip()
            st.code(sql, language="sql")

            if not sql.lower().startswith("select"):
                st.error("Only SELECT queries are allowed.")
                st.stop()

            df = pd.read_sql_query(sql, engine)

            if df.empty:
                st.info("No results found.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "No results found."}
                )
            else:
                table_index = len(st.session_state.results)
                st.session_state.results.append(df)

                st.dataframe(df, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Result:",
                    "has_table": True,
                    "table_index": table_index
                })

        except Exception as e:
            st.error("❌ Query execution failed.")
            st.exception(e)
