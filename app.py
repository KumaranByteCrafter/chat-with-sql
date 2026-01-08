import streamlit as st
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine
import pandas as pd

from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

#app ui
st.set_page_config(page_title="Chat with SQL Database")
st.title("Chat with SQL Database")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = [
    "Use SQLite Database (student.db)",
    "Connect to MySQL Database",
]

selected_opt = st.sidebar.radio(
    "Choose the DB which you want to chat",
    radio_opt,
)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input("Groq API Key", type="password")


# llm Configuration

# llm Configuration

llm = None
if api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
else:
    st.info("🔑 Please enter your Groq API Key in the sidebar to start chatting.")



# db Configuration
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

    elif db_uri == MYSQL:
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



# Prompt

SQL_PROMPT = PromptTemplate(
    input_variables=["question", "tables"],
    template="""
You are a SQL expert.
Generate ONLY a valid SQL SELECT query.
Do not explain anything.
Do not add markdown.
Do not add comments.

Available tables:
{tables}

User question:
{question}

SQL:
"""
)



# session state initialization

if "messages" not in st.session_state:
    st.session_state.messages = []

if "results" not in st.session_state:
    st.session_state.results = []   # store dataframe


if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.results = []



# render previous Chat + tables
for i, msg in enumerate(st.session_state.messages):
    st.chat_message(msg["role"]).write(msg["content"])

    # if message has a table attached → render it
    if msg.get("has_table"):
        df = st.session_state.results[msg["table_index"]]
        st.dataframe(df, width="stretch")



# chat input
user_query = st.chat_input("Ask a question about your data")

if user_query and api_key:
    # store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        try:
            tables = db.get_usable_table_names()

            prompt = SQL_PROMPT.format(
                question=user_query,
                tables=", ".join(tables),
            )

            sql = llm.invoke(prompt).content.strip()
            st.code(sql, language="sql")

            if not sql.lower().startswith("select"):
                st.error("Only SELECT queries are allowed.")
                st.stop()

            # execute query
            df = pd.read_sql_query(sql, engine)

            if df.empty:
                st.info("No results found.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "No results found."}
                )
            else:
                # save dataframe
                table_index = len(st.session_state.results)
                st.session_state.results.append(df)

                # render table
                st.dataframe(df, width="stretch")

                # store assistant message with table reference
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Result:",
                    "has_table": True,
                    "table_index": table_index
                })

        except Exception as e:
            st.error(f"Error: {e}")
