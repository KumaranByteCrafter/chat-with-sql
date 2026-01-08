# 🦜 Chat with SQL Database (Streamlit + LangChain + Groq)

A web application that allows users to query a SQL database using natural language.  
The app converts user questions into SQL queries using an LLM (Groq + Llama 3.1) and displays the results interactively in Streamlit.

### Supported Capabilities
- ✅ SQLite database (local)
- ✅ MySQL database (remote)
- ✅ Natural language → SQL generation
- ✅ Live table rendering
- ✅ Session-based chat history

---

## 🚀 Features

- Natural language query interface
- Automatic SQL generation using LLM
- Safe execution (only SELECT queries allowed)
- Interactive table rendering using Streamlit
- Session persistence for chat history and results
- Supports multiple database backends
- Cached database connections for better performance

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| Backend | Python |
| LLM | Groq API (Llama 3.1) |
| Database | SQLite, MySQL |
| ORM / Engine | SQLAlchemy |
| NL → SQL | LangChain |
| Data Processing | Pandas |

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/KumaranByteCrafter/chat-with-sql.git
cd chat-with-sql
```


## ⚙️ Virtual Environment Setup

Create and activate a virtual environment:

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### Run the Application 
```bash
streamlit run app.py

```

---

# 🧠 System Architecture

This application follows a layered architecture to safely convert natural language questions into executable SQL queries and render results interactively.

---

## 🤖 LLM Layer

### Model Configuration

- **Model:** llama-3.1-8b-instant  
- **Provider:** Groq  
- **Temperature:** 0 (deterministic SQL generation)

### Responsibilities

- Converts user natural language questions into valid SQL queries.
- Enforces strict output rules using prompt engineering.
- Prevents markdown formatting, explanations, and non-SQL responses.
- Ensures output contains only executable SQL.

### Prompt Rules

- Only valid `SELECT` statements are allowed.
- No markdown formatting.
- No comments or explanations.
- No non-SQL text in the response.

This guarantees safe and predictable query generation.

---

## 🗄️ Database Layer

### Supported Databases

- SQLite (local file-based database)  
- MySQL (remote database server)

### Implementation

- Database connectivity is managed using SQLAlchemy Engine.
- SQLite databases are opened in read-only mode for safety.
- MySQL connections use the mysql-connector driver.
- Database connections are cached using `@st.cache_resource` to improve performance and reduce reconnection overhead.

### Configuration

- **SQLite:** No configuration required. The database file path is predefined.
- **MySQL:** Credentials are entered dynamically through the Streamlit sidebar:
  - Host  
  - Port  
  - Username  
  - Password  
  - Database Name  

---

## ⚙️ Execution Layer

### Processing Flow

1. User submits a natural language question.  
2. Application fetches available table names and schema metadata from the database.  
3. A dynamic prompt is constructed using the table metadata.  
4. The LLM generates a SQL `SELECT` query.  
5. The generated SQL is validated for safety.  
6. Query is executed using Pandas and SQLAlchemy.  
7. Results are rendered interactively in Streamlit.  
8. Query history and results are stored in Streamlit session state.  

This flow ensures safe execution, consistent performance, and a persistent user experience.

---

## ✅ Why This Design Works

- **Safety:** Only SELECT queries are allowed. No destructive operations.  
- **Determinism:** Temperature is set to 0 to avoid unpredictable SQL generation.  
- **Performance:** Database connections are cached.  
- **Scalability:** Supports both local and remote databases.  
- **Maintainability:** Clear separation of LLM, Database, and Execution layers. 

## 🌐 Live Application

The application is deployed and accessible at:  
👉 https://chat-db-sql.streamlit.app/

Users can interact with the application directly from the browser.

