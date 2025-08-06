# kavak-travel-assistant
An AI-powered travel assistant that helps users with visa and flight information using OpenAI, FAISS, and Streamlit.

## 🚀 Overview

Kavak Travel Assistant is a modular, prompt-driven chatbot that handles:

* Complex flight searches
* Visa and airline policy queries
* Contextual knowledge retrieval (RAG)
* Natural language understanding with OpenAI function calling

Built as a technical case study for Kavak.

## ✨ Features

* *Natural Language Flight Search:* Handles multi-criteria queries like "round-trip to Tokyo in August under \$1000 with Star Alliance."
* *RAG-based Visa & Policy Info:* Retrieves info from a vectorized knowledge base using sentence-transformers + FAISS.
* *OpenAI Function Calling:* Routes user intent to appropriate tools: search_flights, get_visa_info, get_policy_info.
* *Modular Architecture:* Core logic in main_v1.py, UI in app.py.
* *Streamlit Chat UI:* Simple, intuitive frontend for interaction.

## 🦾 Project Structure


├── main.py          # Core logic and tool implementations
├── app.py      # Streamlit UI
├── data/
│   ├── flights.json      # Mock flight data
│   └── visa_rules.md     # Knowledge base docs
├── .env         # Template for environment variables
├── requirements.txt      # Dependencies
└── README.md             # Project guide


> Note: Data is loaded internally for self-contained testing.

## 🛠️ Installation

1. *Clone the repo*

   bash
   git clone <repo-url>
   cd <repo-name>
   

2. *Create virtual environment*

   bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   

3. *Install dependencies*

   bash
   pip install -r requirements.txt
   

4. *Set OpenAI API Key*

   Option A: Create .env file in root:

   env
   OPENAI_API_KEY="your_api_key_here"
   

   Option B: Enter in Streamlit sidebar at runtime.

5. *Run the app*

   bash
   streamlit run app.py
   

## 🧠 System Logic

1. *Initialization:* Loads flight data, visa/policy knowledge base, and OpenAI client.
2. *User Query → GPT:* Entire conversation sent to OpenAI with tool schemas.
3. *LLM Intent Recognition:* GPT chooses a tool + parameters (e.g., destination, date).
4. *Tool Execution:* Runs corresponding function (e.g., search\_flights).
5. *Tool Output → GPT:* Tool result sent back to GPT.
6. *Final Response:* GPT formats user-facing response.

## 🧪 Sample Interactions

*Flight Search:*

> "Find me a round-trip to Tokyo in August with Star Alliance, under \$1000."

Returns matching flights with prices, airlines, and layover info.

*Visa Info:*

> "Do UAE citizens need a visa for Germany?"

Returns clear visa requirements from the RAG database.

*Policy Info:*

> "What is the baggage policy for economy class?"

Returns baggage limits based on policy documents.

## 🧹 Evaluation Highlights

* *Prompt Engineering:* Tool schemas and system prompt guide GPT behavior.
* *RAG:* Uses embeddings + FAISS for retrieval-based responses.
* *Function Calling:* Accurate tool routing and parameter extraction.
* *Clean Code:* Modular, scalable, type-hinted, and well-documented.
* *Conversational UX:* Helpful and human-like assistant tone.
* *Initiative:* Optional Streamlit UI and enhanced RAG system.

---

Thanks for reviewing! This project demonstrates advanced agent-based architecture, modular design, and robust prompt-driven AI workflows.
