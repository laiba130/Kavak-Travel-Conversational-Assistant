# ✈️ Kavak Travel Assistant

An AI-powered travel planner that simplifies international trip planning through intelligent conversation.

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
* *Modular Architecture:* Core logic in app_logic.py, UI in streamlit_app.py.
* *Streamlit Chat UI:* Simple, intuitive frontend for interaction.

## 🦾 Project Structure

text
├── main.py          # Core logic and tool implementations
├── app.py           # Streamlit UI
├── data/
│   ├── flights.json      # Mock flight data
│   └── visa_rules.md     # Knowledge base docs
├── .env                 # Template for environment variables
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

*Use Case 1*

> user: Find me flights from Dubai to Tokyo

assistant:

I found 3 flight options for you from Dubai to Tokyo:


ANA (Star Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-16 to 2024-08-31
    Layovers: Doha
    Price: $850
    Refundable: No

Turkish Airlines (Star Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-15 to 2024-08-30
    Layovers: Istanbul
    Price: $950
    Refundable: Yes

Emirates (Non-Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-15 to 2024-08-30
    Direct flight
    Price: $1200
    Refundable: Yes


Please let me know if you would like more details or assistance with booking any of these flights.

*Use Case 2*

> user: Show me refundable flights from Dubai to Tokyo under \$1000

assistant:

I found one refundable flight option for you from Dubai to Tokyo under \$1000:


Turkish Airlines (Star Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-15 to 2024-08-30
    Layovers: Istanbul
    Price: $950
    Refundable: Yes


Let me know if you need more information or if you would like to proceed with booking this flight.

*Use Case 3*

> user: Show me all flights to Tokyo but exclude overnight layovers

assistant:

I found 3 flight options for you to Tokyo without overnight layovers:


ANA (Star Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-16 to 2024-08-31
    Layovers: Doha
    Price: $850
    Refundable: No

Turkish Airlines (Star Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-15 to 2024-08-30
    Layovers: Istanbul
    Price: $950
    Refundable: Yes

Emirates (Non-Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-15 to 2024-08-30
    Direct flight
    Price: $1200
    Refundable: Yes


Please let me know if you need more information or assistance with booking any of these flights.

*Use Case 4*

> user: I would love option 01

assistant:

Great choice! I can assist you with booking the flight on ANA from Dubai to Tokyo with a layover in Doha.

Before we proceed, could you please provide me with the passenger's full name, date of birth, and contact information (email and phone number) for the booking?

*Use Case 5*

> user: Do I need a visa for Japan?

assistant:

Based on the information provided, as a UAE passport holder, you can enter Japan visa-free for up to 30 days for tourism purposes. Just make sure your passport is valid for at least 6 months. If you have any specific questions or need further assistance with your travel plans, feel free to ask!

*Use Case 6*

> user: I want Star Alliance flights from Dubai to Tokyo, departing August 15th, returning August 30th, refundable tickets only, under \$1200

assistant:

I found a flight option for you that meets your criteria:


Turkish Airlines (Star Alliance)
    Route: Dubai → Tokyo
    Dates: 2024-08-15 to 2024-08-30
    Layovers: Istanbul
    Price: $950
    Refundable: Yes


Please let me know if you would like to proceed with booking this flight or if you need any further assistance.

## 🧹 Evaluation Highlights

* *Prompt Engineering:* Tool schemas and system prompt guide GPT behavior.
* *RAG:* Uses embeddings + FAISS for retrieval-based responses.
* *Function Calling:* Accurate tool routing and parameter extraction.
* *Clean Code:* Modular, scalable, type-hinted, and well-documented.
* *Conversational UX:* Helpful and human-like assistant tone.
* *Initiative:* Optional Streamlit UI and enhanced RAG system.

---

Thanks for reviewing! This project demonstrates advanced agent-based architecture, modular design, and robust prompt-driven AI workflows.