#!/usr/bin/env python3
"""
Conversational Travel Assistant - Kavak Technical Case Study
Using OpenAI with Function Calling
"""

import json
import re
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FlightSearchCriteria:
    """Structured flight search criteria"""
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    preferred_airlines: List[str] = None
    preferred_alliances: List[str] = None
    avoid_overnight_layovers: bool = False
    max_price: Optional[float] = None
    refundable_only: bool = False

@dataclass
class Flight:
    """Flight data structure"""
    airline: str
    alliance: str
    origin: str
    destination: str
    departure_date: str
    return_date: str
    layovers: List[str]
    price_usd: float
    refundable: bool
    
    def matches_criteria(self, criteria: FlightSearchCriteria) -> bool:
        """Check if flight matches search criteria"""
        # Origin/Destination matching
        if criteria.origin and criteria.origin.lower() not in self.origin.lower():
            return False
        if criteria.destination and criteria.destination.lower() not in self.destination.lower():
            return False
            
        # Date matching (simplified)
        if criteria.departure_date and criteria.departure_date != self.departure_date:
            return False
        if criteria.return_date and criteria.return_date != self.return_date:
            return False
            
        # Airline/Alliance preferences
        if criteria.preferred_airlines:
            if not any(airline.lower() in self.airline.lower() for airline in criteria.preferred_airlines):
                return False
                
        if criteria.preferred_alliances:
            if not any(alliance.lower() in self.alliance.lower() for alliance in criteria.preferred_alliances):
                return False
                
        # Price filter
        if criteria.max_price and self.price_usd > criteria.max_price:
            return False
            
        # Refundable filter
        if criteria.refundable_only and not self.refundable:
            return False
            
        return True

class KnowledgeBase:
    """RAG-based knowledge base for visa and policy information"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_documents(self, file_path: str):
        """Load documents from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split content into chunks
            chunks = self._split_into_chunks(content)
            self.documents = chunks
            
            # Create embeddings
            self.embeddings = self.encoder.encode(chunks)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            # logger.info(f"Loaded {len(chunks)} document chunks")
            
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {file_path}")
            self._create_sample_knowledge_base()
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _create_sample_knowledge_base(self):
        """Create sample knowledge base from predefined data"""
        visa_content = """# Travel Requirements and Policies

## Visa Information

### Japan
UAE passport holders can enter Japan visa-free for up to 30 days for tourism. Passport must be valid for at least 6 months.

### Germany
UAE passport holders need a Schengen visa to enter Germany. Processing time is typically 15 business days.

### USA
UAE passport holders need a B1/B2 visa to enter the USA. ESTA is not available for UAE passports.

## Airline Policies

### Refund Policy
Refundable tickets can be canceled up to 48 hours before departure, subject to a 10% processing fee.
Non-refundable tickets may be eligible for travel credits minus administrative fees.

### Baggage Policy
Economy class passengers are allowed one 23kg checked bag and one carry-on bag.
Business class passengers are allowed two 32kg checked bags.

### Star Alliance Members
Star Alliance members include United Airlines, Lufthansa, Turkish Airlines, ANA, Singapore Airlines, and Swiss International.

### Layover Information
Overnight layovers are considered layovers longer than 8 hours during nighttime (10 PM to 6 AM).
Short layovers under 2 hours may be risky for international connections.
"""
        
        chunks = self._split_into_chunks(visa_content)
        self.documents = chunks
        self.embeddings = self.encoder.encode(chunks)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        # logger.info("Created sample knowledge base")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant documents"""
        if self.index is None:
            return []
            
        query_embedding = self.encoder.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, score in zip(indices[0], scores[0]):
            if score > 0.3:  # Similarity threshold
                results.append(self.documents[i])
                
        return results

class FlightDatabase:
    """Mock flight database with search capabilities"""
    
    def __init__(self):
        self.flights = []
        
    def load_flights(self, file_path: str = None, flight_data: List[Dict] = None):
        """Load flights from JSON file or provided data"""
        if flight_data:
            self.flights = [Flight(**flight) for flight in flight_data]
            # logger.info(f"Loaded {len(self.flights)} flights from provided data")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                flight_data = json.load(f)
                
            self.flights = [Flight(**flight) for flight in flight_data]
            # logger.info(f"Loaded {len(self.flights)} flights")
            
        except FileNotFoundError:
            logger.warning(f"Flight data file not found: {file_path}")
            self._create_sample_flights()
    
    def _create_sample_flights(self):
        """Create sample flight data"""
        sample_flights = [
            {
                "airline": "Turkish Airlines",
                "alliance": "Star Alliance",
                "origin": "Dubai",
                "destination": "Tokyo",
                "departure_date": "2024-08-15",
                "return_date": "2024-08-30",
                "layovers": ["Istanbul"],
                "price_usd": 950,
                "refundable": True
            },
            {
                "airline": "Emirates",
                "alliance": "Non-Alliance",
                "origin": "Dubai",
                "destination": "Tokyo",
                "departure_date": "2024-08-15",
                "return_date": "2024-08-30",
                "layovers": [],
                "price_usd": 1200,
                "refundable": True
            },
            {
                "airline": "ANA",
                "alliance": "Star Alliance",
                "origin": "Dubai",
                "destination": "Tokyo",
                "departure_date": "2024-08-16",
                "return_date": "2024-08-31",
                "layovers": ["Doha"],
                "price_usd": 850,
                "refundable": False
            },
            {
                "airline": "Lufthansa",
                "alliance": "Star Alliance",
                "origin": "Dubai",
                "destination": "Berlin",
                "departure_date": "2024-08-20",
                "return_date": "2024-09-05",
                "layovers": ["Frankfurt"],
                "price_usd": 750,
                "refundable": True
            }
        ]
        
        self.flights = [Flight(**flight) for flight in sample_flights]
        # logger.info("Created sample flight database")
    
    def search_flights(self, criteria: FlightSearchCriteria) -> List[Flight]:
        """Search flights based on criteria"""
        matching_flights = [
            flight for flight in self.flights
            if flight.matches_criteria(criteria)
        ]
        
        # Sort by price
        matching_flights.sort(key=lambda x: x.price_usd)
        
        return matching_flights

class TravelAssistant:
    """Main travel assistant with OpenAI function calling capabilities"""
    
    def __init__(self, openai_api_key: str):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize components
        self.flight_db = FlightDatabase()
        self.knowledge_base = KnowledgeBase()
        
        # Load data with provided dataset
        flight_data = [
            {
                "airline": "Turkish Airlines",
                "alliance": "Star Alliance",
                "origin": "Dubai",
                "destination": "Tokyo",
                "departure_date": "2024-08-15",
                "return_date": "2024-08-30",
                "layovers": ["Istanbul"],
                "price_usd": 950,
                "refundable": True
            },
            {
                "airline": "Emirates",
                "alliance": "Non-Alliance",
                "origin": "Dubai",
                "destination": "Tokyo",
                "departure_date": "2024-08-15",
                "return_date": "2024-08-30",
                "layovers": [],
                "price_usd": 1200,
                "refundable": True
            },
            {
                "airline": "ANA",
                "alliance": "Star Alliance",
                "origin": "Dubai",
                "destination": "Tokyo",
                "departure_date": "2024-08-16",
                "return_date": "2024-08-31",
                "layovers": ["Doha"],
                "price_usd": 850,
                "refundable": False
            },
            {
                "airline": "Lufthansa",
                "alliance": "Star Alliance",
                "origin": "Dubai",
                "destination": "Berlin",
                "departure_date": "2024-08-20",
                "return_date": "2024-09-05",
                "layovers": ["Frankfurt"],
                "price_usd": 750,
                "refundable": True
            }
        ]
        
        self.flight_db.load_flights(flight_data=flight_data)
        self.knowledge_base.load_documents('data/visa_rules.md')
        
        # Define function schemas for OpenAI
        self.function_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search for flights based on specified criteria",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {
                                "type": "string",
                                "description": "Origin city or airport"
                            },
                            "destination": {
                                "type": "string", 
                                "description": "Destination city or airport"
                            },
                            "departure_date": {
                                "type": "string",
                                "description": "Departure date in YYYY-MM-DD format"
                            },
                            "return_date": {
                                "type": "string",
                                "description": "Return date in YYYY-MM-DD format"
                            },
                            "preferred_airlines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of preferred airlines"
                            },
                            "preferred_alliances": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of preferred airline alliances (e.g., 'Star Alliance', 'SkyTeam', 'OneWorld')"
                            },
                            "avoid_overnight_layovers": {
                                "type": "boolean",
                                "description": "Avoid flights with overnight layovers"
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Maximum price in USD"
                            },
                            "refundable_only": {
                                "type": "boolean",
                                "description": "Only show refundable tickets"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_visa_info",
                    "description": "Get visa requirements for a specific country",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "description": "Destination country"
                            },
                            "passport": {
                                "type": "string",
                                "description": "Passport issuing country",
                                "default": "UAE"
                            }
                        },
                        "required": ["country"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_policy_info",
                    "description": "Get airline policy information (refund, baggage, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Policy topic (e.g., 'refund', 'baggage', 'layover')"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            }
        ]
    
    def search_flights(self, **kwargs) -> str:
        """Search flights function"""
        criteria_dict = {k: v for k, v in kwargs.items() if v is not None}
        criteria = FlightSearchCriteria(**criteria_dict)
        flights = self.flight_db.search_flights(criteria)
        
        if not flights:
            return "No flights found matching your criteria. Let me show you all available flights to help you plan better:\n\n" + self._show_all_flights()
        
        result = f"Found {len(flights)} flight(s) matching your criteria:\n\n"
        
        for i, flight in enumerate(flights, 1):
            result += f"**Option {i}:**\n"
            result += f"• **{flight.airline}** ({flight.alliance})\n"
            result += f"• **Route:** {flight.origin} → {flight.destination}\n"
            result += f"• **Dates:** {flight.departure_date} to {flight.return_date}\n"
            
            if flight.layovers:
                result += f"• **Layovers:** {', '.join(flight.layovers)}\n"
            else:
                result += f"• **Direct flight**\n"
                
            result += f"• **Price:** ${flight.price_usd}\n"
            result += f"• **Refundable:** {'Yes' if flight.refundable else 'No'}\n\n"
        
        return result
    
    def _show_all_flights(self) -> str:
        """Show all available flights for debugging"""
        all_flights = self.flight_db.flights
        result = "All available flights:\n\n"
        
        for i, flight in enumerate(all_flights, 1):
            result += f"**Flight {i}:**\n"
            result += f"• **{flight.airline}** ({flight.alliance})\n"
            result += f"• **Route:** {flight.origin} → {flight.destination}\n"
            result += f"• **Dates:** {flight.departure_date} to {flight.return_date}\n"
            if flight.layovers:
                result += f"• **Layovers:** {', '.join(flight.layovers)}\n"
            else:
                result += f"• **Direct flight**\n"
            result += f"• **Price:** ${flight.price_usd}\n"
            result += f"• **Refundable:** {'Yes' if flight.refundable else 'No'}\n\n"
        
        return result
    
    def get_visa_info(self, country: str, passport: str = "UAE") -> str:
        """Get visa information function"""
        query = f"{passport} passport {country} visa requirements"
        relevant_docs = self.knowledge_base.search(query)
        
        if relevant_docs:
            return f"**Visa Information for {country}:**\n\n" + " ".join(relevant_docs)
        else:
            return f"No specific visa information found for {passport} passport holders traveling to {country}."
    
    def get_policy_info(self, topic: str) -> str:
        """Get policy information function"""
        relevant_docs = self.knowledge_base.search(topic)
        
        if relevant_docs:
            return f"**Policy Information ({topic}):**\n\n" + " ".join(relevant_docs)
        else:
            return f"No specific policy information found for: {topic}"
    
    def process_query(self, user_query: str, conversation_history: List[Dict]) -> str:
        """Process user query using OpenAI with function calling"""
        
        # Add user message to a temporary conversation list
        temp_history = conversation_history + [
            {"role": "user", "content": user_query}
        ]
        
        try:
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=temp_history,
                tools=self.function_schemas,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            response_message = response.choices[0].message
            
            # Check if the model wants to call functions
            if response_message.tool_calls:
                
                # Add the assistant's message to conversation
                temp_history.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": response_message.tool_calls
                })
                
                # Process each function call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the appropriate function
                    if function_name == "search_flights":
                        function_result = self.search_flights(**function_args)
                    elif function_name == "get_visa_info":
                        function_result = self.get_visa_info(**function_args)
                    elif function_name == "get_policy_info":
                        function_result = self.get_policy_info(**function_args)
                    else:
                        function_result = f"Unknown function: {function_name}"
                    
                    # Add function result to conversation
                    temp_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_result
                    })
                
                # Get final response from OpenAI
                final_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=temp_history,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_message = final_response.choices[0].message.content
                
                return final_message or "I found some results but couldn't format the response properly."
            
            else:
                # No function calls needed, return direct response
                content = response_message.content or "I'm here to help with your travel planning!"
                return content
                
        except Exception as e:
            logger.error(f"Error processing query with OpenAI: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."