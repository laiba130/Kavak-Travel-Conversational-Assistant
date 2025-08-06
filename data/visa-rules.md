# Flight Data Description

This JSON file contains flight itinerary data with detailed information about airlines, routes, dates, prices, and refundability.

## Structure

The JSON is an array of flight objects. Each flight object contains the following fields:

- **airline**: The name of the airline (e.g., "Turkish Airlines").
- **alliance**: Airline alliance group (e.g., "Star Alliance", "Non-Alliance").
- **origin**: Departure city (e.g., "Dubai").
- **destination**: Arrival city (e.g., "Tokyo").
- **departure_date**: Flight departure date (YYYY-MM-DD format).
- **return_date**: Flight return date (YYYY-MM-DD format).
- **layovers**: An array listing any layover cities during the flight.
- **price_usd**: Ticket price in US dollars.
- **refundable**: Boolean indicating if the ticket is refundable (`true` or `false`).

## Example Entry

```json
{
  "airline": "Turkish Airlines",
  "alliance": "Star Alliance",
  "origin": "Dubai",
  "destination": "Tokyo",
  "departure_date": "2024-08-15",
  "return_date": "2024-08-30",
  "layovers": ["Istanbul"],
  "price_usd": 950,
  "refundable": true
}
