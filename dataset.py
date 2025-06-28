import os
import requests
import overpy
import json
from datetime import datetime
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import time

# Load environment variables
load_dotenv()

# Fetch API keys from .env
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NREL_API_KEY = os.getenv("NREL_API_KEY")

# Simplified geocoding function for single address
def geocode_address(address):
    """
    Geocode a single address using Nominatim
    """
    geolocator = Nominatim(user_agent="uhi_model")
    
    try:
        time.sleep(1)  # Rate limiting for Nominatim
        location = geolocator.geocode(address)
        if location:
            print(f"✓ Found address: {address}")
            print(f"  Coordinates: {location.latitude}, {location.longitude}")
            print(f"  Full address: {location.address}")
            return (location.latitude, location.longitude)
        else:
            raise ValueError(f"Address '{address}' not found")
    except Exception as e:
        raise ValueError(f"Geocoding failed for '{address}': {str(e)}")

# Fetch solar radiation data (NREL API) - with error handling
def get_solar_data(lat, lon):
    if not NREL_API_KEY:
        print("Warning: NREL_API_KEY not found, using dummy data")
        return {"outputs": {"avg_ghi": 5.5}}  # Dummy data in kWh/m2/day
    
    # Using NREL Solar Resource Data API (updated endpoint)
    url = "https://developer.nrel.gov/api/solar/solar_resource/v1.json"
    params = {
        "api_key": NREL_API_KEY,
        "lat": lat,
        "lon": lon
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Extract annual average GHI (Global Horizontal Irradiance)
        if "outputs" in data and "avg_ghi" in data["outputs"]:
            return data
        else:
            print(f"Unexpected NREL API response structure: {data}")
            return {"outputs": {"avg_ghi": 5.5}}  # Fallback
            
    except requests.exceptions.RequestException as e:
        print(f"Solar data API error: {e}")
        return {"outputs": {"avg_ghi": 5.5}}  # Fallback data

# Fetch greenery/land-use data (Overpass API) - with error handling
def get_greenery_data(bbox):
    try:
        api = overpy.Overpass()
        query = f"""
            [out:json][timeout:25];
            (
              way["landuse"="forest"]({bbox});
              way["natural"="wood"]({bbox});
              node["natural"="tree"]({bbox});
            );
            out count;
        """
        result = api.query(query)
        return {
            "forest_areas": len(result.ways),
            "tree_nodes": len(result.nodes)
        }
    except Exception as e:
        print(f"Greenery data error: {e}")
        return {"forest_areas": 5, "tree_nodes": 20}  # Fallback data

# Fetch weather data (OpenWeatherMap API) - with error handling
def get_weather_data(lat, lon):
    if not OPENWEATHER_API_KEY:
        print("Warning: OPENWEATHER_API_KEY not found, using dummy data")
        return {"current": {"temp": 25.0, "humidity": 60}}
    
    # Try the correct API endpoint (v2.5 for free tier)
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "current": {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"]
            }
        }
    except requests.exceptions.RequestException as e:
        print(f"Weather data error: {e}")
        return {"current": {"temp": 25.0, "humidity": 60}}  # Fallback data

# Data saving functions
def create_data_directories():
    """Create necessary directories for saving data"""
    directories = [
        "data",
        "data/training_data",
        "data/raw_api_responses",
        "data/processed"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")

def save_raw_api_data(solar_data, greenery_data, weather_data, address, timestamp):
    """Save raw API responses for debugging and future analysis"""
    try:
        # Create filename with timestamp
        filename = f"data/raw_api_responses/api_data_{timestamp}.json"
        
        raw_data = {
            "timestamp": timestamp,
            "address": address,
            "solar_api_response": solar_data,
            "greenery_api_response": greenery_data,
            "weather_api_response": weather_data
        }
        
        with open(filename, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        print(f"✓ Raw API data saved to: {filename}")
        return filename
    
    except Exception as e:
        print(f"✗ Failed to save raw API data: {e}")
        return None

def save_training_data(training_data, timestamp):
    """Save processed training data"""
    try:
        # Save as JSON
        json_filename = f"data/training_data/training_data_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        csv_filename = f"data/training_data/training_data_{timestamp}.csv"
        
        # Flatten nested data for CSV
        flat_data = {}
        for key, value in training_data.items():
            if isinstance(value, tuple):  # coordinates
                flat_data[f"{key}_lat"] = value[0]
                flat_data[f"{key}_lon"] = value[1]
            else:
                flat_data[key] = value
        
        # Write CSV header and data
        with open(csv_filename, 'w') as f:
            # Write header
            f.write(','.join(flat_data.keys()) + '\n')
            # Write data
            f.write(','.join(str(v) for v in flat_data.values()) + '\n')
        
        print(f"✓ Training data saved to:")
        print(f"  JSON: {json_filename}")
        print(f"  CSV:  {csv_filename}")
        
        return json_filename, csv_filename
    
    except Exception as e:
        print(f"✗ Failed to save training data: {e}")
        return None, None

def append_to_master_dataset(training_data, timestamp):
    """Append new data to master training dataset"""
    try:
        master_file = "data/processed/master_training_data.json"
        
        # Load existing data or create new list
        if os.path.exists(master_file):
            with open(master_file, 'r') as f:
                master_data = json.load(f)
        else:
            master_data = []
        
        # Add timestamp to training data
        training_data_with_timestamp = training_data.copy()
        training_data_with_timestamp["collection_timestamp"] = timestamp
        
        # Append new data
        master_data.append(training_data_with_timestamp)
        
        # Save updated master dataset
        with open(master_file, 'w') as f:
            json.dump(master_data, f, indent=2, default=str)
        
        print(f"✓ Data appended to master dataset: {master_file}")
        print(f"  Total records in dataset: {len(master_data)}")
        
        return master_file
    
    except Exception as e:
        print(f"✗ Failed to append to master dataset: {e}")
        return None

# Main workflow with data saving
def main():
    address = "1090 Murfreesboro Pike Nashville"
    
    try:
        print("=== STARTING UHI MODEL DATA COLLECTION ===")
        
        # Step 0: Create data directories
        print("Setting up data directories...")
        create_data_directories()
        
        # Generate timestamp for this data collection run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Data collection timestamp: {timestamp}")
        
        # Step 1: Geocode address
        print(f"\nGeocoding address: {address}")
        lat, lon = geocode_address(address)
            
        bbox = f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}"  # ~1km bounding box
        print(f"Using coordinates: {lat}, {lon}")
        print(f"Bounding box: {bbox}")
        
        # Step 2: Fetch data from APIs
        print("\n=== FETCHING DATA FROM APIs ===")
        
        print("Fetching solar radiation data from NREL...")
        solar_data = get_solar_data(lat, lon)
        
        print("Fetching greenery/land use data from OpenStreetMap...")
        greenery_data = get_greenery_data(bbox)
        
        print("Fetching current weather data from OpenWeatherMap...")
        weather_data = get_weather_data(lat, lon)
        
        # Step 3: Save raw API responses
        print("\n=== SAVING RAW API DATA ===")
        raw_data_file = save_raw_api_data(solar_data, greenery_data, weather_data, address, timestamp)
        
        # Step 4: Prepare training dataset
        print("\n=== PREPARING TRAINING DATA ===")
        training_data = {
            "address": address,
            "coordinates": (lat, lon),
            "solar_ghi_avg": solar_data["outputs"]["avg_ghi"] if "outputs" in solar_data else "N/A",
            "greenery_score": greenery_data["tree_nodes"] + greenery_data["forest_areas"] * 10,
            "current_temperature": weather_data["current"]["temp"],
            "current_humidity": weather_data["current"]["humidity"],
            "tree_count": greenery_data["tree_nodes"],
            "forest_areas": greenery_data["forest_areas"],
            "data_collection_timestamp": timestamp
        }
        
        print("\n=== TRAINING DATA SUMMARY ===")
        for key, value in training_data.items():
            print(f"  {key}: {value}")
        
        # Step 5: Save processed training data
        print("\n=== SAVING TRAINING DATA ===")
        json_file, csv_file = save_training_data(training_data, timestamp)
        
        # Step 6: Append to master dataset
        print("\n=== UPDATING MASTER DATASET ===")
        master_file = append_to_master_dataset(training_data, timestamp)
        
        print(f"\n✓ Data collection successful for {address}")
        print(f"✓ All data saved with timestamp: {timestamp}")
        
        return {
            "success": True,
            "training_data": training_data,
            "files_created": {
                "raw_data": raw_data_file,
                "training_json": json_file,
                "training_csv": csv_file,
                "master_dataset": master_file
            }
        }
    
    except Exception as e:
        print(f"✗ Error in main workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def collect_multiple_locations(addresses):
    """Collect data for multiple addresses"""
    print("=== COLLECTING DATA FOR MULTIPLE LOCATIONS ===")
    results = []
    
    for i, address in enumerate(addresses, 1):
        print(f"\n--- Processing location {i}/{len(addresses)}: {address} ---")
        result = main_single_location(address)
        results.append(result)
        
        # Add delay between requests to be respectful to APIs
        if i < len(addresses):
            print("Waiting 5 seconds before next location...")
            time.sleep(5)
    
    return results

def main_single_location(address):
    """Modified main function for single location (used by multiple location collector)"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Geocode address
        lat, lon = geocode_address(address)
        bbox = f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}"
        
        # Fetch data from APIs
        solar_data = get_solar_data(lat, lon)
        greenery_data = get_greenery_data(bbox)
        weather_data = get_weather_data(lat, lon)
        
        # Save raw API responses
        save_raw_api_data(solar_data, greenery_data, weather_data, address, timestamp)
        
        # Prepare training dataset
        training_data = {
            "address": address,
            "coordinates": (lat, lon),
            "solar_ghi_avg": solar_data["outputs"]["avg_ghi"] if "outputs" in solar_data else "N/A",
            "greenery_score": greenery_data["tree_nodes"] + greenery_data["forest_areas"] * 10,
            "current_temperature": weather_data["current"]["temp"],
            "current_humidity": weather_data["current"]["humidity"],
            "tree_count": greenery_data["tree_nodes"],
            "forest_areas": greenery_data["forest_areas"],
            "data_collection_timestamp": timestamp
        }
        
        # Save processed training data
        save_training_data(training_data, timestamp)
        append_to_master_dataset(training_data, timestamp)
        
        return {"success": True, "address": address, "data": training_data}
    
    except Exception as e:
        return {"success": False, "address": address, "error": str(e)}

if __name__ == "__main__":
    # Option 1: Collect data for single location
    print("=== SINGLE LOCATION DATA COLLECTION ===")
    result = main()
    
    if result and result["success"]:
        print("\n=== SUCCESS ===")
        print("Training data ready for UHI model!")
        print("\nFiles created:")
        for file_type, filepath in result["files_created"].items():
            if filepath:
                print(f"  {file_type}: {filepath}")
    else:
        print("\n=== FAILED ===")
        print("Data collection failed. Check API keys and network connection.")
    
    # Option 2: Uncomment below to collect data for multiple locations
    """
    print("\n" + "="*60)
    print("=== MULTIPLE LOCATION DATA COLLECTION ===")
    
    # Example addresses in Nashville area
    addresses = [
        "1090 Murfreesboro Pike Nashville",
        "Music Row Nashville TN",
        "Vanderbilt University Nashville TN",
        "Downtown Nashville TN",
        "Nashville International Airport TN"
    ]
    
    results = collect_multiple_locations(addresses)
    
    print(f"\n=== MULTIPLE LOCATION RESULTS ===")
    successful = sum(1 for r in results if r["success"])
    print(f"Successfully collected data for {successful}/{len(addresses)} locations")
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['address']}")
    """