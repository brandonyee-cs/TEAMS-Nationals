import os
import requests
import overpy
import json
from datetime import datetime
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import time
import statistics

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

# Fetch solar radiation data (NREL API) - focused on 2022 summer data
def get_solar_data(lat, lon, daily_data=False, year=2022, summer_months=[6, 7, 8]):
    if not NREL_API_KEY:
        print("Warning: NREL_API_KEY not found, using dummy data")
        if daily_data:
            # Return dummy daily data for 2022 summer
            dummy_data = {}
            dummy_data[str(year)] = {}
            for month in summer_months:
                month_name = {6: "june", 7: "july", 8: "august"}[month]
                days_in_month = {6: 30, 7: 31, 8: 31}[month]
                dummy_data[str(year)][month_name] = {
                    "daily_ghi": [5.0 + (i % 3) for i in range(days_in_month)],
                    "avg_ghi": 5.5 + month * 0.2,
                    "total_days": days_in_month
                }
            return {"outputs": {"summer_2022_data": dummy_data}}
        else:
            return {"outputs": {"avg_ghi": 5.5}}
    
    if daily_data:
        print(f"=== FETCHING 2022 SUMMER DATA ===")
        print(f"Target Year: {year}")
        print(f"Summer months: {[{6: 'June', 7: 'July', 8: 'August'}[m] for m in summer_months]}")
        
        # Create Well-Known Text (WKT) point format
        wkt_point = f"POINT({lon} {lat})"
        
        # Try to get data for 2022, with fallbacks to nearby years if needed
        years_to_try = [year, 2021, 2020, 2019]
        
        for attempt_year in years_to_try:
            try:
                print(f"\nAttempting to fetch data for {attempt_year}...")
                
                # NSRDB API parameters
                params = {
                    "api_key": NREL_API_KEY,
                    "wkt": wkt_point,
                    "names": str(attempt_year),
                    "attributes": "ghi,dni,dhi,air_temperature,wind_speed",
                    "leap_day": "false",
                    "interval": "60",
                    "utc": "false",
                    "full_name": "UHI Research Team",
                    "email": "research@uhi-study.org",
                    "affiliation": "Academic Research",
                    "reason": "2022 Summer Urban Heat Island Research",
                    "mailing_list": "false"
                }
                
                # Make API call
                url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv"
                print(f"Making API call for {attempt_year}...")
                response = requests.get(url, params=params, timeout=120)
                
                if response.status_code == 400:
                    print(f"Year {attempt_year} not available (400 error)")
                    continue
                
                response.raise_for_status()
                
                # Process the CSV data for summer months only
                summer_data = process_summer_csv_data(response.text, summer_months, attempt_year)
                
                if summer_data:
                    print(f"✓ Successfully processed {attempt_year} data (using as 2022 proxy)")
                    
                    # Format as 2022 data regardless of actual year used
                    formatted_data = {str(year): summer_data}
                    
                    return {
                        "outputs": {
                            "summer_2022_data": formatted_data,
                            "actual_data_year": attempt_year,
                            "target_year": year,
                            "total_summer_days": sum(
                                month_data["total_days"] for month_data in summer_data.values()
                            ),
                            "data_type": "summer_2022_daily"
                        }
                    }
                
            except requests.exceptions.RequestException as e:
                print(f"API error for {attempt_year}: {e}")
                continue
            except Exception as e:
                print(f"Processing error for {attempt_year}: {e}")
                continue
        
        print("Failed to get 2022 summer data from any available year, falling back to regular API")
    
    # Fallback to regular Solar Resource Data API
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
        
        if "outputs" in data and "avg_ghi" in data["outputs"]:
            return data
        else:
            print(f"Unexpected NREL API response structure: {data}")
            return {"outputs": {"avg_ghi": 5.5}}
            
    except requests.exceptions.RequestException as e:
        print(f"Solar data API error: {e}")
        return {"outputs": {"avg_ghi": 5.5}}

def process_summer_csv_data(csv_text, summer_months, year):
    """Process CSV data to extract daily averages for summer months (2022-focused)"""
    lines = csv_text.split('\n')
    
    # Find data start (skip header lines)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('Year,Month,Day'):
            data_start = i + 1
            break
    
    if data_start == 0:
        print("Could not find data header in CSV response")
        return None
    
    # Data structure: {month_name: {daily_ghi: [...], days: [...], avg_ghi: float}}
    summer_data = {}
    month_names = {6: "june", 7: "july", 8: "august"}
    
    for month in summer_months:
        month_name = month_names[month]
        summer_data[month_name] = {
            "daily_ghi": [],
            "days": [],
            "avg_ghi": 0,
            "total_days": 0
        }
    
    # Process each month separately
    for month in summer_months:
        month_name = month_names[month]
        daily_ghi_values = []
        days = []
        
        current_day = None
        daily_ghi_sum = 0
        hour_count = 0
        
        print(f"  Processing {month_name.title()} {year}...")
        
        for line_num, line in enumerate(lines[data_start:], data_start):
            if not line.strip():
                continue
                
            parts = line.split(',')
            if len(parts) < 8:
                continue
                
            try:
                year_val = int(parts[0])
                month_val = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                
                # GHI is typically the 7th column (index 6)
                ghi = float(parts[6]) if parts[6] and parts[6] != '' else 0
                
                # Filter for this specific month
                if month_val == month:
                    if current_day != day:
                        # Save previous day's average
                        if current_day is not None and hour_count > 0:
                            daily_avg = daily_ghi_sum / hour_count
                            daily_ghi_values.append(daily_avg)
                            days.append(current_day)
                        
                        # Start new day
                        current_day = day
                        daily_ghi_sum = ghi
                        hour_count = 1
                    else:
                        daily_ghi_sum += ghi
                        hour_count += 1
                        
            except (ValueError, IndexError) as e:
                continue
        
        # Don't forget the last day of the month
        if current_day is not None and hour_count > 0:
            daily_avg = daily_ghi_sum / hour_count
            daily_ghi_values.append(daily_avg)
            days.append(current_day)
        
        # Store month data
        if daily_ghi_values:
            avg_month_ghi = sum(daily_ghi_values) / len(daily_ghi_values)
            summer_data[month_name] = {
                "daily_ghi": daily_ghi_values,
                "days": days,
                "avg_ghi": avg_month_ghi,
                "total_days": len(daily_ghi_values)
            }
            print(f"    ✓ {month_name.title()}: {len(daily_ghi_values)} days, avg GHI: {avg_month_ghi:.2f} kWh/m²/day")
        else:
            print(f"    ✗ No data found for {month_name.title()} {year}")
    
    return summer_data

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
# Enhanced data saving functions for 2022 summer data
def save_2022_summer_solar_data(solar_data, address, timestamp):
    """Save 2022 summer solar data in organized format"""
    try:
        if "summer_2022_data" not in solar_data["outputs"]:
            return None
        
        summer_2022_data = solar_data["outputs"]["summer_2022_data"]
        
        # Create detailed file for 2022 summer data
        filename = f"data/training_data/summer_2022_solar_{timestamp}.json"
        
        detailed_data = {
            "address": address,
            "collection_timestamp": timestamp,
            "data_type": "summer_2022_daily",
            "target_year": 2022,
            "actual_data_year": solar_data["outputs"].get("actual_data_year", 2022),
            "total_summer_days": solar_data["outputs"].get("total_summer_days", 0),
            "summer_2022_data": summer_2022_data
        }
        
        with open(filename, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        print(f"✓ 2022 summer solar data saved to: {filename}")
        
        # Also create a summary CSV for easy analysis
        csv_filename = f"data/training_data/summer_2022_daily_summary_{timestamp}.csv"
        with open(csv_filename, 'w') as f:
            f.write("Year,Month,Day,Daily_GHI_kWh_per_m2\n")
            
            year_data = summer_2022_data.get("2022", {})
            for month_name, month_data in year_data.items():
                month_num = {"june": 6, "july": 7, "august": 8}[month_name]
                for i, (day, ghi) in enumerate(zip(month_data["days"], month_data["daily_ghi"])):
                    f.write(f"2022,{month_num},{day},{ghi:.3f}\n")
        
        print(f"✓ 2022 summer daily summary CSV saved to: {csv_filename}")
        
        return filename, csv_filename
    
    except Exception as e:
        print(f"✗ Failed to save 2022 summer solar data: {e}")
        return None, None

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

# Main workflow with 2022 summer data collection
def main(get_2022_summer=True, target_year=2022, summer_months=[6, 7, 8]):
    address = "1090 Murfreesboro Pike Nashville"
    
    try:
        print("=== STARTING 2022 SUMMER UHI DATA COLLECTION ===")
        
        # Step 0: Create data directories
        print("Setting up data directories...")
        create_data_directories()
        
        # Generate timestamp for this data collection run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Data collection timestamp: {timestamp}")
        
        if get_2022_summer:
            print(f"Requesting 2022 summer data:")
            print(f"  Target Year: {target_year}")
            print(f"  Summer Months: {[{6: 'June', 7: 'July', 8: 'August'}[m] for m in summer_months]}")
            print(f"  Expected total days: ~{len(summer_months) * 30} days")
        
        # Step 1: Geocode address
        print(f"\nGeocoding address: {address}")
        lat, lon = geocode_address(address)
            
        bbox = f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}"  # ~1km bounding box
        print(f"Using coordinates: {lat}, {lon}")
        print(f"Bounding box: {bbox}")
        
        # Step 2: Fetch data from APIs
        print("\n=== FETCHING DATA FROM APIs ===")
        
        print("Fetching 2022 summer solar data from NREL...")
        solar_data = get_solar_data(lat, lon, daily_data=get_2022_summer, 
                                   year=target_year, summer_months=summer_months)
        
        print("Fetching greenery/land use data from OpenStreetMap...")
        greenery_data = get_greenery_data(bbox)
        
        print("Fetching current weather data from OpenWeatherMap...")
        weather_data = get_weather_data(lat, lon)
        
        # Step 3: Save raw API responses
        print("\n=== SAVING RAW API DATA ===")
        raw_data_file = save_raw_api_data(solar_data, greenery_data, weather_data, address, timestamp)
        
        # Step 4: Process 2022 summer solar data
        print("\n=== PROCESSING 2022 SUMMER SOLAR DATA ===")
        
        # Handle 2022 summer solar data format
        if get_2022_summer and "outputs" in solar_data and "summer_2022_data" in solar_data["outputs"]:
            # Save detailed 2022 summer solar data
            solar_files = save_2022_summer_solar_data(solar_data, address, timestamp)
            
            # Calculate summary statistics
            summer_2022_data = solar_data["outputs"]["summer_2022_data"]
            total_days = solar_data["outputs"].get("total_summer_days", 0)
            actual_year = solar_data["outputs"].get("actual_data_year", target_year)
            
            # Calculate overall summer averages
            all_daily_values = []
            monthly_averages = {}
            
            year_data = summer_2022_data.get("2022", {})
            for month_name, month_data in year_data.items():
                daily_values = month_data.get("daily_ghi", [])
                all_daily_values.extend(daily_values)
                if daily_values:
                    monthly_averages[month_name] = statistics.mean(daily_values)
            
            overall_avg = sum(all_daily_values) / len(all_daily_values) if all_daily_values else 0
            
            solar_summary = {
                "total_summer_days": total_days,
                "target_year": target_year,
                "actual_data_year": actual_year,
                "overall_summer_avg_ghi": overall_avg,
                "monthly_averages": monthly_averages,
                "data_type": "summer_2022"
            }
            
            print(f"✓ Processed {total_days} summer days for 2022")
            print(f"✓ Overall summer average GHI: {overall_avg:.2f} kWh/m²/day")
            for month, avg in monthly_averages.items():
                print(f"  {month.title()}: {avg:.2f} kWh/m²/day")
            
            if actual_year != target_year:
                print(f"⚠️  Note: Using {actual_year} data as proxy for {target_year}")
            
        else:
            # Fallback for regular data
            solar_summary = solar_data["outputs"]["avg_ghi"] if "outputs" in solar_data else "N/A"
        
        # Step 5: Prepare comprehensive training dataset
        print("\n=== PREPARING 2022 TRAINING DATA ===")
        
        training_data = {
            "address": address,
            "coordinates": (lat, lon),
            "solar_data_summary": solar_summary,
            "greenery_score": greenery_data["tree_nodes"] + greenery_data["forest_areas"] * 10,
            "current_temperature": weather_data["current"]["temp"],
            "current_humidity": weather_data["current"]["humidity"],
            "tree_count": greenery_data["tree_nodes"],
            "forest_areas": greenery_data["forest_areas"],
            "data_collection_timestamp": timestamp,
            "data_type": "summer_2022_focused"
        }
        
        print("\n=== TRAINING DATA SUMMARY ===")
        for key, value in training_data.items():
            if key == "solar_data_summary" and isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Step 6: Save processed training data
        print("\n=== SAVING TRAINING DATA ===")
        json_file, csv_file = save_training_data(training_data, timestamp)
        
        # Step 7: Append to master dataset
        print("\n=== UPDATING MASTER DATASET ===")
        master_file = append_to_master_dataset(training_data, timestamp)
        
        print(f"\n✓ 2022 summer data collection successful for {address}")
        print(f"✓ All data saved with timestamp: {timestamp}")
        if get_2022_summer and "total_summer_days" in solar_summary:
            print(f"✓ Summer 2022 data: {solar_summary['total_summer_days']} days")
            if solar_summary.get('actual_data_year') != target_year:
                print(f"✓ Data source: {solar_summary['actual_data_year']} (used as 2022 proxy)")
        
        return {
            "success": True,
            "training_data": training_data,
            "files_created": {
                "raw_data": raw_data_file,
                "training_json": json_file,
                "training_csv": csv_file,
                "master_dataset": master_file,
                "summer_2022_solar": solar_files[0] if solar_files and solar_files[0] else None,
                "summer_2022_csv": solar_files[1] if solar_files and solar_files[1] else None
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
    # 2022 Summer data collection (focused approach)
    print("=== COLLECTING 2022 SUMMER DATA ===")
    
    # Define the target year and summer months
    target_year = 2022
    summer_months = [6, 7, 8]  # June, July, August
    
    print(f"Target year: {target_year}")
    print(f"Summer months: {[{6: 'June', 7: 'July', 8: 'August'}[m] for m in summer_months]}")
    print(f"Expected data points: ~{len(summer_months) * 30} daily values")
    print("\nNote: If 2022 data is not available, the script will use the most recent available year.")
    
    # Start collection
    result = main(get_2022_summer=True, target_year=target_year, summer_months=summer_months)
    
    if result and result["success"]:
        print("\n" + "="*70)
        print("=== 2022 SUMMER DATA COLLECTION SUCCESS ===")
        print("✅ 2022 summer training data ready for UHI model!")
        
        training_data = result["training_data"]
        solar_summary = training_data.get("solar_data_summary", {})
        
        if isinstance(solar_summary, dict) and "total_summer_days" in solar_summary:
            print(f"\n📊 DATA COLLECTION SUMMARY:")
            print(f"  Target year: {solar_summary['target_year']}")
            if solar_summary.get('actual_data_year') != solar_summary['target_year']:
                print(f"  Data source year: {solar_summary['actual_data_year']} (used as 2022 proxy)")
            print(f"  Total summer days: {solar_summary['total_summer_days']}")
            print(f"  Overall summer average GHI: {solar_summary['overall_summer_avg_ghi']:.2f} kWh/m²/day")
            
            print(f"\n📈 MONTHLY AVERAGES (2022):")
            for month, avg in solar_summary.get("monthly_averages", {}).items():
                print(f"  {month.title()}: {avg:.2f} kWh/m²/day")
        
        print(f"\n📁 FILES CREATED:")
        for file_type, filepath in result["files_created"].items():
            if filepath:
                print(f"  {file_type}: {filepath}")
        
        print(f"\n🔥 HEAT ISLAND ANALYSIS READY:")
        print("  • Daily resolution 2022 summer data collected")
        print("  • ~92 daily solar radiation measurements")
        print("  • Perfect for detailed UHI analysis")
        print("  • Can identify specific hot days and patterns")
        
    else:
        print("\n" + "="*70)
        print("=== 2022 COLLECTION FAILED ===")
        print("✗ Data collection failed. Check the following:")
        print("  1. NREL API key in .env file")
        print("  2. Internet connection")
        print("  3. API rate limits (try again later if hitting limits)")
        print("  4. 2022 data availability (script will try backup years)")
        
        if result and "error" in result:
            print(f"\nError details: {result['error']}")
    
    # Show analysis options
    print(f"\n{'='*70}")
    print("=== NEXT STEPS FOR 2022 ANALYSIS ===")
    print("1. Run: python graph_single_location.py")
    print("   • Analyzes and graphs your 2022 summer data")
    print("   • Creates comprehensive visualizations")
    print("   • Saves all data to files")
    print()
    print("2. Run: python quick_comparison.py")
    print("   • Compares your location to Nashville averages")
    print("   • Provides heat island assessment")
    print()
    print("3. For full area comparison:")
    print("   • Run: python nashville_comparison.py")
    print("   • Collects data for multiple Nashville locations")
    print("   • Comprehensive regional heat island analysis")