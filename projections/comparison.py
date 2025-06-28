#!/usr/bin/env python3
"""
Nashville Area Comparison Script for Urban Heat Island Analysis

This script collects data for multiple Nashville locations and compares them
to identify potential urban heat island effects.
"""

import os
import json
import time
import statistics
from datetime import datetime
from dataset import (
    geocode_address, get_solar_data, get_greenery_data, get_weather_data,
    create_data_directories, save_raw_api_data, process_summer_csv_data
)

# Define Nashville area representative locations
NASHVILLE_LOCATIONS = {
    "urban_core": [
        "Downtown Nashville TN",
        "Music Row Nashville TN",
        "The Gulch Nashville TN",
        "Broadway Nashville TN"
    ],
    "suburban": [
        "Belle Meade Nashville TN",
        "Green Hills Nashville TN", 
        "Brentwood TN",
        "Franklin TN"
    ],
    "mixed_urban": [
        "Vanderbilt University Nashville TN",
        "Centennial Park Nashville TN",
        "Opryland Nashville TN",
        "Nashville International Airport TN"
    ],
    "target_area": [
        "1090 Murfreesboro Pike Nashville",  # Our target location
        "Murfreesboro Pike Nashville TN",
        "Antioch Nashville TN",
        "Hickory Hollow Nashville TN"
    ]
}

def collect_nashville_area_data(years=[2022, 2023, 2024, 2025], summer_months=[6, 7, 8]):
    """Collect data for multiple Nashville area locations"""
    
    print("=" * 80)
    print("🏙️  NASHVILLE AREA HEAT ISLAND COMPARISON DATA COLLECTION")
    print("=" * 80)
    
    # Create data directories
    create_data_directories()
    
    # Create comparison-specific directory
    comp_dir = "data/nashville_comparison"
    if not os.path.exists(comp_dir):
        os.makedirs(comp_dir)
        print(f"Created directory: {comp_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_location_data = {}
    successful_locations = []
    failed_locations = []
    
    # Flatten all locations into one list with categories
    all_locations = []
    for category, locations in NASHVILLE_LOCATIONS.items():
        for location in locations:
            all_locations.append((location, category))
    
    print(f"\n📍 Collecting data for {len(all_locations)} Nashville area locations")
    print(f"🗓️  Target years: {years}")
    print(f"🌞 Summer months: {[{6: 'June', 7: 'July', 8: 'August'}[m] for m in summer_months]}")
    print(f"⏱️  Estimated time: {len(all_locations) * 2} minutes (due to API rate limits)")
    
    for i, (location, category) in enumerate(all_locations, 1):
        print(f"\n{'-' * 60}")
        print(f"📍 Processing {i}/{len(all_locations)}: {location}")
        print(f"   Category: {category}")
        
        try:
            # Geocode location
            print(f"   🗺️  Geocoding...")
            lat, lon = geocode_address(location)
            bbox = f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}"
            
            # Collect solar data
            print(f"   ☀️  Fetching solar data...")
            solar_data = get_solar_data(lat, lon, daily_data=True, years=years, summer_months=summer_months)
            
            # Collect greenery data  
            print(f"   🌳 Fetching greenery data...")
            greenery_data = get_greenery_data(bbox)
            
            # Collect weather data
            print(f"   🌡️  Fetching weather data...")
            weather_data = get_weather_data(lat, lon)
            
            # Process and store data
            location_data = process_location_data(
                location, category, lat, lon, solar_data, greenery_data, weather_data, timestamp
            )
            
            if location_data:
                all_location_data[location] = location_data
                successful_locations.append(location)
                print(f"   ✅ Success: {location}")
            else:
                failed_locations.append(location)
                print(f"   ❌ Failed: {location}")
            
            # Save raw data
            save_raw_api_data(solar_data, greenery_data, weather_data, 
                            f"{category}_{location.replace(' ', '_')}", timestamp)
            
            # Rate limiting delay
            if i < len(all_locations):
                print(f"   ⏳ Waiting 5 seconds...")
                time.sleep(5)
                
        except Exception as e:
            print(f"   ❌ Error processing {location}: {e}")
            failed_locations.append(location)
            continue
    
    # Save collected data
    comparison_data = {
        "collection_timestamp": timestamp,
        "target_years": years,
        "summer_months": summer_months,
        "successful_locations": successful_locations,
        "failed_locations": failed_locations,
        "location_data": all_location_data,
        "location_categories": NASHVILLE_LOCATIONS
    }
    
    # Save comprehensive comparison data
    comp_file = f"{comp_dir}/nashville_comparison_{timestamp}.json"
    with open(comp_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print(f"📊 DATA COLLECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successful locations: {len(successful_locations)}")
    print(f"❌ Failed locations: {len(failed_locations)}")
    print(f"💾 Data saved to: {comp_file}")
    
    if failed_locations:
        print(f"\n⚠️  Failed locations:")
        for loc in failed_locations:
            print(f"   • {loc}")
    
    return comparison_data

def process_location_data(location, category, lat, lon, solar_data, greenery_data, weather_data, timestamp):
    """Process individual location data into standardized format"""
    
    try:
        # Process solar data
        if "outputs" in solar_data and "multi_year_summer_data" in solar_data["outputs"]:
            multi_year_data = solar_data["outputs"]["multi_year_summer_data"]
            
            # Calculate overall statistics
            all_daily_values = []
            yearly_averages = {}
            monthly_averages = {"june": [], "july": [], "august": []}
            
            for year, year_data in multi_year_data.items():
                year_values = []
                for month_name, month_data in year_data.items():
                    daily_values = month_data.get("daily_ghi", [])
                    year_values.extend(daily_values)
                    all_daily_values.extend(daily_values)
                    monthly_averages[month_name].extend(daily_values)
                
                if year_values:
                    yearly_averages[year] = statistics.mean(year_values)
            
            # Calculate monthly averages
            monthly_stats = {}
            for month, values in monthly_averages.items():
                if values:
                    monthly_stats[month] = {
                        "average": statistics.mean(values),
                        "median": statistics.median(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                        "max": max(values),
                        "min": min(values),
                        "total_days": len(values)
                    }
            
            # Overall statistics
            if all_daily_values:
                solar_stats = {
                    "overall_average": statistics.mean(all_daily_values),
                    "overall_median": statistics.median(all_daily_values),
                    "overall_std_dev": statistics.stdev(all_daily_values) if len(all_daily_values) > 1 else 0,
                    "overall_max": max(all_daily_values),
                    "overall_min": min(all_daily_values),
                    "total_summer_days": len(all_daily_values),
                    "yearly_averages": yearly_averages,
                    "monthly_stats": monthly_stats,
                    "successful_years": solar_data["outputs"].get("successful_years", [])
                }
            else:
                solar_stats = None
        else:
            # Fallback for monthly data
            solar_stats = {
                "fallback_data": True,
                "monthly_average": solar_data.get("outputs", {}).get("avg_ghi", 0)
            }
        
        # Process greenery data
        greenery_stats = {
            "tree_count": greenery_data.get("tree_nodes", 0),
            "forest_areas": greenery_data.get("forest_areas", 0),
            "greenery_score": greenery_data.get("tree_nodes", 0) + greenery_data.get("forest_areas", 0) * 10
        }
        
        # Process weather data
        weather_stats = {
            "current_temperature": weather_data.get("current", {}).get("temp", 0),
            "current_humidity": weather_data.get("current", {}).get("humidity", 0)
        }
        
        return {
            "location": location,
            "category": category,
            "coordinates": (lat, lon),
            "solar_statistics": solar_stats,
            "greenery_statistics": greenery_stats,
            "weather_statistics": weather_stats,
            "collection_timestamp": timestamp
        }
        
    except Exception as e:
        print(f"     Error processing data for {location}: {e}")
        return None

def analyze_nashville_comparison(comparison_file=None):
    """Analyze the collected Nashville comparison data"""
    
    if comparison_file is None:
        # Find the most recent comparison file
        import glob
        comp_files = glob.glob("data/nashville_comparison/nashville_comparison_*.json")
        if not comp_files:
            print("❌ No comparison data found. Run collect_nashville_area_data() first.")
            return
        comparison_file = sorted(comp_files, reverse=True)[0]
    
    print("=" * 80)
    print(f"📊 NASHVILLE AREA HEAT ISLAND ANALYSIS")
    print("=" * 80)
    print(f"📄 Data file: {comparison_file}")
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)
    
    location_data = data["location_data"]
    target_location = "1090 Murfreesboro Pike Nashville"
    
    # Organize data by category
    category_stats = {}
    all_locations_stats = []
    
    for location, loc_data in location_data.items():
        category = loc_data["category"]
        solar_stats = loc_data["solar_statistics"]
        
        if solar_stats and not solar_stats.get("fallback_data", False):
            overall_avg = solar_stats["overall_average"]
            greenery_score = loc_data["greenery_statistics"]["greenery_score"]
            
            location_summary = {
                "location": location,
                "category": category,
                "solar_avg": overall_avg,
                "greenery_score": greenery_score,
                "coordinates": loc_data["coordinates"]
            }
            
            all_locations_stats.append(location_summary)
            
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(overall_avg)
    
    # Calculate category averages
    print(f"\n🏙️  NASHVILLE AREA CATEGORY ANALYSIS")
    print(f"{'=' * 60}")
    
    category_averages = {}
    for category, values in category_stats.items():
        if values:
            avg = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            category_averages[category] = {
                "average": avg,
                "std_dev": std_dev,
                "count": len(values),
                "max": max(values),
                "min": min(values)
            }
            
            print(f"\n📍 {category.replace('_', ' ').title()}:")
            print(f"   Average GHI: {avg:.2f} ± {std_dev:.2f} kWh/m²/day")
            print(f"   Range: {min(values):.2f} - {max(values):.2f} kWh/m²/day")
            print(f"   Locations: {len(values)}")
    
    # Calculate Nashville area overall average
    all_solar_values = [loc["solar_avg"] for loc in all_locations_stats]
    if all_solar_values:
        nashville_avg = statistics.mean(all_solar_values)
        nashville_std = statistics.stdev(all_solar_values) if len(all_solar_values) > 1 else 0
        
        print(f"\n🌍 NASHVILLE AREA OVERALL:")
        print(f"   Average GHI: {nashville_avg:.2f} ± {nashville_std:.2f} kWh/m²/day")
        print(f"   Total locations: {len(all_solar_values)}")
    
    # Target location analysis
    if target_location in location_data:
        target_data = location_data[target_location]
        target_solar = target_data["solar_statistics"]
        
        if target_solar and not target_solar.get("fallback_data", False):
            target_avg = target_solar["overall_average"]
            target_greenery = target_data["greenery_statistics"]["greenery_score"]
            
            print(f"\n🎯 TARGET LOCATION ANALYSIS: {target_location}")
            print(f"{'=' * 60}")
            print(f"🌡️  Solar GHI: {target_avg:.2f} kWh/m²/day")
            print(f"🌳 Greenery Score: {target_greenery}")
            
            # Compare to Nashville average
            if all_solar_values:
                diff_from_avg = target_avg - nashville_avg
                pct_diff = (diff_from_avg / nashville_avg) * 100
                
                print(f"\n📊 COMPARISON TO NASHVILLE AVERAGE:")
                print(f"   Target: {target_avg:.2f} kWh/m²/day")
                print(f"   Nashville avg: {nashville_avg:.2f} kWh/m²/day")
                print(f"   Difference: {diff_from_avg:+.2f} kWh/m²/day ({pct_diff:+.1f}%)")
                
                # Heat island assessment
                print(f"\n🔥 HEAT ISLAND ASSESSMENT:")
                if abs(pct_diff) < 2:
                    assessment = "TYPICAL - Similar to Nashville average"
                    icon = "🟢"
                elif pct_diff > 0:
                    if pct_diff > 10:
                        assessment = "STRONG HEAT ISLAND - Significantly warmer than average"
                        icon = "🔴"
                    elif pct_diff > 5:
                        assessment = "MODERATE HEAT ISLAND - Warmer than average"
                        icon = "🟠"
                    else:
                        assessment = "MILD HEAT ISLAND - Slightly warmer than average"
                        icon = "🟡"
                else:
                    assessment = "COOL SPOT - Cooler than Nashville average"
                    icon = "🔵"
                
                print(f"   {icon} {assessment}")
                
                # Category comparison
                target_category = target_data["category"]
                if target_category in category_averages:
                    cat_avg = category_averages[target_category]["average"]
                    cat_diff = target_avg - cat_avg
                    cat_pct_diff = (cat_diff / cat_avg) * 100
                    
                    print(f"\n📍 COMPARISON TO {target_category.replace('_', ' ').title()} AVERAGE:")
                    print(f"   Target: {target_avg:.2f} kWh/m²/day")
                    print(f"   Category avg: {cat_avg:.2f} kWh/m²/day")
                    print(f"   Difference: {cat_diff:+.2f} kWh/m²/day ({cat_pct_diff:+.1f}%)")
    
    # Generate comparison summary
    generate_comparison_summary(data, category_averages, all_locations_stats, target_location)
    
    return {
        "category_averages": category_averages,
        "nashville_overall": nashville_avg if all_solar_values else None,
        "target_analysis": target_data if target_location in location_data else None,
        "all_locations": all_locations_stats
    }

def generate_comparison_summary(data, category_averages, all_locations_stats, target_location):
    """Generate a detailed comparison summary report"""
    
    timestamp = data["collection_timestamp"]
    summary_file = f"data/nashville_comparison/comparison_summary_{timestamp}.json"
    csv_file = f"data/nashville_comparison/location_comparison_{timestamp}.csv"
    
    # Create summary report
    summary = {
        "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "data_collection_timestamp": timestamp,
        "nashville_category_averages": category_averages,
        "target_location": target_location,
        "all_locations_data": all_locations_stats,
        "collection_info": {
            "successful_locations": len(data["successful_locations"]),
            "failed_locations": len(data["failed_locations"]),
            "target_years": data["target_years"],
            "summer_months": data["summer_months"]
        }
    }
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create CSV for easy analysis
    with open(csv_file, 'w') as f:
        f.write("Location,Category,Latitude,Longitude,Average_Summer_GHI,Greenery_Score\n")
        for loc_data in all_locations_stats:
            f.write(f"{loc_data['location']},{loc_data['category']},{loc_data['coordinates'][0]},{loc_data['coordinates'][1]},{loc_data['solar_avg']:.3f},{loc_data['greenery_score']}\n")
    
    print(f"\n💾 COMPARISON SUMMARY SAVED:")
    print(f"   📊 Summary: {summary_file}")
    print(f"   📈 CSV data: {csv_file}")

def main():
    """Main function to run Nashville area comparison"""
    
    print("🏙️  Nashville Area Heat Island Comparison Tool")
    print("=" * 50)
    print("\nThis tool will:")
    print("1. Collect multi-year summer solar data for multiple Nashville locations")
    print("2. Calculate area averages by location type (urban, suburban, etc.)")
    print("3. Compare your target location to Nashville area averages")
    print("4. Assess potential urban heat island effects")
    
    response = input("\nProceed with data collection? (y/n): ").lower().strip()
    if response != 'y':
        print("Data collection cancelled.")
        return
    
    # Collect Nashville area data
    print("\nStarting Nashville area data collection...")
    comparison_data = collect_nashville_area_data()
    
    # Analyze the collected data
    print("\nAnalyzing collected data...")
    analysis_results = analyze_nashville_comparison()
    
    print(f"\n{'=' * 80}")
    print("🎉 NASHVILLE COMPARISON COMPLETE!")
    print(f"{'=' * 80}")
    print("✅ Multi-location data collected and analyzed")
    print("✅ Heat island assessment completed")
    print("✅ Comparison summaries generated")
    print("\n🔍 Check the 'data/nashville_comparison/' folder for detailed results!")

if __name__ == "__main__":
    main()