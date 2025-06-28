#!/usr/bin/env python3
"""
Single Location Solar Data Visualization Tool - 2022 Focus

Graphs and analyzes ONLY 2022 summer solar data for the target location,
saving all results to files for detailed analysis.
"""

import json
import glob
import statistics
from datetime import datetime, date
import calendar

def find_latest_single_location_data():
    """Find the most recent single location data file"""
    
    # Look for multi-year summer solar data
    summer_files = glob.glob("data/training_data/multi_year_summer_solar_*.json")
    
    # Look for regular training data files
    training_files = glob.glob("data/training_data/training_data_*.json")
    
    # Look for CSV summary files
    csv_files = glob.glob("data/training_data/summer_daily_summary_*.csv")
    
    if summer_files:
        latest_file = sorted(summer_files, reverse=True)[0]
        return latest_file, "multi_year_json"
    elif csv_files:
        latest_file = sorted(csv_files, reverse=True)[0]
        return latest_file, "csv_summary"
    elif training_files:
        latest_file = sorted(training_files, reverse=True)[0]
        return latest_file, "training_json"
    else:
        return None, None

def load_2022_data_only(file_path, file_type):
    """Load ONLY 2022 data from the file"""
    
    if file_type == "multi_year_json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_data = data.get("summer_data", {})
        address = data.get("address", "Unknown Location")
        
        # Extract only 2022 data
        data_2022 = all_data.get("2022", {})
        return {"2022": data_2022} if data_2022 else {}, address
    
    elif file_type == "csv_summary":
        # Parse CSV data for 2022 only
        daily_data = {}
        address = "Location from CSV"
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header, filter for 2022 only
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        year = parts[0]
                        if year != "2022":  # Skip non-2022 data
                            continue
                            
                        month = int(parts[1])
                        day = int(parts[2])
                        ghi = float(parts[3])
                        
                        if year not in daily_data:
                            daily_data[year] = {"june": {"daily_ghi": [], "days": []}, 
                                               "july": {"daily_ghi": [], "days": []}, 
                                               "august": {"daily_ghi": [], "days": []}}
                        
                        month_name = {6: "june", 7: "july", 8: "august"}.get(month)
                        if month_name:
                            daily_data[year][month_name]["daily_ghi"].append(ghi)
                            daily_data[year][month_name]["days"].append(day)
                    except (ValueError, IndexError):
                        continue
        
        return daily_data, address
    
    elif file_type == "training_json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Try to extract solar data from training format
        solar_data = data.get("solar_data", {})
        address = data.get("address", "Unknown Location")
        
        # Convert to expected format if possible (this is likely already 2022 data)
        daily_data = {}
        if isinstance(solar_data, dict) and "daily_ghi_august_2022" in solar_data:
            # Old format - single month
            daily_data["2022"] = {
                "august": {
                    "daily_ghi": solar_data["daily_ghi_august_2022"],
                    "days": solar_data.get("august_days", list(range(1, len(solar_data["daily_ghi_august_2022"]) + 1)))
                }
            }
        
        return daily_data, address
    
    return {}, "Unknown Location"

def save_2022_data_to_files(daily_data, address):
    """Save all 2022 analysis data to comprehensive files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract 2022 data
    data_2022 = daily_data.get("2022", {})
    if not data_2022:
        print("❌ No 2022 data found")
        return None
    
    # Prepare all 2022 values
    all_values_2022 = []
    monthly_data = {"june": [], "july": [], "august": []}
    daily_records = []
    
    month_nums = {"june": 6, "july": 7, "august": 8}
    
    for month_name, month_data in data_2022.items():
        if month_name in month_nums:
            month_num = month_nums[month_name]
            daily_ghi = month_data.get("daily_ghi", [])
            days = month_data.get("days", list(range(1, len(daily_ghi) + 1)))
            
            for day, ghi in zip(days, daily_ghi):
                all_values_2022.append(ghi)
                monthly_data[month_name].append(ghi)
                daily_records.append({
                    "date": f"2022-{month_num:02d}-{day:02d}",
                    "month": month_name,
                    "day": day,
                    "ghi": ghi
                })
    
    if not all_values_2022:
        print("❌ No valid 2022 data points found")
        return None
    
    # Calculate comprehensive statistics
    stats_2022 = {
        "address": address,
        "analysis_timestamp": timestamp,
        "year": 2022,
        "total_days": len(all_values_2022),
        "overall_statistics": {
            "mean": statistics.mean(all_values_2022),
            "median": statistics.median(all_values_2022),
            "std_dev": statistics.stdev(all_values_2022) if len(all_values_2022) > 1 else 0,
            "minimum": min(all_values_2022),
            "maximum": max(all_values_2022),
            "range": max(all_values_2022) - min(all_values_2022)
        },
        "monthly_statistics": {},
        "extreme_values": {
            "highest_10": sorted(all_values_2022, reverse=True)[:10],
            "lowest_10": sorted(all_values_2022)[:10]
        },
        "daily_records": daily_records
    }
    
    # Calculate monthly statistics
    for month_name, values in monthly_data.items():
        if values:
            stats_2022["monthly_statistics"][month_name] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "minimum": min(values),
                "maximum": max(values),
                "range": max(values) - min(values)
            }
    
    # File 1: Complete JSON analysis file
    json_file = f"data/training_data/2022_solar_analysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(stats_2022, f, indent=2)
    
    # File 2: Daily CSV data
    csv_file = f"data/training_data/2022_daily_solar_data_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Date,Month,Day,GHI_kWh_per_m2_per_day\n")
        for record in daily_records:
            f.write(f"{record['date']},{record['month']},{record['day']},{record['ghi']:.4f}\n")
    
    # File 3: Comprehensive text report
    report_file = f"data/training_data/2022_solar_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("2022 SUMMER SOLAR RADIATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Location: {address}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Year: 2022 ONLY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        overall = stats_2022["overall_statistics"]
        f.write("OVERALL 2022 SUMMER STATISTICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Summer Days Analyzed: {stats_2022['total_days']}\n")
        f.write(f"Average Daily GHI: {overall['mean']:.3f} kWh/m²/day\n")
        f.write(f"Standard Deviation: {overall['std_dev']:.3f} kWh/m²/day\n")
        f.write(f"Median Daily GHI: {overall['median']:.3f} kWh/m²/day\n")
        f.write(f"Maximum Daily GHI: {overall['maximum']:.3f} kWh/m²/day\n")
        f.write(f"Minimum Daily GHI: {overall['minimum']:.3f} kWh/m²/day\n")
        f.write(f"Range: {overall['range']:.3f} kWh/m²/day\n\n")
        
        # Monthly breakdown
        f.write("MONTHLY BREAKDOWN (2022):\n")
        f.write("-" * 50 + "\n")
        for month_name, month_stats in stats_2022["monthly_statistics"].items():
            f.write(f"{month_name.upper()}:\n")
            f.write(f"  Days: {month_stats['count']}\n")
            f.write(f"  Average: {month_stats['mean']:.3f} kWh/m²/day\n")
            f.write(f"  Range: {month_stats['minimum']:.3f} - {month_stats['maximum']:.3f}\n")
            f.write(f"  Std Dev: {month_stats['std_dev']:.3f}\n\n")
        
        # Extreme values
        f.write("EXTREME VALUES ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        f.write("TOP 10 HIGHEST SOLAR DAYS:\n")
        for i, val in enumerate(stats_2022["extreme_values"]["highest_10"], 1):
            f.write(f"  {i:2d}. {val:.3f} kWh/m²/day\n")
        
        f.write("\nTOP 10 LOWEST SOLAR DAYS:\n")
        for i, val in enumerate(stats_2022["extreme_values"]["lowest_10"], 1):
            f.write(f"  {i:2d}. {val:.3f} kWh/m²/day\n")
        
        # All daily data
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMPLETE 2022 DAILY DATA:\n")
        f.write("=" * 80 + "\n")
        
        for month_name in ["june", "july", "august"]:
            if month_name in data_2022:
                month_data = data_2022[month_name]
                daily_ghi = month_data.get("daily_ghi", [])
                days = month_data.get("days", list(range(1, len(daily_ghi) + 1)))
                
                f.write(f"\n{month_name.upper()} 2022:\n")
                f.write("-" * 30 + "\n")
                
                for day, ghi in zip(days, daily_ghi):
                    date_str = f"2022-{month_nums[month_name]:02d}-{day:02d}"
                    f.write(f"{date_str}: {ghi:.4f} kWh/m²/day\n")
    
    # File 4: Summary statistics file  
    summary_file = f"data/training_data/2022_summary_stats_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("2022 SUMMER SOLAR - QUICK STATISTICS SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Location: {address}\n")
        f.write(f"Year: 2022 (Summer Only)\n")
        f.write(f"Days Analyzed: {stats_2022['total_days']}\n")
        f.write(f"Average GHI: {overall['mean']:.2f} kWh/m²/day\n")
        f.write(f"Std Deviation: {overall['std_dev']:.2f}\n")
        f.write(f"Range: {overall['minimum']:.2f} - {overall['maximum']:.2f}\n")
        f.write("\nMonthly Averages:\n")
        for month_name, month_stats in stats_2022["monthly_statistics"].items():
            f.write(f"  {month_name.title()}: {month_stats['mean']:.2f} kWh/m²/day\n")
    
    print(f"✅ All 2022 data saved to files:")
    print(f"   📊 Complete analysis: {json_file}")
    print(f"   📈 Daily data CSV: {csv_file}")
    print(f"   📝 Detailed report: {report_file}")
    print(f"   📋 Quick summary: {summary_file}")
    
    return {
        "json_file": json_file,
        "csv_file": csv_file,
        "report_file": report_file,
        "summary_file": summary_file,
        "statistics": stats_2022
    }

def create_2022_text_analysis(daily_data, address):
    """Create comprehensive text analysis for 2022 data"""
    
    data_2022 = daily_data.get("2022", {})
    if not data_2022:
        print("❌ No 2022 data found")
        return
    
    # Prepare 2022 data
    all_values = []
    monthly_data = {"june": [], "july": [], "august": []}
    
    for month_name, month_data in data_2022.items():
        if month_name in monthly_data:
            daily_ghi = month_data.get("daily_ghi", [])
            all_values.extend(daily_ghi)
            monthly_data[month_name].extend(daily_ghi)
    
    if not all_values:
        print("❌ No valid 2022 data found")
        return
    
    print("=" * 80)
    print(f"📊 2022 SUMMER SOLAR RADIATION ANALYSIS: {address}")
    print("=" * 80)
    
    # Overall statistics
    avg_all = statistics.mean(all_values)
    std_all = statistics.stdev(all_values) if len(all_values) > 1 else 0
    max_val = max(all_values)
    min_val = min(all_values)
    
    print(f"\n📈 2022 SUMMER OVERALL STATISTICS")
    print(f"{'=' * 50}")
    print(f"📊 Total Days Analyzed: {len(all_values)}")
    print(f"📅 Year: 2022 (Summer Only)")
    print(f"🌡️  Average GHI: {avg_all:.3f} ± {std_all:.3f} kWh/m²/day")
    print(f"📏 Range: {min_val:.3f} - {max_val:.3f} kWh/m²/day")
    print(f"📊 Median: {statistics.median(all_values):.3f} kWh/m²/day")
    
    # Monthly comparison for 2022
    print(f"\n🗓️  2022 MONTHLY COMPARISON")
    print(f"{'=' * 50}")
    
    month_names = {"june": "June", "july": "July", "august": "August"}
    
    for month_key, month_name in month_names.items():
        if monthly_data[month_key]:
            month_avg = statistics.mean(monthly_data[month_key])
            month_std = statistics.stdev(monthly_data[month_key]) if len(monthly_data[month_key]) > 1 else 0
            month_max = max(monthly_data[month_key])
            month_min = min(monthly_data[month_key])
            
            # Create visual bar relative to overall average
            bar_length = int((month_avg / avg_all) * 30)
            bar = "█" * bar_length
            
            print(f"{month_name:8s}: {bar:<30} {month_avg:.3f} kWh/m²/day")
            print(f"           Range: {month_min:.3f} - {month_max:.3f}, Days: {len(monthly_data[month_key])}")
            print(f"           Std Dev: ±{month_std:.3f}")
            print()
    
    # Extreme values analysis
    print(f"\n🔥 2022 EXTREME VALUES ANALYSIS")
    print(f"{'=' * 50}")
    
    # Find highest and lowest days
    sorted_values = sorted(all_values, reverse=True)
    high_threshold = avg_all + std_all
    low_threshold = avg_all - std_all
    
    high_days = [v for v in all_values if v > high_threshold]
    low_days = [v for v in all_values if v < low_threshold]
    
    print(f"🌡️  High Solar Days (>{high_threshold:.3f}): {len(high_days)} days ({len(high_days)/len(all_values)*100:.1f}%)")
    print(f"❄️  Low Solar Days (<{low_threshold:.3f}): {len(low_days)} days ({len(low_days)/len(all_values)*100:.1f}%)")
    print(f"🌤️  Normal Days: {len(all_values) - len(high_days) - len(low_days)} days")
    
    print(f"\n📊 TOP 10 HIGHEST DAYS (2022):")
    for i, val in enumerate(sorted_values[:10], 1):
        print(f"   {i:2d}. {val:.4f} kWh/m²/day")
    
    print(f"\n📊 TOP 10 LOWEST DAYS (2022):")
    for i, val in enumerate(sorted(all_values)[:10], 1):
        print(f"   {i:2d}. {val:.4f} kWh/m²/day")
    
    # Distribution analysis
    print(f"\n📊 2022 VALUE DISTRIBUTION")
    print(f"{'=' * 50}")
    
    # Create bins
    num_bins = 8
    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]
    
    for i in range(num_bins):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        count = sum(1 for v in all_values if bin_start <= v < bin_end)
        if i == num_bins - 1:  # Include max value in last bin
            count = sum(1 for v in all_values if bin_start <= v <= bin_end)
        
        bar_length = int(count / len(all_values) * 40) if all_values else 0
        bar = "█" * bar_length
        
        print(f"{bin_start:.2f}-{bin_end:.2f}: {bar:<40} {count:3d} ({count/len(all_values)*100:.1f}%)")

def main():
    """Main function to analyze and save 2022 data only"""
    
    print("📊 2022 Summer Solar Data Analysis & Export")
    print("=" * 50)
    
    # Find the latest data file
    data_file, file_type = find_latest_single_location_data()
    
    if not data_file:
        print("❌ No single location data found.")
        print("💡 Run 'python dataset.py' first to collect data.")
        return
    
    print(f"📄 Found data file: {data_file}")
    print(f"📋 File type: {file_type}")
    
    # Load ONLY 2022 data
    daily_data_2022, address = load_2022_data_only(data_file, file_type)
    
    if not daily_data_2022 or "2022" not in daily_data_2022:
        print("❌ No 2022 data found in file.")
        print("💡 Make sure your dataset includes 2022 summer data.")
        return
    
    print(f"📍 Location: {address}")
    
    # Count 2022 data points
    data_2022 = daily_data_2022["2022"]
    total_days = sum(
        len(month_data.get("daily_ghi", []))
        for month_data in data_2022.values()
    )
    print(f"📊 2022 summer data points: {total_days} days")
    print(f"📅 Months available: {sorted(data_2022.keys())}")
    
    # Create text analysis
    print(f"\n🎨 Creating 2022 analysis...")
    create_2022_text_analysis(daily_data_2022, address)
    
    # Save all data to files
    print(f"\n💾 Saving all 2022 data to files...")
    saved_files = save_2022_data_to_files(daily_data_2022, address)
    
    if saved_files:
        print(f"\n✅ 2022 Solar Data Analysis Complete!")
        print(f"📁 All files saved in: data/training_data/")
        print(f"\n📋 Files created:")
        for file_type, filepath in saved_files.items():
            if file_type != "statistics":
                print(f"   • {filepath}")
        
        # Show key statistics
        stats = saved_files["statistics"]
        overall = stats["overall_statistics"]
        print(f"\n📊 Key 2022 Statistics:")
        print(f"   Average: {overall['mean']:.2f} kWh/m²/day")
        print(f"   Range: {overall['minimum']:.2f} - {overall['maximum']:.2f}")
        print(f"   Total Days: {stats['total_days']}")
    else:
        print(f"\n❌ Failed to save data files.")

if __name__ == "__main__":
    main()

def create_2022_text_analysis(daily_data, address):
    """Create comprehensive text analysis for 2022 data"""
    
    data_2022 = daily_data.get("2022", {})
    if not data_2022:
        print("❌ No 2022 data found")
        return
    
    # Prepare 2022 data
    all_values = []
    monthly_data = {"june": [], "july": [], "august": []}
    
    for month_name, month_data in data_2022.items():
        if month_name in monthly_data:
            daily_ghi = month_data.get("daily_ghi", [])
            all_values.extend(daily_ghi)
            monthly_data[month_name].extend(daily_ghi)
    
    if not all_values:
        print("❌ No valid 2022 data found")
        return
    
    print("=" * 80)
    print(f"📊 2022 SUMMER SOLAR RADIATION ANALYSIS: {address}")
    print("=" * 80)
    
    # Overall statistics
    avg_all = statistics.mean(all_values)
    std_all = statistics.stdev(all_values) if len(all_values) > 1 else 0
    max_val = max(all_values)
    min_val = min(all_values)
    
    print(f"\n📈 2022 SUMMER OVERALL STATISTICS")
    print(f"{'=' * 50}")
    print(f"📊 Total Days Analyzed: {len(all_values)}")
    print(f"📅 Year: 2022 (Summer Only)")
    print(f"🌡️  Average GHI: {avg_all:.3f} ± {std_all:.3f} kWh/m²/day")
    print(f"📏 Range: {min_val:.3f} - {max_val:.3f} kWh/m²/day")
    print(f"📊 Median: {statistics.median(all_values):.3f} kWh/m²/day")
    
    # Monthly comparison for 2022
    print(f"\n🗓️  2022 MONTHLY COMPARISON")
    print(f"{'=' * 50}")
    
    month_names = {"june": "June", "july": "July", "august": "August"}
    
    for month_key, month_name in month_names.items():
        if monthly_data[month_key]:
            month_avg = statistics.mean(monthly_data[month_key])
            month_std = statistics.stdev(monthly_data[month_key]) if len(monthly_data[month_key]) > 1 else 0
            month_max = max(monthly_data[month_key])
            month_min = min(monthly_data[month_key])
            
            # Create visual bar relative to overall average
            bar_length = int((month_avg / avg_all) * 30)
            bar = "█" * bar_length
            
            print(f"{month_name:8s}: {bar:<30} {month_avg:.3f} kWh/m²/day")
            print(f"           Range: {month_min:.3f} - {month_max:.3f}, Days: {len(monthly_data[month_key])}")
            print(f"           Std Dev: ±{month_std:.3f}")
            print()
    
    # Extreme values analysis
    print(f"\n🔥 2022 EXTREME VALUES ANALYSIS")
    print(f"{'=' * 50}")
    
    # Find highest and lowest days
    sorted_values = sorted(all_values, reverse=True)
    high_threshold = avg_all + std_all
    low_threshold = avg_all - std_all
    
    high_days = [v for v in all_values if v > high_threshold]
    low_days = [v for v in all_values if v < low_threshold]
    
    print(f"🌡️  High Solar Days (>{high_threshold:.3f}): {len(high_days)} days ({len(high_days)/len(all_values)*100:.1f}%)")
    print(f"❄️  Low Solar Days (<{low_threshold:.3f}): {len(low_days)} days ({len(low_days)/len(all_values)*100:.1f}%)")
    print(f"🌤️  Normal Days: {len(all_values) - len(high_days) - len(low_days)} days")
    
    print(f"\n📊 TOP 10 HIGHEST DAYS (2022):")
    for i, val in enumerate(sorted_values[:10], 1):
        print(f"   {i:2d}. {val:.4f} kWh/m²/day")
    
    print(f"\n📊 TOP 10 LOWEST DAYS (2022):")
    for i, val in enumerate(sorted(all_values)[:10], 1):
        print(f"   {i:2d}. {val:.4f} kWh/m²/day")
    
    # Distribution analysis
    print(f"\n📊 2022 VALUE DISTRIBUTION")
    print(f"{'=' * 50}")
    
    # Create bins
    num_bins = 8
    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]
    
    for i in range(num_bins):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        count = sum(1 for v in all_values if bin_start <= v < bin_end)
        if i == num_bins - 1:  # Include max value in last bin
            count = sum(1 for v in all_values if bin_start <= v <= bin_end)
        
        bar_length = int(count / len(all_values) * 40) if all_values else 0
        bar = "█" * bar_length
        
        print(f"{bin_start:.2f}-{bin_end:.2f}: {bar:<40} {count:3d} ({count/len(all_values)*100:.1f}%)")

def main():
    """Main function to analyze and save 2022 data only"""
    
    print("📊 2022 Summer Solar Data Analysis & Export")
    print("=" * 50)
    
    # Find the latest data file
    data_file, file_type = find_latest_single_location_data()
    
    if not data_file:
        print("❌ No single location data found.")
        print("💡 Run 'python dataset.py' first to collect data.")
        return
    
    print(f"📄 Found data file: {data_file}")
    print(f"📋 File type: {file_type}")
    
    # Load ONLY 2022 data
    daily_data_2022, address = load_2022_data_only(data_file, file_type)
    
    if not daily_data_2022 or "2022" not in daily_data_2022:
        print("❌ No 2022 data found in file.")
        print("💡 Make sure your dataset includes 2022 summer data.")
        return
    
    print(f"📍 Location: {address}")
    
    # Count 2022 data points
    data_2022 = daily_data_2022["2022"]
    total_days = sum(
        len(month_data.get("daily_ghi", []))
        for month_data in data_2022.values()
    )
    print(f"📊 2022 summer data points: {total_days} days")
    print(f"📅 Months available: {sorted(data_2022.keys())}")
    
    # Create text analysis
    print(f"\n🎨 Creating 2022 analysis...")
    create_2022_text_analysis(daily_data_2022, address)
    
    # Save all data to files
    print(f"\n💾 Saving all 2022 data to files...")
    saved_files = save_2022_data_to_files(daily_data_2022, address)
    
    if saved_files:
        print(f"\n✅ 2022 Solar Data Analysis Complete!")
        print(f"📁 All files saved in: data/training_data/")
        print(f"\n📋 Files created:")
        for file_type, filepath in saved_files.items():
            if file_type != "statistics":
                print(f"   • {filepath}")
        
        # Show key statistics
        stats = saved_files["statistics"]
        overall = stats["overall_statistics"]
        print(f"\n📊 Key 2022 Statistics:")
        print(f"   Average: {overall['mean']:.2f} kWh/m²/day")
        print(f"   Range: {overall['minimum']:.2f} - {overall['maximum']:.2f}")
        print(f"   Total Days: {stats['total_days']}")
    else:
        print(f"\n❌ Failed to save data files.")

if __name__ == "__main__":
    main()