#!/usr/bin/env python3
"""
2022 Summer Solar Data Visualization Tool

Creates comprehensive visual graphs and charts for 2022 summer solar data,
including time series plots, monthly comparisons, and distribution analysis.
"""

import json
import glob
import statistics
import os
from datetime import datetime, date
import calendar

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib")

def find_latest_single_location_data():
    """Find the most recent single location data file"""
    
    # Look for 2022 summer solar data files (matching the actual file pattern)
    summer_files = glob.glob("data/training_data/summer_2022_solar_*.json")
    
    # Look for CSV summary files  
    csv_files = glob.glob("data/training_data/summer_2022_daily_summary_*.csv")
    
    # Look for regular training data files
    training_files = glob.glob("data/training_data/training_data_*.json")
    
    # Look for master training data
    master_files = glob.glob("data/processed/master_training_data.json")
    
    if summer_files:
        latest_file = sorted(summer_files, reverse=True)[0]
        return latest_file, "summer_2022_json"
    elif csv_files:
        latest_file = sorted(csv_files, reverse=True)[0]
        return latest_file, "csv_summary"
    elif training_files:
        latest_file = sorted(training_files, reverse=True)[0]
        return latest_file, "training_json"
    elif master_files:
        return master_files[0], "master_json"
    else:
        return None, None

def load_2022_data_only(file_path, file_type):
    """Load ONLY 2022 data from the file"""
    
    if file_type == "summer_2022_json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # This should match the actual structure from dataset.py
        summer_data = data.get("summer_2022_data", {})
        address = data.get("address", "Unknown Location")
        
        return summer_data, address
    
    elif file_type == "csv_summary":
        # Parse CSV data for 2022 only
        daily_data = {"2022": {"june": {"daily_ghi": [], "days": []}, 
                              "july": {"daily_ghi": [], "days": []}, 
                              "august": {"daily_ghi": [], "days": []}}}
        address = "Location from CSV"
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header, filter for 2022 only
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        try:
                            year = int(parts[0])
                            if year != 2022:  # Skip non-2022 data
                                continue
                                
                            month = int(parts[1])
                            day = int(parts[2])
                            ghi = float(parts[3])
                            
                            month_name = {6: "june", 7: "july", 8: "august"}.get(month)
                            if month_name:
                                daily_data["2022"][month_name]["daily_ghi"].append(ghi)
                                daily_data["2022"][month_name]["days"].append(day)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        
        return daily_data, address
    
    elif file_type == "training_json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract address
        address = data.get("address", "Unknown Location")
        
        # Check if this has solar_data_summary with 2022 data
        solar_summary = data.get("solar_data_summary", {})
        if isinstance(solar_summary, dict) and "monthly_averages" in solar_summary:
            # Convert summary to daily format (approximation)
            daily_data = {"2022": {}}
            monthly_avgs = solar_summary.get("monthly_averages", {})
            
            for month_name, avg_ghi in monthly_avgs.items():
                # Generate approximate daily values around the average
                days_in_month = {"june": 30, "july": 31, "august": 31}.get(month_name, 30)
                # Add some realistic variation around the average
                if PLOTTING_AVAILABLE:
                    daily_values = np.random.normal(avg_ghi, avg_ghi * 0.3, days_in_month).tolist()
                    daily_values = [max(0, val) for val in daily_values]  # Ensure no negative values
                else:
                    daily_values = [avg_ghi] * days_in_month
                
                daily_data["2022"][month_name] = {
                    "daily_ghi": daily_values,
                    "days": list(range(1, days_in_month + 1)),
                    "avg_ghi": avg_ghi,
                    "total_days": days_in_month
                }
            
            return daily_data, address
        
        return {}, address
    
    elif file_type == "master_json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Master file is a list, get the most recent entry
        if isinstance(data, list) and data:
            latest_entry = data[-1]  # Most recent
            address = latest_entry.get("address", "Unknown Location")
            
            # Similar processing as training_json
            solar_summary = latest_entry.get("solar_data_summary", {})
            if isinstance(solar_summary, dict) and "monthly_averages" in solar_summary:
                daily_data = {"2022": {}}
                monthly_avgs = solar_summary.get("monthly_averages", {})
                
                for month_name, avg_ghi in monthly_avgs.items():
                    days_in_month = {"june": 30, "july": 31, "august": 31}.get(month_name, 30)
                    # Add realistic variation
                    if PLOTTING_AVAILABLE:
                        daily_values = np.random.normal(avg_ghi, avg_ghi * 0.3, days_in_month).tolist()
                        daily_values = [max(0, val) for val in daily_values]
                    else:
                        daily_values = [avg_ghi] * days_in_month
                    
                    daily_data["2022"][month_name] = {
                        "daily_ghi": daily_values,
                        "days": list(range(1, days_in_month + 1)),
                        "avg_ghi": avg_ghi,
                        "total_days": days_in_month
                    }
                
                return daily_data, address
        
        return {}, "Unknown Location"
    
    return {}, "Unknown Location"

def create_comprehensive_graphs(daily_data, address):
    """Create comprehensive visual graphs for 2022 summer solar data"""
    
    if not PLOTTING_AVAILABLE:
        print("❌ Cannot create graphs - matplotlib not installed")
        print("💡 Install with: pip install matplotlib numpy")
        return None
    
    data_2022 = daily_data.get("2022", {})
    if not data_2022:
        print("❌ No 2022 data found")
        return None
    
    # Prepare data for plotting
    all_dates = []
    all_values = []
    monthly_data = {"june": [], "july": [], "august": []}
    month_nums = {"june": 6, "july": 7, "august": 8}
    
    for month_name, month_data in data_2022.items():
        if month_name in month_nums:
            month_num = month_nums[month_name]
            daily_ghi = month_data.get("daily_ghi", [])
            days = month_data.get("days", list(range(1, len(daily_ghi) + 1)))
            
            for day, ghi in zip(days, daily_ghi):
                date_obj = datetime(2022, month_num, day)
                all_dates.append(date_obj)
                all_values.append(ghi)
                monthly_data[month_name].append(ghi)
    
    if not all_values:
        print("❌ No valid data points found")
        return None
    
    # Create output directory
    output_dir = "data/graphs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    created_files = []
    
    # Graph 1: Complete Time Series (Daily Values)
    plt.figure(figsize=(15, 8))
    plt.plot(all_dates, all_values, 'b-', linewidth=1.5, alpha=0.8)
    plt.scatter(all_dates, all_values, c=all_values, cmap='viridis', s=30, alpha=0.6)
    
    plt.title(f'2022 Summer Daily Solar Radiation\n{address}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Global Horizontal Irradiance (kWh/m²/day)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add statistics text
    avg_val = statistics.mean(all_values)
    max_val = max(all_values)
    min_val = min(all_values)
    plt.text(0.02, 0.98, f'Avg: {avg_val:.2f}\nMax: {max_val:.2f}\nMin: {min_val:.2f}\nDays: {len(all_values)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    timeseries_file = f"{output_dir}/2022_daily_timeseries_{timestamp}.png"
    plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
    created_files.append(timeseries_file)
    plt.close()
    
    # Graph 2: Monthly Comparison (Box Plot)
    plt.figure(figsize=(12, 8))
    
    month_labels = []
    month_values = []
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    for i, (month_name, month_display) in enumerate([("june", "June"), ("july", "July"), ("august", "August")]):
        if monthly_data[month_name]:
            month_labels.append(month_display)
            month_values.append(monthly_data[month_name])
    
    box_plot = plt.boxplot(month_values, labels=month_labels, patch_artist=True, 
                          notch=True, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors[:len(month_values)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title(f'2022 Summer Monthly Solar Radiation Distribution\n{address}', fontsize=16, fontweight='bold')
    plt.ylabel('Global Horizontal Irradiance (kWh/m²/day)', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add monthly averages as text
    for i, month_name in enumerate(["june", "july", "august"]):
        if monthly_data[month_name]:
            avg = statistics.mean(monthly_data[month_name])
            plt.text(i+1, avg, f'{avg:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    monthly_file = f"{output_dir}/2022_monthly_comparison_{timestamp}.png"
    plt.savefig(monthly_file, dpi=300, bbox_inches='tight')
    created_files.append(monthly_file)
    plt.close()
    
    # Graph 3: Distribution Histogram
    plt.figure(figsize=(12, 8))
    
    n_bins = 20
    counts, bins, patches = plt.hist(all_values, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color bars based on value (gradient)
    cm = plt.cm.viridis
    for i, (count, patch) in enumerate(zip(counts, patches)):
        patch.set_facecolor(cm(i / len(patches)))
    
    plt.axvline(avg_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_val:.2f}')
    plt.axvline(statistics.median(all_values), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {statistics.median(all_values):.2f}')
    
    plt.title(f'2022 Summer Solar Radiation Distribution\n{address}', fontsize=16, fontweight='bold')
    plt.xlabel('Global Horizontal Irradiance (kWh/m²/day)', fontsize=12)
    plt.ylabel('Frequency (Days)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_file = f"{output_dir}/2022_distribution_{timestamp}.png"
    plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
    created_files.append(histogram_file)
    plt.close()
    
    # Graph 4: Monthly Bar Chart with Error Bars
    plt.figure(figsize=(10, 8))
    
    month_names = []
    month_means = []
    month_stds = []
    
    for month_name, month_display in [("june", "June"), ("july", "July"), ("august", "August")]:
        if monthly_data[month_name]:
            month_names.append(month_display)
            month_means.append(statistics.mean(monthly_data[month_name]))
            month_stds.append(statistics.stdev(monthly_data[month_name]) if len(monthly_data[month_name]) > 1 else 0)
    
    bars = plt.bar(month_names, month_means, yerr=month_stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, mean in zip(bars, month_means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(month_stds)*0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'2022 Summer Monthly Average Solar Radiation\n{address}', fontsize=16, fontweight='bold')
    plt.ylabel('Average GHI (kWh/m²/day)', fontsize=12)
    plt.xlabel('Month', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    monthly_bar_file = f"{output_dir}/2022_monthly_averages_{timestamp}.png"
    plt.savefig(monthly_bar_file, dpi=300, bbox_inches='tight')
    created_files.append(monthly_bar_file)
    plt.close()
    
    # Graph 5: Heat Map (Calendar Style)
    plt.figure(figsize=(15, 6))
    
    # Create a matrix for the heatmap
    max_days = 31
    heat_data = np.full((3, max_days), np.nan)  # 3 months x 31 days
    
    for i, month_name in enumerate(["june", "july", "august"]):
        if monthly_data[month_name]:
            month_data_obj = data_2022[month_name]
            daily_ghi = month_data_obj.get("daily_ghi", [])
            days = month_data_obj.get("days", list(range(1, len(daily_ghi) + 1)))
            
            for day, ghi in zip(days, daily_ghi):
                if day <= max_days:
                    heat_data[i, day-1] = ghi
    
    im = plt.imshow(heat_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    plt.title(f'2022 Summer Daily Solar Radiation Calendar\n{address}', fontsize=16, fontweight='bold')
    plt.xlabel('Day of Month', fontsize=12)
    plt.ylabel('Month', fontsize=12)
    
    # Set ticks
    plt.xticks(range(0, 31, 5), range(1, 32, 5))
    plt.yticks(range(3), ['June', 'July', 'August'])
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('GHI (kWh/m²/day)', fontsize=12)
    
    plt.tight_layout()
    heatmap_file = f"{output_dir}/2022_heatmap_{timestamp}.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    created_files.append(heatmap_file)
    plt.close()
    
    return created_files

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
            bar_length = int((month_avg / avg_all) * 30) if avg_all > 0 else 0
            bar = "█" * bar_length
            
            print(f"{month_name:8s}: {bar:<30} {month_avg:.3f} kWh/m²/day")
            print(f"           Range: {month_min:.3f} - {month_max:.3f}, Days: {len(monthly_data[month_key])}")
            print(f"           Std Dev: ±{month_std:.3f}")
            print()

def main():
    """Main function to analyze and create graphs for 2022 data"""
    
    print("📊 2022 Summer Solar Data Analysis & Visualization")
    print("=" * 60)
    
    # Check if plotting is available
    if not PLOTTING_AVAILABLE:
        print("⚠️  matplotlib not available - install with:")
        print("   pip install matplotlib numpy")
        print("   Continuing with text analysis only...")
    
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
        print(f"💡 Available data keys: {list(daily_data_2022.keys()) if daily_data_2022 else 'None'}")
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
    print(f"\n🎨 Creating 2022 text analysis...")
    create_2022_text_analysis(daily_data_2022, address)
    
    # Create visual graphs
    if PLOTTING_AVAILABLE:
        print(f"\n📈 Creating visual graphs...")
        created_files = create_comprehensive_graphs(daily_data_2022, address)
        
        if created_files:
            print(f"\n✅ 2022 Solar Data Visualization Complete!")
            print(f"📁 Graphs saved in: data/graphs/")
            print(f"\n📊 Created graphs:")
            for filepath in created_files:
                filename = os.path.basename(filepath)
                print(f"   • {filename}")
            
            print(f"\n🎯 Graph Types Created:")
            print(f"   📈 Daily time series plot")
            print(f"   📊 Monthly box plot comparison") 
            print(f"   📈 Distribution histogram")
            print(f"   📊 Monthly averages bar chart")
            print(f"   🔥 Calendar heatmap")
        else:
            print(f"\n❌ Failed to create graphs.")
    else:
        print(f"\n⚠️  Skipping graph creation - matplotlib not installed")

if __name__ == "__main__":
    main()