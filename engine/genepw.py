import math
import datetime

def generate_nashville_epw():
    """
    Generate an EPW file for Nashville, TN based on the environmental data
    provided in the TUHI calculation documents.
    """
    
    # Location data for Nashville, TN
    city = "Nashville"
    state_province = "TN"
    country = "USA"
    source = "Custom_TUHI_Data"
    wmo_number = "723270"  # Nashville International Airport WMO ID
    latitude = 36.12  # degrees
    longitude = -86.68  # degrees (negative for west)
    time_zone = -6.0  # CST (UTC-6)
    elevation = 183.0  # meters above sea level
    
    # Environmental data from the TUHI document (summer conditions)
    base_temp_c = 32.0  # Air temperature from document (305K = 32°C)
    base_wind_speed = 3.0  # m/s from document
    base_solar_radiation = 850.0  # W/m² from document
    base_relative_humidity = 65.0  # typical summer value for Nashville
    base_pressure = 101325.0  # Pa (standard atmospheric pressure)
    
    # Create EPW file content
    epw_content = []
    
    # Header line 1: Location
    header1 = f"LOCATION,{city},{state_province},{country},{source},{wmo_number},{latitude:.2f},{longitude:.2f},{time_zone:.1f},{elevation:.1f}"
    epw_content.append(header1)
    
    # Header line 2: Design Conditions (simplified)
    header2 = "DESIGN CONDITIONS,0"
    epw_content.append(header2)
    
    # Header line 3: Typical/Extreme Periods (simplified)
    header3 = "TYPICAL/EXTREME PERIODS,0"
    epw_content.append(header3)
    
    # Header line 4: Ground Temperatures (simplified)
    header4 = "GROUND TEMPERATURES,0"
    epw_content.append(header4)
    
    # Header line 5: Holidays/Daylight Savings (simplified)
    header5 = "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0"
    epw_content.append(header5)
    
    # Header line 6: Comments (simplified)
    header6 = "COMMENTS 1,Custom EPW file for Nashville TUHI analysis - Thompson Residential Area"
    epw_content.append(header6)
    
    # Header line 7: Comments (simplified)
    header7 = "COMMENTS 2,Based on summer environmental conditions from TUHI documentation"
    epw_content.append(header7)
    
    # Header line 8: Data Periods
    header8 = "DATA PERIODS,1,1,Data,Sunday,1/1,12/31"
    epw_content.append(header8)
    
    # Generate hourly data for one year (8760 hours)
    for day_of_year in range(1, 366):  # 1 to 365 (366 for leap year handling)
        for hour in range(1, 25):  # 1 to 24
            
            # Calculate date
            date = datetime.datetime(2024, 1, 1) + datetime.timedelta(days=day_of_year-1)
            month = date.month
            day = date.day
            
            # Adjust temperature based on season (higher in summer, lower in winter)
            seasonal_factor = math.cos((day_of_year - 202) * 2 * math.pi / 365) * 0.4  # Peak around July 21 (day 202)
            daily_factor = math.cos((hour - 14) * 2 * math.pi / 24) * 0.3  # Peak around 2 PM
            
            temp_c = base_temp_c + seasonal_factor * 15 + daily_factor * 8  # Vary ±15°C seasonally, ±8°C daily
            temp_c = max(-10, min(40, temp_c))  # Clamp to reasonable range
            
            # Adjust solar radiation based on season and time of day
            if 6 <= hour <= 18:  # Daylight hours
                solar_factor = math.sin((hour - 6) * math.pi / 12)  # Peak at noon
                seasonal_solar = 1 + seasonal_factor * 0.3  # More in summer
                global_horizontal = base_solar_radiation * solar_factor * seasonal_solar
                direct_normal = global_horizontal * 0.8 if global_horizontal > 100 else 0
                diffuse_horizontal = global_horizontal - direct_normal * math.sin(math.radians(45))  # Approximate
            else:
                global_horizontal = 0
                direct_normal = 0
                diffuse_horizontal = 0
                
            global_horizontal = max(0, global_horizontal)
            direct_normal = max(0, direct_normal)
            diffuse_horizontal = max(0, diffuse_horizontal)
            
            # Wind speed with some variation
            wind_speed = base_wind_speed + math.sin(day_of_year * 2 * math.pi / 365) * 1.5 + math.sin(hour * 2 * math.pi / 24) * 0.5
            wind_speed = max(0.5, min(15, wind_speed))
            
            # Wind direction (simplified - prevailing from southwest in Nashville)
            wind_direction = 225 + math.sin(day_of_year * 2 * math.pi / 365) * 45
            
            # Relative humidity (higher at night, lower during day)
            humidity_daily = math.cos((hour - 14) * 2 * math.pi / 24) * 20  # Lower at 2 PM
            relative_humidity = base_relative_humidity + humidity_daily + seasonal_factor * (-10)  # Lower in summer
            relative_humidity = max(20, min(95, relative_humidity))
            
            # Calculate other parameters
            dry_bulb_temp = temp_c
            dew_point_temp = temp_c - ((100 - relative_humidity) / 5)  # Approximate
            atmospheric_pressure = base_pressure * (1 - 0.0065 * elevation / 288.15) ** 5.255  # Barometric formula
            
            # Sky conditions (simplified)
            total_sky_cover = 5  # 0-10 scale, partly cloudy
            opaque_sky_cover = 3
            
            # Visibility and weather conditions
            visibility = 16000  # meters
            ceiling_height = 9999  # meters (clear)
            
            # Weather flags
            present_weather = 0  # No significant weather
            
            # Precipitable water and aerosol optical depth (estimates)
            precipitable_water = 20  # mm
            aerosol_optical_depth = 0.1
            snow_depth = 0
            days_since_last_snow = 99
            
            # Albedo (from document - mixed urban surface)
            albedo = 0.20  # Weighted average of urban surfaces
            
            # Liquid precipitation (simplified)
            liquid_precip_depth = 0  # mm
            liquid_precip_rate = 0  # mm/h
            
            # Create the hourly data line
            # EPW format: Year,Month,Day,Hour,Minute,Data Source and Uncertainty Flags,
            # Dry Bulb Temperature {C}, Dew Point Temperature {C}, Relative Humidity {%},
            # Atmospheric Station Pressure {Pa}, Extraterrestrial Horizontal Radiation {Wh/m2},
            # Extraterrestrial Direct Normal Radiation {Wh/m2}, Horizontal Infrared Radiation Intensity {Wh/m2},
            # Global Horizontal Radiation {Wh/m2}, Direct Normal Radiation {Wh/m2}, Diffuse Horizontal Radiation {Wh/m2},
            # Global Horizontal Illuminance {lux}, Direct Normal Illuminance {lux}, Diffuse Horizontal Illuminance {lux},
            # Zenith Luminance {Cd/m2}, Wind Direction {deg}, Wind Speed {m/s}, Total Sky Cover {tenths},
            # Opaque Sky Cover {tenths}, Visibility {km}, Ceiling Height {m}, Present Weather Observation,
            # Present Weather Codes, Precipitable Water {mm}, Aerosol Optical Depth {thousandths},
            # Snow Depth {cm}, Days Since Last Snowfall, Albedo {hundredths}, Liquid Precipitation Depth {mm},
            # Liquid Precipitation Quantity {hr}
            
            data_line = (
                f"2024,{month},{day},{hour},0,"
                f"?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,"
                f"{dry_bulb_temp:.1f},{dew_point_temp:.1f},{relative_humidity:.0f},"
                f"{atmospheric_pressure:.0f},0,0,300,"
                f"{global_horizontal:.0f},{direct_normal:.0f},{diffuse_horizontal:.0f},"
                f"0,0,0,0,"
                f"{wind_direction:.0f},{wind_speed:.1f},{total_sky_cover},{opaque_sky_cover},"
                f"{visibility/1000:.1f},{ceiling_height},{present_weather},0,"
                f"{precipitable_water},{aerosol_optical_depth*1000:.0f},"
                f"{snow_depth},{days_since_last_snow},{albedo*100:.0f},"
                f"{liquid_precip_depth:.1f},{liquid_precip_rate:.1f}"
            )
            
            epw_content.append(data_line)
    
    return "\n".join(epw_content)

def save_nashville_epw(filename="nashville_tn_custom.epw"):
    """Save the generated EPW file"""
    epw_content = generate_nashville_epw()
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(epw_content)
    
    print(f"EPW file saved as: {filename}")
    print(f"Total lines: {len(epw_content.split())}")
    print("\nFile ready for use with UWG model!")
    print(f"Update your Python script to use: epw_path = '{filename}'")

# Generate the EPW file
if __name__ == "__main__":
    save_nashville_epw()
    
    # Also create a simplified summer-focused version for the UWG simulation
    def create_summer_epw():
        """Create a simplified EPW file focused on summer conditions for the UWG simulation"""
        epw_content = []
        
        # Add headers (simplified)
        headers = [
            "LOCATION,Nashville,TN,USA,Custom_TUHI_Data,723270,36.12,-86.68,-6.0,183.0",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0", 
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            "COMMENTS 1,Summer conditions for Thompson Residential TUHI analysis",
            "COMMENTS 2,Based on environmental parameters from TUHI documentation",
            "DATA PERIODS,1,1,Data,Sunday,7/15,7/15"  # Single summer day
        ]
        
        epw_content.extend(headers)
        
        # Generate 24 hours of summer data (July 15th)
        for hour in range(1, 25):
            # Use the exact values from the TUHI document
            temp_c = 32.0 + math.cos((hour - 14) * 2 * math.pi / 24) * 5  # Peak at 2 PM
            
            if 6 <= hour <= 18:
                solar_factor = math.sin((hour - 6) * math.pi / 12)
                global_horizontal = 850.0 * solar_factor
                direct_normal = global_horizontal * 0.8 if global_horizontal > 100 else 0
                diffuse_horizontal = global_horizontal * 0.2
            else:
                global_horizontal = 0
                direct_normal = 0  
                diffuse_horizontal = 0
            
            wind_speed = 3.0  # Constant as specified in document
            relative_humidity = 65.0 + math.cos((hour - 14) * 2 * math.pi / 24) * 15
            
            dew_point = temp_c - ((100 - relative_humidity) / 5)
            pressure = 101325.0
            
            data_line = (
                f"2024,7,15,{hour},0,"
                f"?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,"
                f"{temp_c:.1f},{dew_point:.1f},{relative_humidity:.0f},"
                f"{pressure:.0f},0,0,400,"  # Using 400 W/m² longwave from document
                f"{global_horizontal:.0f},{direct_normal:.0f},{diffuse_horizontal:.0f},"
                f"0,0,0,0,"
                f"225,{wind_speed:.1f},3,2,"
                f"16.0,9999,0,0,"
                f"20,100,0,99,20,0.0,0.0"
            )
            
            epw_content.append(data_line)
        
        return "\n".join(epw_content)
    
    # Save the summer-focused version
    summer_epw = create_summer_epw()
    with open("nashville_summer_tuhi.epw", 'w', encoding='utf-8') as f:
        f.write(summer_epw)
    
    print("\nAlso created: nashville_summer_tuhi.epw (24-hour summer simulation)")
    print("This file matches the exact environmental conditions from your TUHI document:")
    print("- Air Temperature: 32°C (305K)")  
    print("- Solar Radiation: 850 W/m²")
    print("- Wind Speed: 3 m/s")
    print("- Longwave Radiation: 400 W/m²")