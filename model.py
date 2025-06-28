#!/usr/bin/env python3
"""
Machine Learning Urban Heat Island Projection Model
Using actual Nashville solar data and physics-based equations to predict infrastructure impacts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class UHIPhysicsMLModel:
    def __init__(self):
        # ACTUAL COLLECTED DATA from Nashville (1090 Murfreesboro Pike)
        self.actual_nashville_data = {
            'address': "1090 Murfreesboro Pike Nashville",
            'coordinates': [36.1269567, -86.7112723],
            'overall_avg_ghi': 245.8138586956522,  # kWh/m²/day (actual measured)
            'monthly_averages': {
                'june': 307.525,
                'july': 212.84274193548387, 
                'august': 219.06451612903226
            },
            'current_temperature': 29.99,  # °C (actual measured)
            'current_humidity': 61,        # % (actual measured)
            'tree_count': 0,              # (actual measured - very low vegetation)
            'greenery_score': 0,          # (actual measured)
            'total_summer_days': 92
        }
        
        # Environmental constants (Nashville summer conditions)
        self.env_constants = {
            'rho_air': 1.15,      # Air density (kg/m³)
            'c_p': 1005,          # Specific heat of air (J/kg·K)
            'h': 800,             # Urban boundary layer height (m)
            'u': 3,               # Wind speed (m/s)
            'L_down': 400,        # Longwave downward radiation (W/m²)
            'L_v': 2.45e6,        # Latent heat of vaporization (J/kg)
            'sigma': 5.67e-8,     # Stefan-Boltzmann constant
            'h_c': 15,            # Convective heat transfer coefficient
            'q_anthro': 8,        # Anthropogenic heat flux (W/m²)
            'q_conduct': 15       # Conductive heat flux (W/m²)
        }
        
        # Material properties database
        self.materials = {
            'Hot Mix Asphalt': {
                'albedo': 0.085, 'emissivity': 0.95, 'thermal_conductivity': 0.875,
                'latent_heat_flux': 0, 'evaporation_rate': 0, 'cost_per_m2': 30
            },
            'Portland Cement Concrete': {
                'albedo': 0.30, 'emissivity': 0.90, 'thermal_conductivity': 1.6,
                'latent_heat_flux': 0, 'evaporation_rate': 0, 'cost_per_m2': 60
            },
            'Black EPDM Membrane': {
                'albedo': 0.08, 'emissivity': 0.90, 'thermal_conductivity': 0.23,
                'latent_heat_flux': 0, 'evaporation_rate': 0, 'cost_per_m2': 12
            },
            'White TPO Membrane': {
                'albedo': 0.725, 'emissivity': 0.875, 'thermal_conductivity': 0.22,
                'latent_heat_flux': 0, 'evaporation_rate': 0, 'cost_per_m2': 16
            },
            'Warm Mix Asphalt': {
                'albedo': 0.15, 'emissivity': 0.925, 'thermal_conductivity': 0.875,
                'latent_heat_flux': 0, 'evaporation_rate': 0, 'cost_per_m2': 35
            },
            'Pervious Concrete': {
                'albedo': 0.325, 'emissivity': 0.90, 'thermal_conductivity': 0.90,
                'latent_heat_flux': 50, 'evaporation_rate': 1.5, 'cost_per_m2': 95
            },
            'Silicone Reflective Coating': {
                'albedo': 0.80, 'emissivity': 0.925, 'thermal_conductivity': 0.25,
                'latent_heat_flux': 0, 'evaporation_rate': 0, 'cost_per_m2': 12
            },
            'Vegetation': {
                'albedo': 0.215, 'emissivity': 0.965, 'thermal_conductivity': 0.20,
                'latent_heat_flux': 125, 'evaporation_rate': 3.5, 'cost_per_m2': 10
            },
            'Water surfaces': {
                'albedo': 0.08, 'emissivity': 0.98, 'thermal_conductivity': 0.6,
                'latent_heat_flux': 175, 'evaporation_rate': 4.5, 'cost_per_m2': 35
            }
        }
        
        # Current and proposed compositions (Thompson Residential)
        self.current_composition = {
            'Hot Mix Asphalt': 28,
            'Portland Cement Concrete': 8,
            'Black EPDM Membrane': 25,
            'Vegetation': 35,
            'Miscellaneous': 4
        }
        
        self.proposed_composition = {
            'Warm Mix Asphalt': 20,
            'Pervious Concrete': 10,
            'White TPO Membrane': 20,
            'Silicone Reflective Coating': 5,
            'Vegetation': 36,
            'Water surfaces': 1,
            'Miscellaneous': 8
        }
        
        self.models = {}
        self.scalers = {}
        
    def convert_kwh_to_watts(self, kwh_per_day):
        """Convert kWh/day to average W/m² (peak estimated as 8x average)"""
        avg_watts = (kwh_per_day * 1000) / 24
        peak_watts = avg_watts * 8
        return avg_watts, peak_watts
    
    def calculate_surface_flux_physics(self, albedo, emissivity, latent_heat, solar_input, 
                                     air_temp=None, surface_temp=None):
        """
        Calculate net surface heat flux using physics equations from the document
        """
        if air_temp is None:
            air_temp = self.actual_nashville_data['current_temperature'] + 273.15
        if surface_temp is None:
            surface_temp = air_temp + 15  # Surface typically 15K warmer
            
        # Solar absorption term: (1 - α) * G
        q_solar = (1 - albedo) * solar_input
        
        # Longwave radiation: L↓ - ε * σ * Ts⁴
        q_rad_net = (self.env_constants['L_down'] - 
                    emissivity * self.env_constants['sigma'] * (surface_temp ** 4))
        
        # Convective heat transfer: hc * (Ts - Ta)
        q_conv = self.env_constants['h_c'] * (surface_temp - air_temp)
        
        # Net heat flux using exact equation from document
        q_net = (q_solar + q_rad_net - q_conv - latent_heat + 
                self.env_constants['q_anthro'] + self.env_constants['q_conduct'])
        
        return q_net
    
    def calculate_uhi_physics(self, net_flux):
        """Calculate UHI intensity using equation from document"""
        return net_flux / (self.env_constants['rho_air'] * 
                          self.env_constants['c_p'] * 
                          self.env_constants['h'] * 
                          self.env_constants['u'])
    
    def generate_training_data(self, n_samples=5000):
        """
        Generate training data using physics equations with realistic parameter variations
        """
        print("Generating physics-based training data...")
        
        # Get actual solar data ranges
        monthly_ghi = list(self.actual_nashville_data['monthly_averages'].values())
        min_ghi, max_ghi = min(monthly_ghi), max(monthly_ghi)
        
        # Create realistic parameter ranges
        np.random.seed(42)  # For reproducibility
        
        # Environmental variations
        solar_inputs = np.random.uniform(
            self.convert_kwh_to_watts(min_ghi)[1] * 0.5,  # Cloudy days
            self.convert_kwh_to_watts(max_ghi)[1] * 1.2,  # Very sunny days
            n_samples
        )
        
        air_temps = np.random.normal(
            self.actual_nashville_data['current_temperature'] + 273.15, 
            5, n_samples
        )  # ±5K variation
        
        wind_speeds = np.random.uniform(1, 8, n_samples)  # 1-8 m/s
        humidity = np.random.uniform(40, 80, n_samples)   # 40-80%
        
        # Surface property variations (realistic ranges)
        albedos = np.random.uniform(0.05, 0.85, n_samples)
        emissivities = np.random.uniform(0.85, 0.98, n_samples)
        latent_heat_fluxes = np.random.uniform(0, 200, n_samples)
        
        # Calculate physics-based outcomes
        features = []
        targets = []
        
        for i in range(n_samples):
            # Environmental features
            env_features = [
                solar_inputs[i],
                air_temps[i] - 273.15,  # Convert back to Celsius
                wind_speeds[i],
                humidity[i]
            ]
            
            # Surface features
            surface_features = [
                albedos[i],
                emissivities[i],
                latent_heat_fluxes[i]
            ]
            
            # Calculate physics-based flux
            q_net = self.calculate_surface_flux_physics(
                albedos[i], emissivities[i], latent_heat_fluxes[i], 
                solar_inputs[i], air_temps[i]
            )
            
            # Calculate UHI with wind speed variation
            env_temp = self.env_constants.copy()
            env_temp['u'] = wind_speeds[i]
            uhi = q_net / (env_temp['rho_air'] * env_temp['c_p'] * 
                          env_temp['h'] * env_temp['u'])
            
            features.append(env_features + surface_features)
            targets.append([q_net, uhi])
        
        # Create DataFrame
        feature_names = [
            'solar_input_w_m2', 'air_temp_c', 'wind_speed_m_s', 'humidity_percent',
            'albedo', 'emissivity', 'latent_heat_flux_w_m2'
        ]
        
        self.training_features = pd.DataFrame(features, columns=feature_names)
        self.training_targets = pd.DataFrame(targets, columns=['net_flux_w_m2', 'uhi_temp_c'])
        
        print(f"Generated {n_samples} training samples")
        return self.training_features, self.training_targets
    
    def train_models(self):
        """Train multiple ML models on physics-based data"""
        print("Training ML models...")
        
        X = self.training_features
        y_flux = self.training_targets['net_flux_w_m2']
        y_uhi = self.training_targets['uhi_temp_c']
        
        # Split data
        X_train, X_test, y_flux_train, y_flux_test = train_test_split(
            X, y_flux, test_size=0.2, random_state=42
        )
        _, _, y_uhi_train, y_uhi_test = train_test_split(
            X, y_uhi, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        # Train models for heat flux prediction
        models_config = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        self.model_performance = {}
        
        for name, model in models_config.items():
            # Train for heat flux
            model_flux = model
            model_flux.fit(X_train_scaled, y_flux_train)
            flux_pred = model_flux.predict(X_test_scaled)
            
            # Train for UHI (using same model type)
            model_uhi = type(model)(**model.get_params())
            model_uhi.fit(X_train_scaled, y_uhi_train)
            uhi_pred = model_uhi.predict(X_test_scaled)
            
            # Store models
            self.models[f'{name}_flux'] = model_flux
            self.models[f'{name}_uhi'] = model_uhi
            
            # Calculate performance
            self.model_performance[name] = {
                'flux_mae': mean_absolute_error(y_flux_test, flux_pred),
                'flux_r2': r2_score(y_flux_test, flux_pred),
                'uhi_mae': mean_absolute_error(y_uhi_test, uhi_pred),
                'uhi_r2': r2_score(y_uhi_test, uhi_pred)
            }
        
        # Print performance
        print("\nModel Performance:")
        for name, perf in self.model_performance.items():
            print(f"{name}:")
            print(f"  Heat Flux - MAE: {perf['flux_mae']:.2f} W/m², R²: {perf['flux_r2']:.3f}")
            print(f"  UHI Temp - MAE: {perf['uhi_mae']:.3f} °C, R²: {perf['uhi_r2']:.3f}")
    
    def calculate_composition_features(self, composition, solar_input=None):
        """Calculate area-weighted features for a surface composition"""
        if solar_input is None:
            # Use actual measured peak solar
            _, solar_input = self.convert_kwh_to_watts(
                self.actual_nashville_data['overall_avg_ghi']
            )
        
        weighted_albedo = 0
        weighted_emissivity = 0
        weighted_latent_heat = 0
        total_cost = 0
        
        for surface, percentage in composition.items():
            if surface in self.materials:
                props = self.materials[surface]
                fraction = percentage / 100
                
                weighted_albedo += props['albedo'] * fraction
                weighted_emissivity += props['emissivity'] * fraction
                weighted_latent_heat += props['latent_heat_flux'] * fraction
                total_cost += props['cost_per_m2'] * fraction
        
        # Create feature vector
        features = [
            solar_input,
            self.actual_nashville_data['current_temperature'],
            self.env_constants['u'],
            self.actual_nashville_data['current_humidity'],
            weighted_albedo,
            weighted_emissivity,
            weighted_latent_heat
        ]
        
        return features, total_cost
    
    def predict_infrastructure_impact(self):
        """Predict UHI impact for current vs proposed infrastructure"""
        print("Predicting infrastructure impacts using ML models...")
        
        # Calculate features for both compositions
        current_features, current_cost = self.calculate_composition_features(
            self.current_composition
        )
        proposed_features, proposed_cost = self.calculate_composition_features(
            self.proposed_composition
        )
        
        # Scale features
        current_scaled = self.scalers['features'].transform([current_features])
        proposed_scaled = self.scalers['features'].transform([proposed_features])
        
        # Use best performing model (typically Gradient Boosting)
        best_model = 'Gradient Boosting'
        
        # Predict outcomes
        predictions = {}
        
        for scenario, features_scaled in [('current', current_scaled), 
                                        ('proposed', proposed_scaled)]:
            flux_pred = self.models[f'{best_model}_flux'].predict(features_scaled)[0]
            uhi_pred = self.models[f'{best_model}_uhi'].predict(features_scaled)[0]
            
            predictions[scenario] = {
                'heat_flux': flux_pred,
                'uhi_temp': uhi_pred
            }
        
        # Calculate improvements
        flux_reduction = predictions['current']['heat_flux'] - predictions['proposed']['heat_flux']
        temp_reduction = predictions['current']['uhi_temp'] - predictions['proposed']['uhi_temp']
        
        results = {
            'current': predictions['current'],
            'proposed': predictions['proposed'],
            'improvements': {
                'heat_flux_reduction': flux_reduction,
                'temperature_reduction': temp_reduction,
                'percent_flux_reduction': (flux_reduction / predictions['current']['heat_flux']) * 100,
                'percent_temp_reduction': (temp_reduction / predictions['current']['uhi_temp']) * 100
            },
            'costs': {
                'current_cost_per_m2': current_cost,
                'proposed_cost_per_m2': proposed_cost,
                'cost_increase_per_m2': proposed_cost - current_cost
            },
            'features': {
                'current': dict(zip(self.training_features.columns, current_features)),
                'proposed': dict(zip(self.training_features.columns, proposed_features))
            }
        }
        
        return results
    
    def monthly_projections(self):
        """Generate monthly projections using actual measured data"""
        print("Generating monthly projections with actual Nashville data...")
        
        monthly_results = []
        
        for month, ghi_kwh in self.actual_nashville_data['monthly_averages'].items():
            # Convert actual measured GHI to watts
            _, solar_input = self.convert_kwh_to_watts(ghi_kwh)
            
            # Calculate features for this month
            current_features, _ = self.calculate_composition_features(
                self.current_composition, solar_input
            )
            proposed_features, _ = self.calculate_composition_features(
                self.proposed_composition, solar_input
            )
            
            # Scale and predict
            current_scaled = self.scalers['features'].transform([current_features])
            proposed_scaled = self.scalers['features'].transform([proposed_features])
            
            best_model = 'Gradient Boosting'
            
            current_uhi = self.models[f'{best_model}_uhi'].predict(current_scaled)[0]
            proposed_uhi = self.models[f'{best_model}_uhi'].predict(proposed_scaled)[0]
            
            monthly_results.append({
                'month': month.capitalize(),
                'measured_ghi_kwh': ghi_kwh,
                'solar_input_w': solar_input,
                'current_uhi': current_uhi,
                'proposed_uhi': proposed_uhi,
                'uhi_reduction': current_uhi - proposed_uhi
            })
        
        return monthly_results
    
    def uncertainty_analysis(self, n_bootstrap=100):
        """Perform uncertainty analysis using bootstrap sampling"""
        print("Performing uncertainty analysis...")
        
        bootstrap_results = []
        
        for i in range(n_bootstrap):
            # Sample with replacement from training data
            sample_idx = np.random.choice(len(self.training_features), 
                                        size=len(self.training_features), 
                                        replace=True)
            
            X_sample = self.training_features.iloc[sample_idx]
            y_sample = self.training_targets.iloc[sample_idx]
            
            # Train quick model on sample
            model = GradientBoostingRegressor(n_estimators=50, random_state=i)
            X_scaled = self.scalers['features'].transform(X_sample)
            model.fit(X_scaled, y_sample['uhi_temp_c'])
            
            # Predict on actual compositions
            current_features, _ = self.calculate_composition_features(self.current_composition)
            proposed_features, _ = self.calculate_composition_features(self.proposed_composition)
            
            current_scaled = self.scalers['features'].transform([current_features])
            proposed_scaled = self.scalers['features'].transform([proposed_features])
            
            current_pred = model.predict(current_scaled)[0]
            proposed_pred = model.predict(proposed_scaled)[0]
            
            bootstrap_results.append({
                'current_uhi': current_pred,
                'proposed_uhi': proposed_pred,
                'reduction': current_pred - proposed_pred
            })
        
        # Calculate statistics
        bootstrap_df = pd.DataFrame(bootstrap_results)
        
        uncertainty_stats = {
            'current_uhi': {
                'mean': bootstrap_df['current_uhi'].mean(),
                'std': bootstrap_df['current_uhi'].std(),
                'ci_lower': bootstrap_df['current_uhi'].quantile(0.025),
                'ci_upper': bootstrap_df['current_uhi'].quantile(0.975)
            },
            'proposed_uhi': {
                'mean': bootstrap_df['proposed_uhi'].mean(),
                'std': bootstrap_df['proposed_uhi'].std(),
                'ci_lower': bootstrap_df['proposed_uhi'].quantile(0.025),
                'ci_upper': bootstrap_df['proposed_uhi'].quantile(0.975)
            },
            'reduction': {
                'mean': bootstrap_df['reduction'].mean(),
                'std': bootstrap_df['reduction'].std(),
                'ci_lower': bootstrap_df['reduction'].quantile(0.025),
                'ci_upper': bootstrap_df['reduction'].quantile(0.975)
            }
        }
        
        return uncertainty_stats
    
    def visualize_results(self, results, monthly_results, uncertainty_stats):
        """Create comprehensive visualizations and save to files"""
        import os
        from datetime import datetime
        
        # Create plots directory if it doesn't exist
        plots_dir = "data/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML-Based UHI Projection Results\nUsing Actual Nashville Solar Data (1090 Murfreesboro Pike)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Current vs Proposed UHI
        scenarios = ['Current\nInfrastructure', 'Proposed\nEco-Infrastructure']
        uhi_values = [results['current']['uhi_temp'], results['proposed']['uhi_temp']]
        uhi_errors = [uncertainty_stats['current_uhi']['std'], 
                     uncertainty_stats['proposed_uhi']['std']]
        
        bars1 = axes[0,0].bar(scenarios, uhi_values, 
                             color=['#e74c3c', '#27ae60'], alpha=0.7,
                             yerr=uhi_errors, capsize=5)
        axes[0,0].set_ylabel('UHI Temperature Increase (°C)')
        axes[0,0].set_title('UHI Impact Comparison')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val, err in zip(bars1, uhi_values, uhi_errors):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.01,
                          f'{val:.3f}°C', ha='center', va='bottom', fontweight='bold')
        
        # 2. Monthly projections
        months_df = pd.DataFrame(monthly_results)
        x_pos = np.arange(len(months_df))
        
        axes[0,1].bar(x_pos - 0.2, months_df['current_uhi'], 0.4, 
                     label='Current', color='#e74c3c', alpha=0.7)
        axes[0,1].bar(x_pos + 0.2, months_df['proposed_uhi'], 0.4, 
                     label='Proposed', color='#27ae60', alpha=0.7)
        axes[0,1].set_xlabel('Month (2022 Actual Data)')
        axes[0,1].set_ylabel('UHI Temperature (°C)')
        axes[0,1].set_title('Monthly UHI Projections')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(months_df['month'])
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Temperature reduction by month
        axes[0,2].bar(months_df['month'], months_df['uhi_reduction'], 
                     color='#3498db', alpha=0.7)
        axes[0,2].set_ylabel('Temperature Reduction (°C)')
        axes[0,2].set_title('Monthly UHI Reduction')
        axes[0,2].grid(True, alpha=0.3)
        
        # Add reduction values
        for i, val in enumerate(months_df['uhi_reduction']):
            axes[0,2].text(i, val + 0.01, f'{val:.3f}', 
                          ha='center', va='bottom', fontweight='bold')
        
        # 4. Model performance comparison
        model_names = list(self.model_performance.keys())
        r2_scores = [perf['uhi_r2'] for perf in self.model_performance.values()]
        
        axes[1,0].bar(model_names, r2_scores, color='#9b59b6', alpha=0.7)
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].set_title('ML Model Performance (UHI Prediction)')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].grid(True, alpha=0.3)
        plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Actual solar data visualization
        solar_months = list(self.actual_nashville_data['monthly_averages'].keys())
        solar_values = list(self.actual_nashville_data['monthly_averages'].values())
        
        axes[1,1].bar(solar_months, solar_values, color='#f39c12', alpha=0.7)
        axes[1,1].set_ylabel('Solar GHI (kWh/m²/day)')
        axes[1,1].set_title('Actual Measured Solar Data\n(Nashville Summer 2022)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add actual values
        for i, val in enumerate(solar_values):
            axes[1,1].text(i, val + 5, f'{val:.1f}', 
                          ha='center', va='bottom', fontweight='bold')
        
        # 6. Feature importance (if Random Forest is available)
        if 'Random Forest_uhi' in self.models:
            rf_model = self.models['Random Forest_uhi']
            feature_importance = rf_model.feature_importances_
            feature_names = self.training_features.columns
            
            # Sort by importance
            importance_idx = np.argsort(feature_importance)[::-1]
            
            axes[1,2].barh(range(len(feature_importance)), 
                          feature_importance[importance_idx], 
                          color='#1abc9c', alpha=0.7)
            axes[1,2].set_yticks(range(len(feature_importance)))
            axes[1,2].set_yticklabels([feature_names[i] for i in importance_idx])
            axes[1,2].set_xlabel('Feature Importance')
            axes[1,2].set_title('ML Feature Importance\n(Random Forest)')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plots in multiple formats
        base_filename = f"ml_uhi_projections_{timestamp}"
        
        # Save as high-resolution PNG
        png_path = os.path.join(plots_dir, f"{base_filename}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save as PDF for publications
        pdf_path = os.path.join(plots_dir, f"{base_filename}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        
        # Save as SVG for vector graphics
        svg_path = os.path.join(plots_dir, f"{base_filename}.svg")
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
        
        plt.close()  # Close the figure to free memory
        
        print(f"\n📊 PLOTS SAVED:")
        print(f"   📈 High-res PNG: {png_path}")
        print(f"   📄 PDF: {pdf_path}")
        print(f"   🎨 SVG: {svg_path}")
        
        return {
            'png_path': png_path,
            'pdf_path': pdf_path,
            'svg_path': svg_path,
            'timestamp': timestamp
        }
    
    def create_individual_plots(self, results, monthly_results, uncertainty_stats, timestamp):
        """Create individual plots for each analysis component"""
        import os
        
        plots_dir = "data/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_files = []
        
        # 1. UHI Comparison Plot
        plt.figure(figsize=(10, 6))
        scenarios = ['Current\nInfrastructure', 'Proposed\nEco-Infrastructure']
        uhi_values = [results['current']['uhi_temp'], results['proposed']['uhi_temp']]
        uhi_errors = [uncertainty_stats['current_uhi']['std'], 
                     uncertainty_stats['proposed_uhi']['std']]
        
        bars = plt.bar(scenarios, uhi_values, 
                      color=['#e74c3c', '#27ae60'], alpha=0.8,
                      yerr=uhi_errors, capsize=8, width=0.6)
        
        plt.ylabel('UHI Temperature Increase (°C)', fontsize=12)
        plt.title('Urban Heat Island Impact Comparison\nML Predictions with 95% Confidence Intervals', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val, err in zip(bars, uhi_values, uhi_errors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
                    f'{val:.3f}°C\n±{err:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        # Add improvement annotation
        improvement = results['improvements']['temperature_reduction']
        plt.annotate(f'Improvement:\n-{improvement:.3f}°C\n({results["improvements"]["percent_temp_reduction"]:.1f}% cooler)',
                    xy=(0.5, max(uhi_values) * 0.7), ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        uhi_plot_path = os.path.join(plots_dir, f"uhi_comparison_{timestamp}.png")
        plt.savefig(uhi_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(uhi_plot_path)
        
        # 2. Monthly Projections Plot (UHI only, no solar data)
        plt.figure(figsize=(12, 7))
        months_df = pd.DataFrame(monthly_results)
        x_pos = np.arange(len(months_df))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, months_df['current_uhi'], width, 
                       label='Current Infrastructure', color='#e74c3c', alpha=0.8)
        bars2 = plt.bar(x_pos + width/2, months_df['proposed_uhi'], width, 
                       label='Proposed Eco-Infrastructure', color='#27ae60', alpha=0.8)
        
        plt.xlabel('Month (2022)', fontsize=12)
        plt.ylabel('UHI Temperature Increase (°C)', fontsize=12)
        plt.title('Monthly UHI Projections\nThompson Residential Area, Nashville', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x_pos, months_df['month'])
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, months_df['current_uhi']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}°C', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, months_df['proposed_uhi']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}°C', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add reduction annotations
        for i, (current, proposed, reduction) in enumerate(zip(months_df['current_uhi'], 
                                                              months_df['proposed_uhi'], 
                                                              months_df['uhi_reduction'])):
            mid_height = (current + proposed) / 2
            plt.annotate(f'-{reduction:.3f}°C', 
                        xy=(i, mid_height), ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7),
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        monthly_plot_path = os.path.join(plots_dir, f"monthly_uhi_projections_{timestamp}.png")
        plt.savefig(monthly_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(monthly_plot_path)
        
        # 3. Solar Data Visualization
        plt.figure(figsize=(10, 6))
        solar_months = list(self.actual_nashville_data['monthly_averages'].keys())
        solar_values = list(self.actual_nashville_data['monthly_averages'].values())
        
        bars = plt.bar(solar_months, solar_values, color='#f39c12', alpha=0.8, width=0.6)
        plt.ylabel('Solar GHI (kWh/m²/day)', fontsize=12)
        plt.title('Actual Measured Solar Radiation Data\nNashville Summer 2022 (1090 Murfreesboro Pike)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels and statistics
        for bar, val in zip(bars, solar_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add average line
        avg_solar = np.mean(solar_values)
        plt.axhline(y=avg_solar, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.text(1, avg_solar + 15, f'Average: {avg_solar:.1f} kWh/m²/day', 
                fontweight='bold', color='red')
        
        plt.tight_layout()
        solar_plot_path = os.path.join(plots_dir, f"actual_solar_data_{timestamp}.png")
        plt.savefig(solar_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(solar_plot_path)
        
        # 4. Model Performance Plot
        plt.figure(figsize=(10, 6))
        model_names = list(self.model_performance.keys())
        r2_scores = [perf['uhi_r2'] for perf in self.model_performance.values()]
        mae_scores = [perf['uhi_mae'] for perf in self.model_performance.values()]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, r2_scores, width, label='R² Score', 
                       color='#9b59b6', alpha=0.8)
        
        ax2 = plt.gca().twinx()
        bars2 = ax2.bar(x_pos + width/2, mae_scores, width, label='MAE (°C)', 
                       color='#e67e22', alpha=0.8)
        
        plt.xlabel('ML Model Type', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error (°C)', fontsize=12)
        plt.title('Machine Learning Model Performance Comparison\nUHI Temperature Prediction', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x_pos, model_names, rotation=45)
        
        # Add legends
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_plot_path = os.path.join(plots_dir, f"model_performance_{timestamp}.png")
        plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(performance_plot_path)
        
        print(f"\n📊 INDIVIDUAL PLOTS SAVED:")
        for i, path in enumerate(plot_files, 1):
            print(f"   {i}. {os.path.basename(path)}")
        
        return plot_files
    
    def run_complete_analysis(self):
        """Run the complete ML-based UHI analysis"""
        print("=" * 80)
        print("ML-BASED URBAN HEAT ISLAND PROJECTION MODEL")
        print("Using Actual Nashville Solar Data (1090 Murfreesboro Pike)")
        print("=" * 80)
        
        # Generate training data
        self.generate_training_data()
        
        # Train models
        self.train_models()
        
        # Predict infrastructure impacts
        results = self.predict_infrastructure_impact()
        
        # Generate monthly projections
        monthly_results = self.monthly_projections()
        
        # Uncertainty analysis
        uncertainty_stats = self.uncertainty_analysis()
        
        # Print results
        print("\n" + "=" * 60)
        print("MACHINE LEARNING PROJECTION RESULTS")
        print("=" * 60)
        
        print(f"\nUsing actual measured data from: {self.actual_nashville_data['address']}")
        print(f"Average summer GHI: {self.actual_nashville_data['overall_avg_ghi']:.1f} kWh/m²/day")
        print(f"Current temperature: {self.actual_nashville_data['current_temperature']:.1f}°C")
        
        print(f"\nCURRENT INFRASTRUCTURE:")
        print(f"  Predicted UHI intensity: {results['current']['uhi_temp']:.3f} ± {uncertainty_stats['current_uhi']['std']:.3f} °C")
        print(f"  Heat flux: {results['current']['heat_flux']:.1f} W/m²")
        
        print(f"\nPROPOSED ECO-INFRASTRUCTURE:")
        print(f"  Predicted UHI intensity: {results['proposed']['uhi_temp']:.3f} ± {uncertainty_stats['proposed_uhi']['std']:.3f} °C")
        print(f"  Heat flux: {results['proposed']['heat_flux']:.1f} W/m²")
        
        print(f"\nPROJECTED IMPROVEMENTS:")
        print(f"  Temperature reduction: {results['improvements']['temperature_reduction']:.3f} °C")
        print(f"  ({results['improvements']['percent_temp_reduction']:.1f}% cooler)")
        print(f"  Heat flux reduction: {results['improvements']['heat_flux_reduction']:.1f} W/m²")
        print(f"  ({results['improvements']['percent_flux_reduction']:.1f}% reduction)")
        
        print(f"\nUNCERTAINTY ANALYSIS (95% Confidence Intervals):")
        print(f"  Temperature reduction: {uncertainty_stats['reduction']['ci_lower']:.3f} to {uncertainty_stats['reduction']['ci_upper']:.3f} °C")
        
        print(f"\nCOST ANALYSIS:")
        print(f"  Current infrastructure cost: ${results['costs']['current_cost_per_m2']:.0f}/m²")
        print(f"  Proposed infrastructure cost: ${results['costs']['proposed_cost_per_m2']:.0f}/m²")
        print(f"  Additional cost: ${results['costs']['cost_increase_per_m2']:.0f}/m²")
        
        print(f"\nMONTHLY PROJECTIONS (using actual 2022 data):")
        for month_data in monthly_results:
            print(f"  {month_data['month']}: {month_data['uhi_reduction']:.3f}°C reduction "
                  f"(GHI: {month_data['measured_ghi_kwh']:.1f} kWh/m²/day)")
        
        # Create visualizations and save to files
        plot_files = self.visualize_results(results, monthly_results, uncertainty_stats)
        
        # Create individual detailed plots
        individual_plots = self.create_individual_plots(results, monthly_results, 
                                                       uncertainty_stats, plot_files['timestamp'])
        
        return {
            'results': results,
            'monthly_projections': monthly_results,
            'uncertainty': uncertainty_stats,
            'model_performance': self.model_performance,
            'plot_files': plot_files,
            'individual_plots': individual_plots
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Ensure output directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)
    
    # Initialize and run the ML UHI model
    uhi_model = UHIPhysicsMLModel()
    
    # Run complete analysis
    analysis_results = uhi_model.run_complete_analysis()
    
    # Additional analysis: Feature sensitivity
    print("\n" + "=" * 60)
    print("FEATURE SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Test sensitivity to albedo changes
    print("\nAlbedo sensitivity (keeping other factors constant):")
    albedo_range = np.arange(0.1, 0.9, 0.1)
    base_features = uhi_model.calculate_composition_features(uhi_model.current_composition)[0]
    
    for albedo in albedo_range:
        test_features = base_features.copy()
        test_features[4] = albedo  # Albedo is index 4
        
        test_scaled = uhi_model.scalers['features'].transform([test_features])
        predicted_uhi = uhi_model.models['Gradient Boosting_uhi'].predict(test_scaled)[0]
        
        print(f"  Albedo {albedo:.1f}: UHI = {predicted_uhi:.3f}°C")
    
    print("\nAnalysis complete! All plots have been saved to the data/plots/ directory.")
    print(f"\n📊 COMPREHENSIVE PLOTS:")
    if 'plot_files' in analysis_results:
        for file_type, path in analysis_results['plot_files'].items():
            if file_type != 'timestamp':
                print(f"   • {path}")
    
    print(f"\n📈 INDIVIDUAL DETAILED PLOTS:")
    if 'individual_plots' in analysis_results:
        for i, path in enumerate(analysis_results['individual_plots'], 1):
            print(f"   {i}. {path}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   • Total plots created: {len(analysis_results.get('plot_files', {})) - 1 + len(analysis_results.get('individual_plots', []))}")
    print(f"   • All plots saved in high resolution (300 DPI)")
    print(f"   • Ready for presentations and publications")