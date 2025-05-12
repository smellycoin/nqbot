import os
import sys
import json
import time
import argparse
import pickle
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import threading
import warnings
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import joblib
warnings.filterwarnings('ignore')

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Bright text colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

# Optional LLM components
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pass

class NQTradingAssistant:
    def __init__(self, use_llm=False, use_market_data=False, llm_model="gpt2", default_ticker="NQ", default_timeframe="1h", json_output_enabled=False):
        self.output_dir = "trading_recommendations"
        self.models_dir = "trading_models"
        self.cache_dir = "market_data_cache"
        self.logs_dir = "trading_logs"
        self.history_file = os.path.join(self.output_dir, "trade_history.csv")
        self.learning_file = os.path.join(self.models_dir, "learning_data.pkl")
        self.model_file = os.path.join(self.models_dir, "liquidity_detector.joblib")
        self.pattern_model_file = os.path.join(self.models_dir, "pattern_recognizer.joblib")
        self.is_model_loaded = True
        self.use_llm = use_llm
        self.use_market_data = use_market_data
        self.default_ticker = default_ticker
        self.default_timeframe = default_timeframe
        self.json_output_enabled = json_output_enabled
        self.llm = None
        self.liquidity_model = None
        self.pattern_model = None
        self.scaler = None
        self.cached_market_data = {}
        self.market_data_timestamp = datetime.now() - timedelta(days=1)  # Force initial refresh
        self.learning_data = {
            "image_features": [],
            "analysis_results": [],
            "success_metrics": [],
            "liquidity_zones": [],
            "timeframes": [],
            "tickers": [],
            "last_training": None
        }
        
        # Create necessary directories
        for directory in [self.output_dir, self.models_dir, self.cache_dir, self.logs_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize trade history
        if not os.path.exists(self.history_file):
            pd.DataFrame(columns=[
                'timestamp', 'source', 'trend', 'support', 'resistance', 
                'recommended_trade', 'confidence', 'outcome', 'liquidity_zones',
                'success_rate', 'image_hash', 'timeframe', 'ticker'
            ]).to_csv(self.history_file, index=False)
        
        # Load learning data if available
        self._load_learning_data()
        
        # Initialize or load models
        self._initialize_models()
            
        # Initialize lightweight LLM
        if self.use_llm:
            try:
                self.llm = pipeline('text-generation', 
                                   model=llm_model,  # Can use better models like 'microsoft/phi-2'
                                   device=-1,     # CPU only
                                   max_length=300)
                self.print_formatted(f"LLM ({llm_model}) loaded successfully...", color=Colors.GREEN)
            except Exception as e:
                self.print_formatted(f"Could not load LLM: {e}", color=Colors.RED)
                self.use_llm = False
        
        # Initialize market data if requested
        if self.use_market_data:
            self._refresh_market_data()
            
        self.print_formatted("Enhanced NQ Trading Assistant initialized with dynamic learning capabilities...", color=Colors.CYAN, bold=True)
        
    def print_formatted(self, message, color=Colors.WHITE, bold=False, save_to_file=True, prefix=None, message_type=None):
        """Print colorized and formatted output to terminal and optionally save to log file
        
        Args:
            message: The message to print
            color: ANSI color code to use
            bold: Whether to make the text bold
            save_to_file: Whether to save this message to the log file
            prefix: Optional prefix for the log file name
            message_type: Type of message for automatic styling (info, success, error, warning, 
                          bullish, bearish, neutral, header, subheader, liquidity, support, resistance)
        """
        # Auto-assign colors based on message type if provided
        if message_type:
            if message_type == "header":
                color = Colors.BRIGHT_CYAN
                bold = True
                message = f"\n{'='*50}\n{message}\n{'='*50}"
            elif message_type == "subheader":
                color = Colors.BRIGHT_BLUE
                bold = True
                message = f"\n{'-'*40}\n{message}\n{'-'*40}"
            elif message_type == "info":
                color = Colors.CYAN
            elif message_type == "success":
                color = Colors.BRIGHT_GREEN
                bold = True
            elif message_type == "error":
                color = Colors.BRIGHT_RED
                bold = True
                message = f"ERROR: {message}"
            elif message_type == "warning":
                color = Colors.BRIGHT_YELLOW
                bold = True
            elif message_type == "bullish":
                color = Colors.GREEN
                bold = True
            elif message_type == "bearish":
                color = Colors.RED
            elif message_type == "neutral":
                color = Colors.BLUE
            elif message_type == "header":
                color = Colors.BRIGHT_WHITE
                bold = True
                message = f"\n{'=' * 50}\n{message}\n{'=' * 50}"
            elif message_type == "subheader":
                color = Colors.BRIGHT_WHITE
                bold = True
                message = f"\n{'-' * 40}\n{message}\n{'-' * 40}"
            elif message_type == "liquidity":
                color = Colors.MAGENTA
                bold = True
            elif message_type == "support":
                color = Colors.GREEN
            elif message_type == "resistance":
                color = Colors.RED
        
        # Format for terminal
        formatted_msg = ""
        if bold:
            formatted_msg += Colors.BOLD
        formatted_msg += color + message + Colors.RESET
        
        # Print to terminal
        if not self.json_output_enabled:
            print(formatted_msg)
        
        # Save to log file if requested
        if save_to_file:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            prefix = prefix or self.default_ticker
            log_filename = os.path.join(self.logs_dir, f"{prefix}_{timestamp}_log.txt")
            
            # Create plain text version (without ANSI codes)
            plain_msg = message
            
            # Append to log file with timestamp and optional type indicator
            with open(log_filename, 'a', encoding='utf-8') as f:
                type_indicator = f"[{message_type.upper()}] " if message_type else ""
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {type_indicator}{plain_msg}\n")
                
    def print_analysis_section(self, title, content, save_to_file=True):
        """Print a formatted analysis section with title and content
        
        Args:
            title: Section title
            content: Section content (string or list of strings)
            save_to_file: Whether to save to log file
        """
        self.print_formatted(title, message_type="subheader", save_to_file=save_to_file)
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, tuple) and len(item) == 2:
                    # Handle (message, message_type) tuples
                    self.print_formatted(f"  • {item[0]}", message_type=item[1], save_to_file=save_to_file)
                else:
                    # Regular string items
                    self.print_formatted(f"  • {item}", save_to_file=save_to_file)
        else:
            # Single string content
            self.print_formatted(content, save_to_file=save_to_file)
        
        # Add a blank line after each section
        print()
        if save_to_file:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            prefix = self.default_ticker
            log_filename = os.path.join(self.logs_dir, f"{prefix}_{timestamp}_log.txt")
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write("\n")
                
    def print_trade_signal(self, signal_type, confidence, details=None, save_to_file=True):
        """Print a formatted trade signal with appropriate styling
        
        Args:
            signal_type: Type of signal (bullish, bearish, neutral)
            confidence: Confidence score (0.0-1.0)
            details: Additional details about the signal
            save_to_file: Whether to save to log file
        """
        # Format confidence as percentage
        confidence_pct = confidence * 100
        
        # Determine signal styling
        if signal_type.lower() == "bullish":
            message_type = "bullish"
            signal_icon = "▲"
        elif signal_type.lower() == "bearish":
            message_type = "bearish"
            signal_icon = "▼"
        else:
            message_type = "neutral"
            signal_icon = "◆"
        
        # Create signal header
        signal_header = f"{signal_icon} {signal_type.upper()} SIGNAL {signal_icon} (Confidence: {confidence_pct:.1f}%)"
        self.print_formatted(signal_header, message_type=message_type, bold=True, save_to_file=save_to_file)
        
        # Print additional details if provided
        if details:
            if isinstance(details, list):
                for detail in details:
                    self.print_formatted(f"  • {detail}", message_type=message_type, save_to_file=save_to_file)
            else:
                self.print_formatted(f"  • {details}", message_type=message_type, save_to_file=save_to_file)
        
        # Add a blank line
        print()
        if save_to_file:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            prefix = self.default_ticker
            log_filename = os.path.join(self.logs_dir, f"{prefix}_{timestamp}_log.txt")
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write("\n")
    
    def _load_learning_data(self):
        """Load previous learning data"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'rb') as f:
                    self.learning_data = pickle.load(f)
                print(f"Loaded learning data with {len(self.learning_data['image_features'])} training examples")
            except Exception as e:
                print(f"Error loading learning data: {e}")
    
    def _save_learning_data(self):
        """Save current learning data"""
        try:
            with open(self.learning_file, 'wb') as f:
                pickle.dump(self.learning_data, f)
        except Exception as e:
            print(f"Error saving learning data: {e}")
    
    def _initialize_models(self):
        """Initialize or load ML models"""
        # Load or create liquidity detection model
        if os.path.exists(self.model_file):
            try:
                self.liquidity_model = joblib.load(self.model_file)
                self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.joblib"))
                print("Liquidity detection model loaded...")
            except Exception as e:
                print(f"Error loading liquidity model: {e}")
                self._create_new_models()
        else:
            self._create_new_models()
            
        # Load or create pattern recognition model
        if os.path.exists(self.pattern_model_file):
            try:
                self.pattern_model = joblib.load(self.pattern_model_file)
                print("Pattern recognition model loaded...")
            except Exception as e:
                print(f"Error loading pattern model: {e}")
                self._create_new_pattern_model()
        else:
            self._create_new_pattern_model()
    
    def _create_new_models(self):
        """Create new ML models"""
        print("Creating new liquidity detection model...")
        self.liquidity_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # If we have existing learning data, train immediately
        if len(self.learning_data["image_features"]) > 5:
            self._retrain_models()
        else:
            # Initialize scaler with default data to prevent "not fitted" errors
            print("Not enough training data for initial model training, initializing with defaults")
            # Create a small default dataset to fit the scaler
            default_data = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
            self.scaler.fit(default_data)
            # Train model with default data
            self.liquidity_model.fit(default_data, ["none", "both"])
            
        # Save even untrained models (will be updated as data comes in)
        joblib.dump(self.liquidity_model, self.model_file)
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.joblib"))
    
    def _create_new_pattern_model(self):
        """Create new pattern recognition model"""
        print("Creating new pattern recognition model...")
        self.pattern_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # If we have existing learning data, train immediately
        if len(self.learning_data["image_features"]) > 5:
            self._retrain_pattern_model()
        else:
            # Initialize with default data to prevent errors
            print("Not enough training data for initial pattern model training, initializing with defaults")
            # Create a small default dataset to train the model
            if hasattr(self.scaler, 'n_features_in_'):
                # Use the already fitted scaler if available
                default_data = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
                scaled_default = self.scaler.transform(default_data)
                # Train with default success metrics
                self.pattern_model.fit(scaled_default, [0.5, 0.5])
            
        # Save even untrained models (will be updated as data comes in)
        joblib.dump(self.pattern_model, self.pattern_model_file)
    
    def _retrain_models(self):
        """Retrain models with accumulated data"""
        if len(self.learning_data["image_features"]) < 5:
            print("Not enough data to retrain models")
            return
            
        try:
            # Prepare data for liquidity model
            X = np.array(self.learning_data["image_features"])
            y = np.array(self.learning_data["liquidity_zones"])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train liquidity model
            self.liquidity_model.fit(X_scaled, y)
            
            # Save models
            joblib.dump(self.liquidity_model, self.model_file)
            joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.joblib"))
            
            self.learning_data["last_training"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_learning_data()
            
            print(f"Models retrained with {len(X)} examples")
        except Exception as e:
            print(f"Error retraining models: {e}")
    
    def _retrain_pattern_model(self):
        """Retrain pattern recognition model"""
        if len(self.learning_data["image_features"]) < 5:
            print("Not enough data to retrain pattern model")
            return
            
        try:
            # Prepare data for pattern model
            X = np.array(self.learning_data["image_features"])
            y = np.array(self.learning_data["success_metrics"])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Train pattern model
            self.pattern_model.fit(X_scaled, y)
            
            # Save model
            joblib.dump(self.pattern_model, self.pattern_model_file)
            
            print(f"Pattern model retrained with {len(X)} examples")
        except Exception as e:
            print(f"Error retraining pattern model: {e}")
    
    def learn_from_outcome(self, image_hash, outcome_score):
        """Update models with trade outcome"""
        self.print_formatted(f"Recording trade outcome for image hash: {image_hash}", message_type="header")
        
        try:
            history_df = pd.read_csv(self.history_file)
            
            # Find the entry with matching image hash
            matching_rows = history_df[history_df['image_hash'] == image_hash]
            if len(matching_rows) == 0:
                self.print_formatted(f"No matching trade found for hash {image_hash}", message_type="error")
                return False
                
            # Get trade details for reporting
            trade_details = matching_rows.iloc[0]
            ticker = trade_details.get('ticker', self.default_ticker)
            timeframe = trade_details.get('timeframe', self.default_timeframe)
            trend = trade_details.get('trend', 'unknown')
            
            # Update outcome
            row_idx = matching_rows.index[0]
            history_df.at[row_idx, 'outcome'] = outcome_score
            history_df.to_csv(self.history_file, index=False)
            
            # Print trade details
            self.print_formatted(f"Trade details:", message_type="info")
            self.print_formatted(f"  • Ticker: {ticker}", message_type="info")
            self.print_formatted(f"  • Timeframe: {timeframe}", message_type="info")
            self.print_formatted(f"  • Trend: {trend}", message_type=trend.lower())
            
            # Format outcome message based on score
            if outcome_score >= 0.7:
                outcome_message = f"Successful trade recorded (Score: {outcome_score:.2f})"
                message_type = "success"
            elif outcome_score >= 0.5:
                outcome_message = f"Moderately successful trade recorded (Score: {outcome_score:.2f})"
                message_type = "bullish"
            else:
                outcome_message = f"Unsuccessful trade recorded (Score: {outcome_score:.2f})"
                message_type = "bearish"
                
            self.print_formatted(outcome_message, message_type=message_type)
            
            # Find corresponding learning data entry
            updated = False
            for i, features in enumerate(self.learning_data["image_features"]):
                if i < len(self.learning_data["success_metrics"]):
                    self.learning_data["success_metrics"][i] = outcome_score
                    self._save_learning_data()
                    updated = True
                    
                    # Retrain if we have enough updates (at least 3 new outcomes)
                    recent_outcomes = sum(1 for metric in self.learning_data["success_metrics"][-10:] 
                                         if metric is not None)
                    if recent_outcomes >= 3:
                        self.print_formatted("Retraining models with new outcome data...", message_type="warning")
                        self._retrain_models()
                        self._retrain_pattern_model()
                    break
            
            if updated:
                self.print_formatted("Learning data updated successfully", message_type="success")
                return True
            else:
                self.print_formatted("Failed to update learning data", message_type="error")
                return False
                
        except Exception as e:
            self.print_formatted(f"Error recording outcome: {str(e)}", message_type="error")
            return False
    
    def _compute_image_hash(self, image_path):
        """Compute a hash for the image to track it"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(int(time.time()))
    
    def extract_features_from_image(self, image_path):
        """Enhanced feature extraction with liquidity analysis"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            pixel_data = np.array(img)
            
            # Color analysis
            green_mask = (pixel_data[:,:,1] > pixel_data[:,:,0] + 30) & (pixel_data[:,:,1] > pixel_data[:,:,2] + 30)
            green_count = np.sum(green_mask)
            red_mask = (pixel_data[:,:,0] > pixel_data[:,:,1] + 30) & (pixel_data[:,:,0] > pixel_data[:,:,2] + 30)
            red_count = np.sum(red_mask)
            
            # Edge detection
            gray = np.array(ImageOps.grayscale(img))
            v_edges = np.abs(np.diff(gray, axis=0))
            h_edges = np.abs(np.diff(gray, axis=1))
            v_edges = np.pad(v_edges, ((0, 1), (0, 0)), mode='constant')
            h_edges = np.pad(h_edges, ((0, 0), (0, 1)), mode='constant')
            edges = np.maximum(v_edges, h_edges)
            
            # Enhanced liquidity analysis
            # Split image into vertical regions
            num_regions = 10
            region_height = height // num_regions
            region_densities = []
            
            for i in range(num_regions):
                start_y = i * region_height
                end_y = (i+1) * region_height
                region = edges[start_y:end_y, :]
                density = np.sum(region > 40) / (region_height * width)
                region_densities.append(density)
            
            # Find peaks in density (potential liquidity zones)
            peaks, _ = find_peaks(region_densities, height=0.1, distance=2)
            peak_densities = [region_densities[p] for p in peaks] if len(peaks) > 0 else [0]
            
            # Advanced horizontal line detection
            horizontal_lines = []
            for y in range(0, height, 3):  # More granular scanning
                row = h_edges[y, :]
                if np.sum(row > 30) > width * 0.4:  # More sensitive threshold
                    horizontal_lines.append(y / height)  # Normalize position
            
            # Volume analysis (approximated from color intensity in lower part)
            lower_third = pixel_data[int(height*0.7):, :, :]
            volume_intensity = np.mean(np.max(lower_third, axis=2))
            volume_normalized = min(volume_intensity / 128, 1.0)
            
            # Compute higher-level features
            features = {
                "green_ratio": green_count / (green_count + red_count + 1),
                "red_ratio": red_count / (green_count + red_count + 1),
                "color_intensity": np.mean(pixel_data),
                "edge_density": np.sum(edges > 30) / (width * height),
                "max_region_density": max(region_densities),
                "min_region_density": min(region_densities),
                "avg_region_density": np.mean(region_densities),
                "num_density_peaks": len(peaks),
                "max_peak_density": max(peak_densities),
                "horizontal_line_count": len(horizontal_lines),
                "horizontal_line_spacing": np.std(horizontal_lines) if len(horizontal_lines) > 1 else 0,
                "upper_wick_density": np.mean(region_densities[:3]),
                "lower_wick_density": np.mean(region_densities[-3:]),
                "middle_region_density": np.mean(region_densities[4:6]),
                "volume_indicator": volume_normalized
            }
            
            # Create flat feature vector for ML model
            feature_vector = [
                features["green_ratio"],
                features["red_ratio"],
                features["edge_density"],
                features["max_region_density"],
                features["min_region_density"],
                features["avg_region_density"],
                features["num_density_peaks"],
                features["max_peak_density"],
                features["horizontal_line_count"], 
                features["horizontal_line_spacing"],
                features["upper_wick_density"],
                features["lower_wick_density"],
                features["middle_region_density"],
                features["volume_indicator"]
            ]
            
            features["vector"] = feature_vector
            features["image_size"] = (width, height)
            features["density_regions"] = region_densities
            features["horizontal_lines"] = horizontal_lines
            
            return features
        except Exception as e:
            print(f"Image processing error: {e}")
            return {"error": str(e)}
    
    def analyze_liquidity_with_ml(self, features):
        """Use ML model to identify liquidity zones"""
        if self.liquidity_model is None:
            return self.analyze_liquidity(features)  # Fallback to rule-based
            
        try:
            # Scale features using the same scaler used for training
            feature_vector = np.array(features["vector"]).reshape(1, -1)
            
            # Check if scaler is fitted before using it
            if not hasattr(self.scaler, 'n_features_in_'):
                self.print_formatted("Scaler not fitted yet, using default analysis", message_type="warning", save_to_file=False)
                return self.analyze_liquidity(features)  # Fallback to rule-based
                
            scaled_features = self.scaler.transform(feature_vector)
            
            # Predict liquidity zones (multi-class: none, upper, lower, both)
            prediction = self.liquidity_model.predict(scaled_features)[0]
            
            # Get probabilities
            probabilities = self.liquidity_model.predict_proba(scaled_features)[0]
            confidence = max(probabilities)
            
            liquidity = {
                "upper_liquidity": False,
                "lower_liquidity": False,
                "confidence": confidence,
                "zones": []
            }
            
            # Enhanced with actual liquidity zone positions
            if prediction == "upper" or prediction == "both":
                liquidity["upper_liquidity"] = True
                # Find actual positions in upper half
                for i, density in enumerate(features["density_regions"][:5]):
                    if density > 0.15:
                        liquidity["zones"].append({
                            "position": i / len(features["density_regions"]),
                            "strength": density,
                            "type": "upper"
                        })
                        
            if prediction == "lower" or prediction == "both":
                liquidity["lower_liquidity"] = True
                # Find actual positions in lower half
                for i, density in enumerate(features["density_regions"][5:]):
                    actual_i = i + 5
                    if density > 0.15:
                        liquidity["zones"].append({
                            "position": actual_i / len(features["density_regions"]),
                            "strength": density,
                            "type": "lower"
                        })
            
            return liquidity
            
        except Exception as e:
            print(f"Error in ML liquidity analysis: {e}")
            return self.analyze_liquidity(features)  # Fallback
    
    def analyze_liquidity(self, features):
        """Traditional rule-based liquidity analysis"""
        liquidity = {
            "upper_liquidity": False,
            "lower_liquidity": False,
            "confidence": 0.0,
            "zones": []
        }
        
        upper_wick = features.get("upper_wick_density", 0)
        lower_wick = features.get("lower_wick_density", 0)
        
        if upper_wick > 0.15:
            liquidity["upper_liquidity"] = True
            liquidity["confidence"] += min(upper_wick * 2, 0.6)
            liquidity["zones"].append({
                "position": 0.2,  # 20% from top
                "strength": upper_wick,
                "type": "upper"
            })
            
        if lower_wick > 0.15:
            liquidity["lower_liquidity"] = True
            liquidity["confidence"] += min(lower_wick * 2, 0.6)
            liquidity["zones"].append({
                "position": 0.8,  # 80% from top (20% from bottom)
                "strength": lower_wick,
                "type": "lower"
            })
        
        liquidity["confidence"] = min(liquidity["confidence"], 0.95)
        return liquidity
    
    def _refresh_market_data(self, ticker=None):
        """Fetch latest market data from public APIs for specified ticker"""
        if not self.use_market_data:
            return {}
            
        # Use provided ticker or default
        ticker = ticker or self.default_ticker
            
        # Check if we need to refresh (max once per hour)
        if (datetime.now() - self.market_data_timestamp).total_seconds() < 3600 and ticker in self.cached_market_data:
            return self.cached_market_data[ticker]
            
        try:
            # Fetch market data for the specified ticker
            r = requests.get(f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1h&apikey=demo")
            data = r.json()
            
            if "values" in data:
                ticker_data = {
                    "ticker": ticker,
                    "last_price": float(data["values"][0]["close"]),
                    "daily_change": float(data["values"][0]["close"]) - float(data["values"][1]["close"]),
                    "daily_range": float(data["values"][0]["high"]) - float(data["values"][0]["low"]),
                    "volume": float(data["values"][0]["volume"]),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Update cache for this ticker
                if not isinstance(self.cached_market_data, dict):
                    self.cached_market_data = {}
                self.cached_market_data[ticker] = ticker_data
                
                # Cache the data to file
                cache_file = os.path.join(self.cache_dir, f"{ticker}_market_data.json")
                with open(cache_file, 'w') as f:
                    json.dump(ticker_data, f)
                    
                self.market_data_timestamp = datetime.now()
                print(f"Market data for {ticker} refreshed successfully")
                return ticker_data
            else:
                # Load from cache if API fails
                return self._load_cached_market_data(ticker)
                
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            # Fallback to cached data
            return self._load_cached_market_data(ticker)
            
        return {}
    
    def _load_cached_market_data(self, ticker=None):
        """Load market data from cache for specified ticker"""
        ticker = ticker or self.default_ticker
        try:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_market_data.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    ticker_data = json.load(f)
                    if not isinstance(self.cached_market_data, dict):
                        self.cached_market_data = {}
                    self.cached_market_data[ticker] = ticker_data
                print(f"Loaded market data for {ticker} from cache")
                return ticker_data
            else:
                print(f"No cached data found for {ticker}")
                return {}
        except Exception as e:
            print(f"Error loading cached market data for {ticker}: {e}")
            return {}
    
    def analyze_candlestick_patterns(self, features):
        """Enhanced pattern analysis with ML validation"""
        basic_analysis = {
            "trend": "neutral",
            "confidence": 0.5,
            "liquidity_zones": [],
            "key_levels": [],
            "pattern": "neutral_pattern",
            "description": "Neutral market conditions with no clear directional bias.",
            "recommendation": "HOLD"
        }
        
        # Basic trend analysis
        gr_ratio = features["green_ratio"] - features["red_ratio"]
        if abs(gr_ratio) > 0.3:
            basic_analysis["trend"] = "bullish" if gr_ratio > 0 else "bearish"
            basic_analysis["confidence"] = min(0.5 + abs(gr_ratio) * 0.8, 0.95)
            
            # Set basic pattern info based on trend
            if basic_analysis["trend"] == "bullish":
                basic_analysis["pattern"] = "bullish_trend"
                basic_analysis["description"] = "Strong bullish momentum detected in recent price action."
                basic_analysis["recommendation"] = "BUY"
            else:
                basic_analysis["pattern"] = "bearish_trend"
                basic_analysis["description"] = "Strong bearish momentum detected in recent price action."
                basic_analysis["recommendation"] = "SELL"
        
        # Try ML-based pattern recognition if model exists
        if self.pattern_model is not None:
            try:
                feature_vector = np.array(features["vector"]).reshape(1, -1)
                
                # Check if scaler is fitted before using it
                if not hasattr(self.scaler, 'n_features_in_'):
                    self.print_formatted("Scaler not fitted yet, using basic pattern analysis", message_type="warning", save_to_file=False)
                else:
                    scaled_features = self.scaler.transform(feature_vector)
                    
                    # Predict success probability
                    success_prob = self.pattern_model.predict(scaled_features)[0]
                    
                    # Incorporate prediction into confidence
                    basic_analysis["ml_confidence"] = success_prob
                    basic_analysis["confidence"] = (basic_analysis["confidence"] + success_prob) / 2
                    basic_analysis["confidence"] = min(basic_analysis["confidence"], 0.95)
            except Exception as e:
                self.print_formatted(f"Pattern analysis continuing with basic approach", message_type="info", save_to_file=False)
        
        # Horizontal line analysis for support/resistance
        if "horizontal_lines" in features and features["horizontal_lines"]:
            top_third = [line for line in features["horizontal_lines"] if line < 0.33]
            bottom_third = [line for line in features["horizontal_lines"] if line > 0.66]
            middle_third = [line for line in features["horizontal_lines"] 
                           if line >= 0.33 and line <= 0.66]
                           
            if len(top_third) > 0:
                basic_analysis["key_levels"].append("resistance")
            if len(bottom_third) > 0:
                basic_analysis["key_levels"].append("support")
            if len(middle_third) > 0:
                basic_analysis["key_levels"].append("mid-range")
        
        return basic_analysis
    
    def generate_llm_insight(self, analysis, features, market_data=None, user_context=None):
        """Generate advanced trading insights using LLM"""
        if not self.use_llm or not self.llm:
            return ""
        
        # Get ticker and timeframe from analysis
        ticker = analysis.get('ticker', self.default_ticker)
        timeframe = analysis.get('timeframe', self.default_timeframe)
        
        # Create an enhanced prompt with all available information
        market_info = ""
        if market_data and len(market_data) > 0:
            market_info = f"""
Current Market Data for {ticker}:
- Price: {market_data.get('last_price', 'N/A')}
- Daily Change: {market_data.get('daily_change', 'N/A')}
- Daily Range: {market_data.get('daily_range', 'N/A')}
- Updated: {market_data.get('timestamp', 'N/A')}
"""
        
        liquidity_info = ""
        if analysis.get('liquidity_details') and analysis['liquidity_details'].get('zones'):
            zones = analysis['liquidity_details']['zones']
            liquidity_info = "Specific Liquidity Zones:\n"
            for zone in zones:
                liquidity_info += f"- {zone['type'].upper()} zone at {int(zone['position']*100)}% level (strength: {zone['strength']:.2f})\n"
        
        # Add user context if provided
        user_input = ""
        if user_context:
            user_input = f"\nTrader's Analysis/Context:\n{user_context}\n"
        
        prompt = f"""As an expert trading advisor for {ticker} on the {timeframe} timeframe, analyze this market situation:

Technical Analysis:
- Trend: {analysis.get('trend', 'neutral').upper()} 
- Confidence: {analysis.get('confidence', 0.5)*100:.1f}%
- Pattern Recognition Confidence: {analysis.get('ml_confidence', 0)*100:.1f}%
- Liquidity Zones: {', '.join(analysis.get('liquidity_zones', [])) or 'None detected'}
- Key Levels: {', '.join(analysis.get('key_levels', [])) or 'None detected'}

{liquidity_info}

Chart Indicators:
- Green/Red Ratio: {features.get('green_ratio', 0):.2f}/{features.get('red_ratio', 0):.2f}
- Volume Indicator: {features.get('volume_indicator', 0):.2f}
- Horizontal Support/Resistance Lines: {features.get('horizontal_line_count', 0)}

{market_info}{user_input}

Provide professional trading advice for {ticker} on the {timeframe} timeframe in 3-4 sentences including:
1. Primary directional bias with reasoning
2. Key entry and exit levels
3. Risk management
"""
        
        try:
            response = self.llm(prompt, max_length=300, num_return_sequences=1)
            # Extract just the generated part (removing the prompt)
            generated_text = response[0]['generated_text']
            if "Provide professional trading advice" in generated_text:
                return generated_text.split("Provide professional trading advice")[1].strip()
            return generated_text.split("Risk management")[-1].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""
    
    def analyze_image(self, image_path, timeframe=None, ticker=None, user_context=None, extra_tips=False, json_output=False):
        """Complete analysis pipeline with learning capability
        
        Args:
            image_path: Path to the chart image
            timeframe: Trading timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            ticker: Trading instrument symbol (e.g., 'NQ', 'ES', 'BTC', 'AAPL')
            user_context: Optional user-provided analysis or context
            extra_tips: Whether to include enhanced trading tips with entry points and predictions
            json_output: Whether to output analysis in JSON format for web applications
        """
        try:
            # Use provided values or defaults
            timeframe = timeframe or self.default_timeframe
            ticker = ticker or self.default_ticker
            
            # Log the analysis start with improved formatting
            self.print_formatted(f"Analysis of {ticker} ({timeframe}) Chart", message_type="header", prefix=ticker)
            self.print_formatted(f"Chart Image: {image_path}", message_type="info", prefix=ticker)
            
            # Extract features from image
            self.print_formatted("Extracting features from chart image...", message_type="info", prefix=ticker)
            features = self.extract_features_from_image(image_path)
            if "error" in features:
                error_msg = f"Error: {features['error']}"
                self.print_formatted(error_msg, message_type="error", prefix=ticker)
                return error_msg
            
            self.print_formatted("Feature extraction complete", message_type="success", prefix=ticker)
            
            # Get image hash for tracking
            image_hash = self._compute_image_hash(image_path)
            
            # Use ML model for liquidity analysis
            self.print_formatted("Analyzing liquidity zones...", message_type="info", prefix=ticker)
            liquidity_analysis = self.analyze_liquidity_with_ml(features)
            
            # Print liquidity analysis results
            liquidity_zone_type = "none"
            if liquidity_analysis["upper_liquidity"] and liquidity_analysis["lower_liquidity"]:
                liquidity_zone_type = "both"
                self.print_formatted("Detected BOTH upper and lower liquidity zones", message_type="liquidity", prefix=ticker)
            elif liquidity_analysis["upper_liquidity"]:
                liquidity_zone_type = "upper"
                self.print_formatted("Detected UPPER liquidity zone", message_type="liquidity", prefix=ticker)
            elif liquidity_analysis["lower_liquidity"]:
                liquidity_zone_type = "lower"
                self.print_formatted("Detected LOWER liquidity zone", message_type="liquidity", prefix=ticker)
            else:
                self.print_formatted("No significant liquidity zones detected", message_type="info", prefix=ticker)
            
            # Print liquidity confidence
            self.print_formatted(f"Liquidity analysis confidence: {liquidity_analysis['confidence']*100:.1f}%", 
                               message_type="info", prefix=ticker)
            
            # Store for learning
            self.learning_data["image_features"].append(features["vector"])
            self.learning_data["liquidity_zones"].append(liquidity_zone_type)
            self.learning_data["timeframes"].append(timeframe)
            self.learning_data["tickers"].append(ticker)
            
            # Success metrics will be updated when outcome is provided
            self.learning_data["success_metrics"].append(None)
            self._save_learning_data()
            
            # Get pattern analysis
            self.print_formatted("Analyzing candlestick patterns...", message_type="info", prefix=ticker)
            try:
                pattern_analysis = self.analyze_candlestick_patterns(features)
            except Exception as e:
                self.print_formatted(f"Analysis error: {str(e)}", message_type="error", prefix=ticker)
                # Create a basic pattern analysis to ensure we can continue
                pattern_analysis = {
                    "trend": "neutral",
                    "confidence": 0.5,
                    "liquidity_zones": [],
                    "key_levels": [],
                    "pattern": "neutral_pattern",
                    "description": "Neutral market conditions with no clear directional bias.",
                    "recommendation": "HOLD",
                    "ml_confidence": 0.0
                }
            
            # Print pattern analysis results
            try:
                pattern_name = pattern_analysis["pattern"].replace("_", " ").title()
                if pattern_analysis["trend"] == "bullish":
                    self.print_formatted(f"Detected {pattern_name} pattern", message_type="bullish", prefix=ticker)
                elif pattern_analysis["trend"] == "bearish":
                    self.print_formatted(f"Detected {pattern_name} pattern", message_type="bearish", prefix=ticker)
                else:
                    self.print_formatted(f"Detected {pattern_name} pattern", message_type="neutral", prefix=ticker)
                    
                self.print_formatted(f"Pattern description: {pattern_analysis['description']}", message_type="info", prefix=ticker)
            except Exception as e:
                # Don't show error to user, just log it internally
                print(f"Error displaying pattern details: {e}")
                # Ensure pattern_analysis has required fields
                if "pattern" not in pattern_analysis:
                    pattern_analysis["pattern"] = "neutral_pattern"
                if "description" not in pattern_analysis:
                    pattern_analysis["description"] = "Market analysis completed with limited pattern recognition."
                if "trend" not in pattern_analysis:
                    pattern_analysis["trend"] = "neutral"
                if "recommendation" not in pattern_analysis:
                    pattern_analysis["recommendation"] = "HOLD"
                if "liquidity_zones" not in pattern_analysis:
                    pattern_analysis["liquidity_zones"] = []
                if "key_levels" not in pattern_analysis:
                    pattern_analysis["key_levels"] = []
                if "confidence" not in pattern_analysis:
                    pattern_analysis["confidence"] = 0.5
                    
                # Print a neutral pattern message to ensure user sees something
                self.print_formatted(f"Market analysis completed", message_type="neutral", prefix=ticker)
            
            # Add timeframe and ticker information to analysis
            pattern_analysis["timeframe"] = timeframe
            pattern_analysis["ticker"] = ticker
            
            # Add liquidity details to analysis
            pattern_analysis["liquidity_details"] = liquidity_analysis
            
            # Add liquidity zones to the main list
            for zone in liquidity_analysis["zones"]:
                zone_type = zone["type"]
                if zone_type not in pattern_analysis["liquidity_zones"]:
                    pattern_analysis["liquidity_zones"].append(zone_type)
            
            # Get market data if enabled
            market_data = self._refresh_market_data(ticker) if self.use_market_data else {}
            
            # Generate analysis with LLM
            if self.use_llm:
                self.print_formatted("Generating advanced trading insights with LLM...", 
                                   message_type="info", prefix=ticker)
                llm_insight = self.generate_llm_insight(pattern_analysis, features, market_data, user_context)
            else:
                llm_insight = ""
            
            # Generate final report
            analysis = self._generate_report(pattern_analysis, features, market_data, timeframe, ticker)
            
            # Print trade signal with appropriate styling
            self.print_trade_signal(
                signal_type=pattern_analysis["trend"],
                confidence=pattern_analysis["confidence"],
                details=[
                    f"Pattern: {pattern_analysis['pattern'].replace('_', ' ').title()}",
                    f"Recommendation: {pattern_analysis.get('recommendation', 'HOLD').upper()}"
                ],
                save_to_file=True
            )
            
            # Print LLM insights if available
            if llm_insight:
                self.print_analysis_section("Advanced Trading Insight", llm_insight, save_to_file=True)
                analysis += f"\n\n## Advanced Trading Insight\n{llm_insight}"
            
            # Add user context if provided
            if user_context:
                self.print_analysis_section("User-Provided Context", user_context, save_to_file=True)
                analysis += f"\n\n## User-Provided Context\n{user_context}"
            
            # Add learning status
            training_status = f"Last model training: {self.learning_data['last_training']}" if self.learning_data['last_training'] else "Model not yet trained"
            
            learning_status_items = [
                f"Training examples: {len(self.learning_data['image_features'])}",
                training_status
            ]
            
            self.print_analysis_section("Learning Status", learning_status_items, save_to_file=True)
            analysis += f"\n\n## Learning Status\n- Training examples: {len(self.learning_data['image_features'])}\n- {training_status}"
            
            # Save to history with image hash
            self.print_formatted("Saving analysis to trade history...", message_type="info", prefix=ticker)
            self._save_to_history(
                source=image_path,
                trend=pattern_analysis["trend"],
                liquidity_zones=",".join(pattern_analysis["liquidity_zones"]),
                confidence=pattern_analysis["confidence"],
                image_hash=image_hash,
                timeframe=timeframe,
                ticker=ticker
            )
            
            # Periodically retrain models if we have new data
            if len(self.learning_data["image_features"]) % 5 == 0 and len(self.learning_data["image_features"]) > 10:
                self.print_formatted("Scheduling model retraining in background...", message_type="warning", prefix=ticker)
                threading.Thread(target=self._retrain_models).start()
                threading.Thread(target=self._retrain_pattern_model).start()
            
            # Add enhanced trading tips if requested
            if extra_tips:
                self.print_formatted("Generating enhanced trading tips...", message_type="info", prefix=ticker)
                
                # Create enhanced tips section
                enhanced_tips = "\n\n## Enhanced Trading Tips"
                
                # Entry points based on pattern analysis
                enhanced_tips += "\n### Recommended Entry Points"
                if pattern_analysis["trend"] == "bullish":
                    # Find support level for entry
                    entry_level = "recent support levels"
                    if "support" in pattern_analysis["key_levels"] and liquidity_analysis.get("zones"):
                        lower_zones = [z for z in liquidity_analysis["zones"] if z["type"] == "lower"]
                        if lower_zones:
                            entry_level = f"around {int(lower_zones[0]['position']*100)}% chart level"
                    
                    enhanced_tips += f"\n- **BUY Entry**: Look for pullbacks to {entry_level}"
                    enhanced_tips += "\n- **Entry Timing**: Wait for price consolidation or bullish reversal candle pattern"
                    enhanced_tips += f"\n- **Stop Loss**: Place below {entry_level} with sufficient buffer to avoid noise"
                    
                elif pattern_analysis["trend"] == "bearish":
                    # Find resistance level for entry
                    entry_level = "recent resistance levels"
                    if "resistance" in pattern_analysis["key_levels"] and liquidity_analysis.get("zones"):
                        upper_zones = [z for z in liquidity_analysis["zones"] if z["type"] == "upper"]
                        if upper_zones:
                            entry_level = f"around {int(upper_zones[0]['position']*100)}% chart level"
                    
                    enhanced_tips += f"\n- **SELL Entry**: Look for rallies to {entry_level}"
                    enhanced_tips += "\n- **Entry Timing**: Wait for price rejection or bearish reversal candle pattern"
                    enhanced_tips += f"\n- **Stop Loss**: Place above {entry_level} with sufficient buffer to avoid noise"
                else:
                    enhanced_tips += "\n- **Range Trading**: Look for entries near range boundaries"
                    enhanced_tips += "\n- **Breakout Setup**: Wait for clear breakout with volume confirmation"
                
                # Market prediction
                enhanced_tips += "\n\n### Market Prediction"
                if pattern_analysis["trend"] == "bullish":
                    enhanced_tips += f"\n- **Short-term Outlook**: Bullish momentum likely to continue with {pattern_analysis['confidence']*100:.1f}% confidence"
                    enhanced_tips += "\n- **Potential Targets**: Previous swing highs and upper liquidity zones"
                    enhanced_tips += "\n- **Reversal Warning Signs**: Watch for bearish divergence, overbought conditions, or rejection at upper liquidity zones"
                elif pattern_analysis["trend"] == "bearish":
                    enhanced_tips += f"\n- **Short-term Outlook**: Bearish momentum likely to continue with {pattern_analysis['confidence']*100:.1f}% confidence"
                    enhanced_tips += "\n- **Potential Targets**: Previous swing lows and lower liquidity zones"
                    enhanced_tips += "\n- **Reversal Warning Signs**: Watch for bullish divergence, oversold conditions, or support at lower liquidity zones"
                else:
                    enhanced_tips += "\n- **Short-term Outlook**: Consolidation likely to continue until breakout occurs"
                    enhanced_tips += "\n- **Breakout Direction**: Monitor volume and price action for clues on breakout direction"
                
                # Support and resistance levels
                enhanced_tips += "\n\n### Key Support & Resistance Levels"
                if "support" in pattern_analysis["key_levels"] and liquidity_analysis.get("zones"):
                    lower_zones = [z for z in liquidity_analysis["zones"] if z["type"] == "lower"]
                    if lower_zones:
                        enhanced_tips += f"\n- **Support**: Strong support at {int(lower_zones[0]['position']*100)}% chart level (strength: {lower_zones[0]['strength']:.2f})"
                else:
                    enhanced_tips += "\n- **Support**: No significant support levels detected"
                    
                if "resistance" in pattern_analysis["key_levels"] and liquidity_analysis.get("zones"):
                    upper_zones = [z for z in liquidity_analysis["zones"] if z["type"] == "upper"]
                    if upper_zones:
                        enhanced_tips += f"\n- **Resistance**: Strong resistance at {int(upper_zones[0]['position']*100)}% chart level (strength: {upper_zones[0]['strength']:.2f})"
                else:
                    enhanced_tips += "\n- **Resistance**: No significant resistance levels detected"
                
                # Add enhanced tips to analysis
                analysis += enhanced_tips
                
                # Print enhanced tips
                self.print_analysis_section("Enhanced Trading Tips", [
                    ("Entry points and predictions generated based on advanced analysis", "info")
                ])
                
                # Print entry points
                if pattern_analysis["trend"] == "bullish":
                    self.print_formatted("Recommended BUY Entry Points:", message_type="bullish", prefix=ticker)
                    entry_level = "recent support levels"
                    if "support" in pattern_analysis["key_levels"] and liquidity_analysis.get("zones"):
                        lower_zones = [z for z in liquidity_analysis["zones"] if z["type"] == "lower"]
                        if lower_zones:
                            entry_level = f"around {int(lower_zones[0]['position']*100)}% chart level"
                    self.print_formatted(f"  • Look for pullbacks to {entry_level}", message_type="bullish", prefix=ticker)
                    self.print_formatted(f"  • Wait for price consolidation or bullish reversal candle pattern", message_type="bullish", prefix=ticker)
                elif pattern_analysis["trend"] == "bearish":
                    self.print_formatted("Recommended SELL Entry Points:", message_type="bearish", prefix=ticker)
                    entry_level = "recent resistance levels"
                    if "resistance" in pattern_analysis["key_levels"] and liquidity_analysis.get("zones"):
                        upper_zones = [z for z in liquidity_analysis["zones"] if z["type"] == "upper"]
                        if upper_zones:
                            entry_level = f"around {int(upper_zones[0]['position']*100)}% chart level"
                    self.print_formatted(f"  • Look for rallies to {entry_level}", message_type="bearish", prefix=ticker)
                    self.print_formatted(f"  • Wait for price rejection or bearish reversal candle pattern", message_type="bearish", prefix=ticker)
            
            # Save the complete analysis to a dedicated file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_filename = os.path.join(self.output_dir, f"{ticker}_{timeframe}_analysis_{timestamp}.txt")
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                f.write(analysis)
            
            # Generate JSON output if requested
            if json_output:
                try:
                    # Create JSON structure with all analysis data
                    json_data = {
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_hash": image_hash,
                        "analysis": {
                            "trend": pattern_analysis["trend"],
                            "confidence": pattern_analysis["confidence"],
                            "pattern": pattern_analysis["pattern"],
                            "recommendation": pattern_analysis["recommendation"],
                            "description": pattern_analysis["description"],
                            "key_levels": pattern_analysis["key_levels"],
                        },
                        "liquidity": {
                            "zones": liquidity_analysis.get("zones", []),
                            "confidence": liquidity_analysis.get("confidence", 0)
                        },
                        "files": {
                            "analysis": analysis_filename
                        }
                    }
                    
                    # Add enhanced tips if available
                    if extra_tips:
                        json_data["enhanced_tips"] = {
                            "entry_points": {
                                "direction": pattern_analysis["recommendation"],
                                "level": entry_level if 'entry_level' in locals() else "not specified",
                                "confidence": pattern_analysis["confidence"]
                            },
                            "support_resistance": {
                                "support": [int(z["position"]*100) for z in liquidity_analysis.get("zones", []) if z["type"] == "lower"],
                                "resistance": [int(z["position"]*100) for z in liquidity_analysis.get("zones", []) if z["type"] == "upper"]
                            },
                            "market_prediction": pattern_analysis["trend"]
                        }
                    
                    # Save JSON output
                    json_filename = os.path.join(self.output_dir, f"{ticker}_{timeframe}_analysis_{timestamp}.json")
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)
                    
                    self.print_formatted(f"JSON output saved to: {json_filename}", message_type="success", prefix=ticker)
                    json_data["files"]["json"] = json_filename
                except Exception as e:
                    self.print_formatted(f"Error creating JSON output: {str(e)}", message_type="error", prefix=ticker)
            
            self.print_formatted(f"Analysis Complete", message_type="header", prefix=ticker)
            self.print_formatted(f"Complete analysis saved to: {analysis_filename}", message_type="success", prefix=ticker)
            self.print_formatted(f"Image Hash: {image_hash}", message_type="info", prefix=ticker)
            self.print_formatted(f"Use this hash to record trade outcomes for learning:", message_type="info", prefix=ticker)
            self.print_formatted(f"python nqbot.py --record_outcome {image_hash} <score>", message_type="info", prefix=ticker)
            
            # Return JSON data if requested, otherwise return text analysis
            if json_output and 'json_data' in locals():
                return json_data
            return analysis
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.print_formatted(error_msg, message_type="error", prefix=ticker)
            self.print_formatted("Analysis failed to complete", message_type="header", prefix=ticker)
            return error_msg
    
    def _generate_report(self, analysis, features, market_data=None, timeframe=None, ticker=None):
        """Generate enhanced trading report with ML insights"""
        # Use values from analysis if not provided directly
        timeframe = timeframe or analysis.get('timeframe', self.default_timeframe)
        ticker = ticker or analysis.get('ticker', self.default_ticker)
        
        # Determine trend color and message type
        trend_color = Colors.GREEN if analysis['trend'] == 'bullish' else Colors.RED if analysis['trend'] == 'bearish' else Colors.BLUE
        trend_type = "bullish" if analysis['trend'] == 'bullish' else "bearish" if analysis['trend'] == 'bearish' else "neutral"
        
        # Create report with colorized sections for terminal output
        title = f"# {ticker} Trading Analysis Report ({timeframe} Timeframe)"
        self.print_formatted(title, color=Colors.CYAN, bold=True)
        
        self.print_formatted("## Market Assessment", color=Colors.BRIGHT_BLUE, bold=True)
        self.print_formatted(f"- Trend: {analysis['trend'].upper()} ({analysis['confidence']*100:.1f}% confidence)", 
                           color=trend_color, bold=True)
        self.print_formatted(f"- Liquidity Zones: {', '.join(analysis['liquidity_zones']) or 'None detected'}", 
                           color=Colors.MAGENTA)
        self.print_formatted(f"- Key Levels: {', '.join(analysis['key_levels']) or 'None detected'}", 
                           color=Colors.YELLOW)
        
        self.print_formatted("\n## Liquidity Analysis", color=Colors.BRIGHT_BLUE, bold=True)
        
        # Enhanced liquidity reporting
        if analysis['liquidity_details'] and analysis['liquidity_details']['zones']:
            zones = analysis['liquidity_details']['zones']
            self.print_formatted("Detected liquidity zones:", color=Colors.WHITE)
            for zone in zones:
                zone_color = Colors.RED if zone['type'] == 'upper' else Colors.GREEN if zone['type'] == 'lower' else Colors.WHITE
                self.print_formatted(f"- {zone['type'].upper()} zone at {int(zone['position']*100)}% chart level (strength: {zone['strength']:.2f})", 
                                   color=zone_color)
        else:
            self.print_formatted("No significant liquidity zones detected.", color=Colors.WHITE)
            
        self.print_formatted("\n## Price Action Observations", color=Colors.BRIGHT_BLUE, bold=True)
        if features['green_ratio'] > 0.7:
            self.print_formatted("- Strong bullish momentum with consecutive green candles", color=Colors.GREEN)
        elif features['red_ratio'] > 0.7:
            self.print_formatted("- Strong bearish pressure with consecutive red candles", color=Colors.RED)
            
        if features.get('horizontal_line_count', 0) > 3:
            self.print_formatted("- Multiple horizontal support/resistance levels detected", color=Colors.YELLOW)
            
        if features.get('volume_indicator', 0) > 0.7:
            self.print_formatted("- High volume activity detected", color=Colors.BRIGHT_MAGENTA)
        elif features.get('volume_indicator', 0) < 0.3:
            self.print_formatted("- Low volume conditions observed", color=Colors.BRIGHT_BLACK)
            
        # Add market data if available
        if market_data and len(market_data) > 0:
            self.print_formatted("\n## Current Market Data", color=Colors.BRIGHT_BLUE, bold=True)
            self.print_formatted(f"- {ticker} Price: {market_data.get('last_price', 'N/A')}", color=Colors.CYAN)
            
            daily_change = market_data.get('daily_change', 0)
            change_color = Colors.GREEN if daily_change > 0 else Colors.RED if daily_change < 0 else Colors.WHITE
            self.print_formatted(f"- Daily Change: {daily_change}", color=change_color)
            
            self.print_formatted(f"- Daily Range: {market_data.get('daily_range', 'N/A')}", color=Colors.WHITE)
            self.print_formatted(f"- Last Updated: {market_data.get('timestamp', 'N/A')}", color=Colors.BRIGHT_BLACK)
            
        self.print_formatted("\n## Trading Plan", color=Colors.BRIGHT_BLUE, bold=True)
        if analysis['trend'] == 'bullish':
            self.print_formatted("Consider long positions with:", color=Colors.GREEN, bold=True)
            entry_text = "- Entry: Pullbacks to support levels"
            
            # Add specific support level if detected
            if 'support' in analysis['key_levels'] and analysis['liquidity_details'].get('zones'):
                lower_zones = [z for z in analysis['liquidity_details']['zones'] if z['type'] == 'lower']
                if lower_zones:
                    zone_pos = int(lower_zones[0]['position'] * 100)
                    entry_text += f" (around {zone_pos}% chart level)"
            
            self.print_formatted(entry_text, color=Colors.GREEN)
            
            stop_text = "- Stop Loss: Below recent swing lows"
            if 'lower' in analysis['liquidity_zones']:
                stop_text += " and below lower liquidity zone"
            self.print_formatted(stop_text, color=Colors.RED)
            
            target_text = "- Target: Previous highs"
            if 'upper' in analysis['liquidity_zones']:
                target_text += " and upper liquidity zone"
            self.print_formatted(target_text, color=Colors.BRIGHT_GREEN)
                
        elif analysis['trend'] == 'bearish':
            self.print_formatted("Consider short positions with:", color=Colors.RED, bold=True)
            entry_text = "- Entry: Rallies to resistance levels"
            self.print_formatted(entry_text, color=Colors.RED)
        
        # Create a plain text version of the report for returning to the caller
        report = f"""# {ticker} Trading Analysis Report ({timeframe} Timeframe)
## Market Assessment
- **Trend**: {analysis['trend'].upper()} ({analysis['confidence']*100:.1f}% confidence)
- **Liquidity Zones**: {', '.join(analysis['liquidity_zones']) or 'None detected'}
- **Key Levels**: {', '.join(analysis['key_levels']) or 'None detected'}

## Liquidity Analysis"""
        
        # Enhanced liquidity reporting for plain text
        if analysis['liquidity_details'] and analysis['liquidity_details']['zones']:
            zones = analysis['liquidity_details']['zones']
            report += "\nDetected liquidity zones:"
            for zone in zones:
                report += f"\n- {zone['type'].upper()} zone at {int(zone['position']*100)}% chart level (strength: {zone['strength']:.2f})"
        else:
            report += "\nNo significant liquidity zones detected."
            
        report += "\n\n## Price Action Observations"
        if features['green_ratio'] > 0.7:
            report += "\n- Strong bullish momentum with consecutive green candles"
        elif features['red_ratio'] > 0.7:
            report += "\n- Strong bearish pressure with consecutive red candles"
            
        if features.get('horizontal_line_count', 0) > 3:
            report += "\n- Multiple horizontal support/resistance levels detected"
            
        if features.get('volume_indicator', 0) > 0.7:
            report += "\n- High volume activity detected"
        elif features.get('volume_indicator', 0) < 0.3:
            report += "\n- Low volume conditions observed"
            
        # Add market data if available
        if market_data and len(market_data) > 0:
            report += "\n\n## Current Market Data"
            report += f"\n- {ticker} Price: {market_data.get('last_price', 'N/A')}"
            report += f"\n- Daily Change: {market_data.get('daily_change', 'N/A')}"
            report += f"\n- Daily Range: {market_data.get('daily_range', 'N/A')}"
            report += f"\n- Last Updated: {market_data.get('timestamp', 'N/A')}"
            
        report += "\n\n## Trading Plan\n"
        if analysis['trend'] == 'bullish':
            report += "Consider long positions with:\n"
            report += "- Entry: Pullbacks to support levels"
            
            # Add specific support level if detected
            if 'support' in analysis['key_levels'] and analysis['liquidity_details'].get('zones'):
                lower_zones = [z for z in analysis['liquidity_details']['zones'] if z['type'] == 'lower']
                if lower_zones:
                    zone_pos = int(lower_zones[0]['position'] * 100)
                    report += f" (around {zone_pos}% chart level)"
                    
            report += "\n- Stop Loss: Below recent swing lows"
            if 'lower' in analysis['liquidity_zones']:
                report += " and below lower liquidity zone"
                
            report += "\n- Target: Previous highs"
            if 'upper' in analysis['liquidity_zones']:
                report += " and upper liquidity zone"
                
        elif analysis['trend'] == 'bearish':
            report += "Consider short positions with:\n"
            report += "- Entry: Rallies to resistance levels"
            
            # Add specific resistance level if detected
            if 'resistance' in analysis['key_levels'] and analysis['liquidity_details'].get('zones'):
                upper_zones = [z for z in analysis['liquidity_details']['zones'] if z['type'] == 'upper']
                if upper_zones:
                    zone_pos = int(upper_zones[0]['position'] * 100)
                    report += f" (around {zone_pos}% chart level)"
                    
            report += "\n- Stop Loss: Above recent swing highs"
            if 'upper' in analysis['liquidity_zones']:
                report += " and above upper liquidity zone"
                
            report += "\n- Target: Previous lows"
            if 'lower' in analysis['liquidity_zones']:
                report += " and lower liquidity zone"
        else:
            report += "Market in consolidation. Wait for breakout:\n"
            report += "- Long above recent highs with volume confirmation\n"
            report += "- Short below recent lows with momentum"
            
        report += "\n\n## Risk Management"
        report += "\n- Use 1:2 risk-reward ratio minimum"
        report += "\n- Position size 1-2% of capital per trade"
        
        # Add confidence-based risk adjustment
        if analysis['confidence'] < 0.6:
            report += "\n- Consider reducing position size by 50% due to lower confidence"
        elif analysis['confidence'] > 0.8:
            report += "\n- High-confidence setup with strong signal quality"
            
        return report
    
    def _save_to_history(self, **kwargs):
        """Save analysis to history CSV"""
        entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': '',
            'success_rate': '',
            **kwargs
        }
        
        df = pd.read_csv(self.history_file) if os.path.exists(self.history_file) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.history_file, index=False)
        
    def visualize_liquidity_zones(self, image_path, save_path=None, timeframe=None, ticker=None, extra_tips=False):
        """Visualize detected liquidity zones on the chart with optional enhanced visualization
        
        Args:
            image_path: Path to the chart image
            save_path: Path to save the visualization
            timeframe: Trading timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            ticker: Trading instrument symbol (e.g., 'NQ', 'ES', 'BTC', 'AAPL')
            extra_tips: Whether to include enhanced visualization with entry points, predictions, etc.
        """
        try:
            # Use provided values or defaults
            timeframe = timeframe or self.default_timeframe
            ticker = ticker or self.default_ticker
            
            # Extract features and analyze
            self.print_formatted("Generating enhanced visualization...", message_type="info", save_to_file=False)
            features = self.extract_features_from_image(image_path)
            if "error" in features:
                self.print_formatted(f"Error in feature extraction: {features['error']}", message_type="warning", save_to_file=False)
                # Create a basic visualization anyway
                liquidity_analysis = {"zones": [], "confidence": 0.5}
                pattern_analysis = {
                    "trend": "neutral",
                    "confidence": 0.5,
                    "liquidity_zones": [],
                    "key_levels": [],
                    "pattern": "neutral_pattern",
                    "description": "Neutral market conditions with no clear directional bias.",
                    "recommendation": "HOLD"
                }
            else:
                try:
                    liquidity_analysis = self.analyze_liquidity_with_ml(features)
                    pattern_analysis = self.analyze_candlestick_patterns(features)
                except Exception as e:
                    self.print_formatted(f"Using fallback analysis", message_type="info", save_to_file=False)
                    # Use rule-based analysis as fallback
                    liquidity_analysis = self.analyze_liquidity(features)
                    pattern_analysis = {
                        "trend": "neutral",
                        "confidence": 0.5,
                        "liquidity_zones": [],
                        "key_levels": [],
                        "pattern": "neutral_pattern",
                        "description": "Neutral market conditions with no clear directional bias.",
                        "recommendation": "HOLD"
                    }
            
            # Load original image
            img = Image.open(image_path)
            width, height = img.size
            
            # Create figure
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(img))
            
            # Plot liquidity zones
            zones_found = False
            for zone in liquidity_analysis.get('zones', []):
                y_pos = int(zone['position'] * height)
                zone_type = zone['type']
                strength = zone['strength']
                
                color = 'r' if zone_type == 'upper' else 'g'
                alpha = min(0.8, strength + 0.3)
                line_width = min(5, strength * 10)
                
                plt.axhline(y=y_pos, color=color, alpha=alpha, linewidth=line_width)
                plt.text(width * 0.05, y_pos - 10, 
                        f"{zone_type.upper()} LIQUIDITY ({strength:.2f})", 
                        color=color, fontsize=12, fontweight='bold')
                zones_found = True
            
            # Enhanced visualization with extra tips
            if extra_tips:
                # Add support and resistance lines
                if 'support' in pattern_analysis['key_levels']:
                    # Find approximate support level (lower third of chart)
                    support_y = int(height * 0.75)  # Default position if not found
                    lower_zones = [z for z in liquidity_analysis.get('zones', []) if z['type'] == 'lower']
                    if lower_zones:
                        support_y = int(lower_zones[0]['position'] * height)
                    
                    # Draw support line
                    plt.axhline(y=support_y, color='lime', alpha=0.6, linewidth=2, linestyle='--')
                    plt.text(width * 0.75, support_y - 15, "SUPPORT", color='lime', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7))
                
                if 'resistance' in pattern_analysis['key_levels']:
                    # Find approximate resistance level (upper third of chart)
                    resistance_y = int(height * 0.25)  # Default position if not found
                    upper_zones = [z for z in liquidity_analysis.get('zones', []) if z['type'] == 'upper']
                    if upper_zones:
                        resistance_y = int(upper_zones[0]['position'] * height)
                    
                    # Draw resistance line
                    plt.axhline(y=resistance_y, color='red', alpha=0.6, linewidth=2, linestyle='--')
                    plt.text(width * 0.75, resistance_y - 15, "RESISTANCE", color='red', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7))
                
                # Add trend direction arrow
                if pattern_analysis['trend'] == 'bullish':
                    # Draw bullish arrow
                    arrow_start_x = width * 0.85
                    arrow_start_y = height * 0.7
                    arrow_length = height * 0.2
                    plt.arrow(arrow_start_x, arrow_start_y, 0, -arrow_length, 
                             head_width=width*0.03, head_length=height*0.05, 
                             fc='lime', ec='lime', alpha=0.8)
                    plt.text(arrow_start_x - width*0.05, arrow_start_y - arrow_length/2, 
                            "BULLISH\nTREND", color='lime', fontsize=12, fontweight='bold',
                            ha='right', va='center', bbox=dict(facecolor='black', alpha=0.7))
                    
                elif pattern_analysis['trend'] == 'bearish':
                    # Draw bearish arrow
                    arrow_start_x = width * 0.85
                    arrow_start_y = height * 0.3
                    arrow_length = height * 0.2
                    plt.arrow(arrow_start_x, arrow_start_y, 0, arrow_length, 
                             head_width=width*0.03, head_length=height*0.05, 
                             fc='red', ec='red', alpha=0.8)
                    plt.text(arrow_start_x - width*0.05, arrow_start_y + arrow_length/2, 
                            "BEARISH\nTREND", color='red', fontsize=12, fontweight='bold',
                            ha='right', va='center', bbox=dict(facecolor='black', alpha=0.7))
                
                # Add entry point marker
                if pattern_analysis['recommendation'] == 'BUY':
                    # Find entry point (typically near support)
                    entry_y = int(height * 0.7)  # Default position
                    if 'support' in pattern_analysis['key_levels']:
                        lower_zones = [z for z in liquidity_analysis.get('zones', []) if z['type'] == 'lower']
                        if lower_zones:
                            entry_y = int(lower_zones[0]['position'] * height)
                    
                    # Draw entry point marker
                    plt.scatter(width * 0.2, entry_y, color='lime', s=150, marker='^', alpha=0.8)
                    plt.text(width * 0.2 + 20, entry_y, "ENTRY POINT", color='lime', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7))
                    
                elif pattern_analysis['recommendation'] == 'SELL':
                    # Find entry point (typically near resistance)
                    entry_y = int(height * 0.3)  # Default position
                    if 'resistance' in pattern_analysis['key_levels']:
                        upper_zones = [z for z in liquidity_analysis.get('zones', []) if z['type'] == 'upper']
                        if upper_zones:
                            entry_y = int(upper_zones[0]['position'] * height)
                    
                    # Draw entry point marker
                    plt.scatter(width * 0.2, entry_y, color='red', s=150, marker='v', alpha=0.8)
                    plt.text(width * 0.2 + 20, entry_y, "ENTRY POINT", color='red', fontsize=12, fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.7))
                
                # Add reversal zone if applicable
                if 'upper' in pattern_analysis['liquidity_zones'] and pattern_analysis['trend'] == 'bullish':
                    # Potential reversal zone at upper liquidity
                    upper_zones = [z for z in liquidity_analysis.get('zones', []) if z['type'] == 'upper']
                    if upper_zones:
                        reversal_y = int(upper_zones[0]['position'] * height)
                        plt.axhspan(reversal_y-10, reversal_y+10, color='yellow', alpha=0.3)
                        plt.text(width * 0.4, reversal_y, "POTENTIAL REVERSAL ZONE", color='yellow', fontsize=12, fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.7))
                
                elif 'lower' in pattern_analysis['liquidity_zones'] and pattern_analysis['trend'] == 'bearish':
                    # Potential reversal zone at lower liquidity
                    lower_zones = [z for z in liquidity_analysis.get('zones', []) if z['type'] == 'lower']
                    if lower_zones:
                        reversal_y = int(lower_zones[0]['position'] * height)
                        plt.axhspan(reversal_y-10, reversal_y+10, color='yellow', alpha=0.3)
                        plt.text(width * 0.4, reversal_y, "POTENTIAL REVERSAL ZONE", color='yellow', fontsize=12, fontweight='bold',
                                bbox=dict(facecolor='black', alpha=0.7))
            
            # If no zones were found, add a message
            if not zones_found and not extra_tips:
                plt.text(width * 0.5, height * 0.5, 
                       "No liquidity zones detected", 
                       color='white', fontsize=14, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.7))
            
            # Add title and info with ticker and timeframe
            title_text = f"{ticker} Trading Assistant - {timeframe} Timeframe"
            if extra_tips:
                title_text += " - Enhanced Analysis"
            else:
                title_text += " - Liquidity Analysis"
            plt.title(title_text, fontsize=14)
            
            plt.text(width * 0.05, height * 0.05, 
                   f"Confidence: {pattern_analysis.get('confidence', 0)*100:.1f}%", 
                   color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
            
            # Add timeframe and ticker info
            plt.text(width * 0.05, height * 0.95, 
                   f"Instrument: {ticker} | Timeframe: {timeframe}", 
                   color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
            
            # Remove axes
            plt.axis('off')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                self.print_formatted(f"Enhanced visualization saved to: {save_path}", message_type="success", save_to_file=True)
                return save_path
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                return buf
                
        except Exception as e:
            self.print_formatted(f"Visualization completed with limited features: {str(e)}", message_type="info", save_to_file=True)
            # If visualization fails, at least return a path so the process can continue
            if save_path:
                return save_path
            return None
            
    def generate_trade_report(self, start_date=None, end_date=None, ticker=None, timeframe=None):
        """Generate performance report based on trade history
        
        Args:
            start_date: Start date for filtering trades (YYYY-MM-DD)
            end_date: End date for filtering trades (YYYY-MM-DD)
            ticker: Filter by specific trading instrument
            timeframe: Filter by specific timeframe
        """
        try:
            df = pd.read_csv(self.history_file)
            if len(df) == 0:
                return "No trade history found."
                
            # Filter by date if provided
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
                
            # Filter by ticker and timeframe if provided
            if ticker and 'ticker' in df.columns:
                df = df[df['ticker'] == ticker]
            if timeframe and 'timeframe' in df.columns:
                df = df[df['timeframe'] == timeframe]
                
            # Check if we have data after filtering
            if len(df) == 0:
                return "No trade history found matching the specified filters."
                
            # Calculate performance metrics
            completed_trades = df[df['outcome'] != '']
            if len(completed_trades) == 0:
                return "No completed trades found in the specified period."
                
            success_count = len(completed_trades[completed_trades['outcome'].astype(float) > 0.5])
            win_rate = success_count / len(completed_trades) * 100
            
            # Calculate average confidence
            avg_confidence = df['confidence'].astype(float).mean() * 100
            
            # Analyze by trend
            trend_analysis = df.groupby('trend').agg({
                'timestamp': 'count',
                'confidence': lambda x: np.mean(x.astype(float)) * 100
            }).rename(columns={'timestamp': 'count'})
            
            # Generate report title based on filters
            report_title = "Trading Performance Report"
            if ticker and timeframe:
                report_title = f"{ticker} {timeframe} Trading Performance Report"
            elif ticker:
                report_title = f"{ticker} Trading Performance Report"
            elif timeframe:
                report_title = f"{timeframe} Timeframe Trading Performance Report"
            
            # Generate report
            report = f"""# {report_title}
## Overview
- **Period**: {df['timestamp'].min()} to {df['timestamp'].max()}
- **Total Analyses**: {len(df)}
- **Completed Trades**: {len(completed_trades)}
- **Success Rate**: {win_rate:.1f}%
- **Average Confidence**: {avg_confidence:.1f}%
"""

            # Add ticker and timeframe breakdown if available and not filtered
            if 'ticker' in df.columns and not ticker:
                ticker_analysis = df.groupby('ticker').agg({
                    'timestamp': 'count',
                    'outcome': lambda x: (x.astype(float) > 0.5).mean() * 100 if len(x.dropna()) > 0 else float('nan')
                }).rename(columns={'timestamp': 'count', 'outcome': 'win_rate'})
                
                if not ticker_analysis.empty:
                    report += "\n## Instrument Analysis\n"
                    for ticker_name, data in ticker_analysis.iterrows():
                        win_rate_str = f"{data['win_rate']:.1f}% win rate" if not pd.isna(data['win_rate']) else "no completed trades"
                        report += f"- **{ticker_name}**: {data['count']} analyses ({win_rate_str})\n"
            
            if 'timeframe' in df.columns and not timeframe:
                timeframe_analysis = df.groupby('timeframe').agg({
                    'timestamp': 'count',
                    'outcome': lambda x: (x.astype(float) > 0.5).mean() * 100 if len(x.dropna()) > 0 else float('nan')
                }).rename(columns={'timestamp': 'count', 'outcome': 'win_rate'})
                
                if not timeframe_analysis.empty:
                    report += "\n## Timeframe Analysis\n"
                    for tf_name, data in timeframe_analysis.iterrows():
                        win_rate_str = f"{data['win_rate']:.1f}% win rate" if not pd.isna(data['win_rate']) else "no completed trades"
                        report += f"- **{tf_name}**: {data['count']} analyses ({win_rate_str})\n"
            
            # Add trend analysis
            report += "\n## Trend Analysis\n"
            for trend, data in trend_analysis.iterrows():
                report += f"- **{trend.capitalize()}**: {data['count']} analyses ({data['confidence']:.1f}% confidence)\n"
                
            # Add liquidity zone effectiveness if we have enough data
            if len(completed_trades) >= 5:
                report += "\n## Liquidity Zone Effectiveness\n"
                for zone_type in ['upper', 'lower', 'upper,lower', 'lower,upper']:
                    zone_trades = completed_trades[completed_trades['liquidity_zones'].str.contains(zone_type, na=False)]
                    if len(zone_trades) > 0:
                        zone_success = len(zone_trades[zone_trades['outcome'].astype(float) > 0.5]) / len(zone_trades) * 100
                        report += f"- **{zone_type.replace(',', '+')}**: {zone_success:.1f}% success rate ({len(zone_trades)} trades)\n"
            
            return report
            
        except Exception as e:
            return f"Error generating report: {e}"

def main():
    parser = argparse.ArgumentParser(description='Enhanced Trading Assistant')
    parser.add_argument('--image', help='Path to market chart image')
    parser.add_argument('--ticker', default='NQ', help='Trading instrument symbol (e.g., NQ, ES, BTC, AAPL)')
    parser.add_argument('--timeframe', default='1h', help='Chart timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--use_llm', action='store_true', help='Enable LLM insights')
    parser.add_argument('--llm_model', default='gpt2', help='LLM model to use (if available)')
    parser.add_argument('--use_market_data', action='store_true', help='Enable real-time market data')
    parser.add_argument('--visualize', action='store_true', help='Generate liquidity visualization')
    parser.add_argument('--extra_tips', action='store_true', help='Enable enhanced trading tips with entry points, predictions, and visualizations')
    parser.add_argument('--json_output', action='store_true', help='Output analysis in JSON format for web applications')
    parser.add_argument('--user_context', help='User-provided analysis or context')
    parser.add_argument('--record_outcome', nargs=2, metavar=('IMAGE_HASH', 'SCORE'), 
                        help='Record trade outcome (0.0-1.0) for learning')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--start_date', help='Start date for report (YYYY-MM-DD)')
    parser.add_argument('--end_date', help='End date for report (YYYY-MM-DD)')
    
    args = parser.parse_args()

    if not args.json_output:
        print(f"{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}║             NQ TRADING ASSISTANT v2.0                    ║{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}╚══════════════════════════════════════════════════════════╝{Colors.RESET}")
    assistant = NQTradingAssistant(use_llm=args.use_llm, use_market_data=args.use_market_data, 
                                  llm_model=args.llm_model, default_ticker=args.ticker,
                                  default_timeframe=args.timeframe, json_output_enabled=args.json_output)
    
    if args.record_outcome:
        image_hash, score = args.record_outcome
        success = assistant.learn_from_outcome(image_hash, float(score))
        if success:
            assistant.print_formatted(f"Outcome recorded successfully. Models will be updated with this data.", 
                                    color=Colors.GREEN, bold=True)
        else:
            assistant.print_formatted("Failed to record outcome. Make sure the image hash is correct.", 
                                    color=Colors.RED, bold=True)
    
    elif args.report:
        assistant.print_formatted(f"Generating performance report for {args.ticker} ({args.timeframe})...", 
                                color=Colors.CYAN, bold=True)
        report = assistant.generate_trade_report(
            start_date=args.start_date, 
            end_date=args.end_date,
            ticker=args.ticker,
            timeframe=args.timeframe
        )
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = os.path.join(assistant.output_dir, f"{args.ticker}_{args.timeframe}_report_{timestamp}.txt")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        assistant.print_formatted(f"\nComplete report saved to: {report_filename}", 
                               color=Colors.BRIGHT_GREEN, bold=True)
        
        # JSON output if requested
        if args.json_output:
            try:
                # Convert report to JSON format
                report_data = {
                    "ticker": args.ticker,
                    "timeframe": args.timeframe,
                    "report_type": "performance",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "content": report,
                    "file_path": report_filename
                }
                
                # Save JSON report
                json_filename = os.path.join(assistant.output_dir, f"{args.ticker}_{args.timeframe}_report_{timestamp}.json")
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2)
                assistant.print_formatted(f"JSON report saved to: {json_filename}", 
                                       color=Colors.BRIGHT_GREEN, bold=True)
            except Exception as e:
                assistant.print_formatted(f"Error creating JSON output: {str(e)}", 
                                       message_type="error", save_to_file=True)
        
        # Print report with some formatting
        lines = report.split('\n')
        for line in lines:
            if line.startswith('#'):
                assistant.print_formatted(line, color=Colors.CYAN, bold=True)
            elif line.startswith('##'):
                assistant.print_formatted(line, color=Colors.BRIGHT_BLUE, bold=True)
            elif 'bullish' in line.lower():
                assistant.print_formatted(line, color=Colors.GREEN)
            elif 'bearish' in line.lower():
                assistant.print_formatted(line, color=Colors.RED)
            elif '%' in line and any(x in line.lower() for x in ['success', 'win', 'rate']):
                # Extract percentage value to determine color
                try:
                    pct = float(line.split('%')[0].split(':')[-1].strip())
                    color = Colors.GREEN if pct >= 50 else Colors.RED
                    assistant.print_formatted(line, color=color, bold=True)
                except:
                    assistant.print_formatted(line, color=Colors.WHITE)
            else:
                assistant.print_formatted(line, color=Colors.WHITE)
        
    elif args.image:
        # Analysis is handled by analyze_image method
        analysis_content = assistant.analyze_image(
            args.image, 
            timeframe=args.timeframe, 
            ticker=args.ticker,
            user_context=args.user_context,
            extra_tips=args.extra_tips,
            json_output=args.json_output
        )

        if args.json_output:
            if isinstance(analysis_content, dict):
                print(json.dumps(analysis_content, indent=2))
            elif isinstance(analysis_content, str):
                # Attempt to parse if it looks like a JSON string, otherwise print if it's meant to be raw output
                try:
                    json.loads(analysis_content) # Validate if it's JSON
                    print(analysis_content)
                except json.JSONDecodeError:
                    # If it's not a valid JSON string but json_output was true, this indicates an issue
                    # For now, we won't print non-JSON string in json_output mode to keep stdout clean.
                    # Alternatively, print an error JSON: print(json.dumps({"error": "analyze_image returned non-JSON string in json_output mode"}))
                    pass 
            # If analysis_content is None, app.py's error handling for no JSON output will take effect.
        
        if args.visualize:
            # Create visualization with proper error handling
            save_path = os.path.join(assistant.output_dir, f"{args.ticker}_{args.timeframe}_viz_{int(time.time())}.png")
            try:
                viz_result = assistant.visualize_liquidity_zones(
                    args.image, 
                    save_path,
                    timeframe=args.timeframe,
                    ticker=args.ticker,
                    extra_tips=args.extra_tips
                )
            except Exception as e:
                if not args.json_output:
                    assistant.print_formatted(f"Error during visualization: {str(e)}", 
                                            message_type="error", save_to_file=True)
                # If json_output is true, this error won't be printed to stdout.
                # The main JSON output from analyze_image would have already been printed.
                # We can still log "Visualization completed" if successful and not in JSON mode.
                if not args.json_output: # Or just let visualize_liquidity_zones handle its own prints
                    assistant.print_formatted("Visualization attempt finished.", 
                                            message_type="info", save_to_file=True)



if __name__ == "__main__":
    main()