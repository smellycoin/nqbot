# Enhanced NQ Trading Assistant

A powerful, self-learning trading assistant for analyzing market chart images, detecting liquidity zones, candlestick patterns, and generating advanced trading insights using machine learning and optional LLMs. Supports dynamic learning from user feedback and real-time market data.

## Features
- Chart image analysis for trend, support/resistance, and liquidity zones
- Candlestick pattern recognition
- Machine learning models that improve with user feedback
- Optional LLM (language model) insights for advanced analysis
- Real-time market data integration (optional)
- Trade history and performance reporting
- Visualization of liquidity zones
- Enhanced trading tips with entry points and market predictions
- Support/resistance lines and trend direction indicators
- JSON output for web application integration

## Web Application Integration
The trading assistant now supports integration with web applications through the `--json_output` option. This generates a structured JSON file containing all analysis data, which can be easily consumed by web applications.

### JSON Output Structure
```json
{
  "ticker": "NQ",
  "timeframe": "1h",
  "timestamp": "2023-06-01 12:34:56",
  "image_hash": "abc123...",
  "analysis": {
    "trend": "bullish",
    "confidence": 0.85,
    "pattern": "bullish_trend",
    "recommendation": "BUY",
    "description": "Strong bullish momentum detected...",
    "key_levels": ["support", "resistance"]
  },
  "liquidity": {
    "zones": [
      {"type": "upper", "position": 0.25, "strength": 0.75},
      {"type": "lower", "position": 0.75, "strength": 0.85}
    ],
    "confidence": 0.8
  },
  "enhanced_tips": {
    "entry_points": {
      "direction": "BUY",
      "level": "around 75% chart level",
      "confidence": 0.85
    },
    "support_resistance": {
      "support": [75],
      "resistance": [25]
    },
    "market_prediction": "bullish"
  },
  "files": {
    "analysis": "trading_recommendations/NQ_1h_analysis_20230601_123456.txt",
    "json": "trading_recommendations/NQ_1h_analysis_20230601_123456.json"
  }
}
```

## Installation
1. Clone this repository:
   ```sh
   git clone <repo-url>
   cd nqbot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   - For LLM support, install `transformers` and download a compatible model (e.g., `gpt2`).

## Usage
Run the assistant from the command line:
```sh
python nqbot.py --image path/to/chart.png --ticker NQ --timeframe 1h
```

### Main Arguments
- `--image`: Path to the chart image (required for analysis)
- `--ticker`: Trading instrument symbol (default: NQ)
- `--timeframe`: Chart timeframe (default: 1h)
- `--use_llm`: Enable advanced LLM insights
- `--llm_model`: LLM model name (default: gpt2)
- `--market_data`: Enable real-time market data
- `--visualize`: Generate and save liquidity zone visualization
- `--extra_tips`: Enable enhanced trading tips with entry points, predictions, and visualizations
- `--json_output`: Output analysis in JSON format for web applications
- `--user_context`: Add your own analysis or notes to the report
- `--record_outcome IMAGE_HASH SCORE`: Record trade outcome (0.0-1.0) for learning
- `--report`: Generate a performance report
- `--start_date`/`--end_date`: Filter report by date (YYYY-MM-DD)

### Example Use Cases
#### 1. Basic Chart Analysis
```sh
python nqbot.py --image charts/nq_2024-06-01.png --ticker NQ --timeframe 15m
```

#### 2. Enhanced Analysis with Trading Tips
```sh
python nqbot.py --image charts/nq_2024-06-01.png --ticker NQ --timeframe 15m --extra_tips
```

#### 3. Enhanced Visualization with Entry Points
```sh
python nqbot.py --image charts/nq_2024-06-01.png --ticker NQ --timeframe 15m --visualize --extra_tips
```

#### 4. JSON Output for Web Applications
```sh
python nqbot.py --image charts/nq_2024-06-01.png --ticker NQ --timeframe 15m --extra_tips --json_output
```

#### 5. Advanced Analysis with LLM and Market Data
```sh
python nqbot.py --image charts/es_2024-06-01.png --ticker ES --timeframe 1h --use_llm --llm_model gpt2 --market_data
```

#### 3. Add User Context
```sh
python nqbot.py --image charts/btc_2024-06-01.png --ticker BTC --timeframe 4h --user_context "Watching for breakout above resistance."
```

#### 4. Visualize Liquidity Zones
```sh
python nqbot.py --image charts/aapl_2024-06-01.png --ticker AAPL --timeframe 1d --visualize
```

#### 5. Record Trade Outcome for Learning
After receiving an analysis, note the `image_hash` from the output. Then:
```sh
python nqbot.py --record_outcome <IMAGE_HASH> 0.8
```

#### 6. Generate Performance Report
```sh
python nqbot.py --report --start_date 2024-06-01 --end_date 2024-06-30 --ticker NQ
```

#### 3. Using a Mistral Model (<1GB) for LLM Analysis
To use a quantized or small Mistral model (under 1GB) for LLM-powered analysis, specify the model path or Hugging Face repo with `--llm_model` and enable LLM insights:
```sh
python nqbot.py --image charts/nq_2024-06-01.png --ticker NQ --timeframe 1h --use_llm --llm_model path/to/mistral-quantized
```
- Replace `path/to/mistral-quantized` with your downloaded Mistral model directory or Hugging Face model name (e.g., `mistralai/Mistral-7B-v0.1-GGUF` or a quantized variant).
- For best performance on limited hardware, use a quantized GGUF/GGML or similar version under 1GB.
- Ensure the `transformers` library supports the model format. Some quantized models may require `auto-gptq`, `bitsandbytes`, or `gguf` support.
- If you encounter memory errors, try a smaller or more aggressively quantized model.

See the code and comments in `nqbot.py` for more details on LLM model loading and requirements.
## Notes
- The assistant creates folders for recommendations, models, and cache automatically.
- Models improve as you provide more feedback via `--record_outcome`.
- LLM and market data features are optional but enhance analysis.
- For best results, use clear chart images and provide feedback regularly.

## Requirements
- Python 3.8+
- See `requirements.txt` for full dependency list

## License
MIT