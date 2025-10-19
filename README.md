# Pattern Trading Toolkit

This project provides utilities for analysing price patterns using Dynamic Time Warping (DTW) and
k-nearest neighbour aggregation. It is written in Python and is intended for educational and
exploratory trading research.

## 1. Downloading the project

1. Click the green **Code** button on the GitHub page.
2. Choose **Download ZIP** and save the file to your computer.
3. Right-click the downloaded ZIP file and select **Extract All...** to unpack it. Remember the
   folder where you extract the files (for example, `Downloads/pattern-trading-toolkit`).

## 2. Setting up Python (one-time)

1. Install Python 3.10 or newer from [python.org/downloads](https://www.python.org/downloads/).
2. During installation, tick the option **Add Python to PATH** if it appears.
3. After installation finishes, open a terminal:
   * **Windows**: press `Win + R`, type `cmd`, and press Enter.
   * **macOS**: open **Terminal** from Spotlight.
   * **Linux**: open your preferred terminal application.

## 3. Creating an isolated environment

1. In the terminal, navigate to the folder that contains the extracted project. Example for
   Windows (replace the path with your actual folder):
   ```bash
   cd %HOMEPATH%\Downloads\pattern-trading-toolkit
   ```
   macOS / Linux example:
   ```bash
   cd ~/Downloads/pattern-trading-toolkit
   ```
2. Create a virtual environment so the dependencies stay isolated:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   * **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   * **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

## 4. Installing the required packages

With the virtual environment active, install the project dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas pytest
```

## 5. Using the pattern trading module

The main functionality lives in `src/pattern_trading.py`. To try it out with your own Excel file:

1. Place your OHLC Excel file (must contain columns like `Date`, `Open`, `High`, `Low`, `Close`)
   inside the project folder.
2. Create a simple Python script named `run_analysis.py` (or open a Python notebook) with the
   following example code. Update the file name and column names if yours differ:
   ```python
   from src.pattern_trading import (
       load_ohlc_from_excel,
       generate_trade_guidance,
       default_visualisation_hook,
   )

   # 1. Load your Excel data (replace with your file name)
   data = load_ohlc_from_excel("my_prices.xlsx", sheet_name=0)

   # 2. Generate guidance using the latest 20 candles
   guidance = generate_trade_guidance(
       data,
       column="Close",
       window=20,
       k=5,
       forecast_horizon=10,
       tp_pct=0.02,
       sl_pct=0.02,
       lookback_years=10,
       visualisation_hook=default_visualisation_hook,
   )

   # 3. Print the results in a readable format
   print("Entry confidence:", f"{guidance.entry_confidence:.0%}")
   print("Expected return:", f"{guidance.expected_return:.2%}")
   print("Take-profit (relative):", f"{guidance.take_profit:.2%}")
   print("Stop-loss (relative):", f"{guidance.stop_loss:.2%}")
   print("Suggested exit horizon (candles):", guidance.exit_horizon)
   print("TP hit probability:", f"{guidance.tp_hit_probability:.0%}")
   print("SL hit probability:", f"{guidance.sl_hit_probability:.0%}")

   # Optional: access individual pattern matches
   for match in guidance.matches:
       print(match)
   ```
3. Run the script from the terminal while the virtual environment is active:
   ```bash
   python run_analysis.py
   ```
4. The output prints the aggregated trade guidance. Adjust the parameters (`window`, `k`,
   `forecast_horizon`, etc.) as you learn what works best for your dataset.

## 6. Running the automated tests

Tests use synthetic data to verify the DTW distances, neighbour selection, and trade guidance
summary logic.

1. Ensure your virtual environment is active (see step 3 above).
2. In the project folder run:
   ```bash
   pytest
   ```
3. All tests should finish with an "OK" message. If any fail, the terminal will show details you can
   share when seeking help.

## 7. Troubleshooting tips

* **`ModuleNotFoundError: No module named 'pandas'`** – confirm that the virtual environment is
  activated and re-run `python -m pip install pandas`.
* **Excel file not found** – verify the path you pass to `load_ohlc_from_excel` matches the file's
  name and location.
* **Date parsing issues** – ensure your Excel sheet has a column named `Date` (or update the
  `date_column` argument) and the values are valid Excel/ISO date formats.

You are now ready to explore pattern-based trading ideas using this toolkit. Happy analysing!
