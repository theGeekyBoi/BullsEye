Script 01 — 01_inventory_sources.py
Purpose: Inventory the available raw data sources.

This script examines the project's available datasets and identifies the files that will be used to construct the stock prediction dataset.

It:
- Searches the FNSPID price archive for individual stock ticker CSV files.
- Examines the Yahoo Finance prototype data.
- Identifies available CSV files in the project directories.
- Displays column names and sample records to understand the structure of each dataset.
- Helps identify missing ticker files and differences in data formats.
Why it is important: Before combining or processing financial data, we need to understand what information is available, how it is organized, and whether the sources contain the necessary fields.
Script 02 — 02_audit_yahoo_prototype.py
Purpose: Audit the Yahoo Finance price dataset for completeness and data quality.

This script evaluates the consolidated Yahoo Finance dataset containing the 50 selected companies and SPY.

It checks:
- Whether all expected tickers are present.
- The number of records available for each ticker.
- The earliest and latest dates available.
- Missing values in OHLCV columns.
- Duplicate ticker/date records.
- Invalid price relationships, such as a high price below the low price.
- Whether each ticker has sufficient historical data.
It also generates:

processed_fnspid\yahoo_prototype_audit.csv

Why it is important: This establishes whether the Yahoo Finance dataset is sufficiently complete and consistent to serve as the source for the prediction dataset.
Script 03 — 03_inspect_price_adjustments.py
Purpose: Compare price adjustments between Yahoo Finance and FNSPID.

This script investigates differences between the two available historical price sources.

It:
- Compares overlapping historical dates for selected companies.
- Examines Open, High, Low, Close, and Adjusted Close values.
- Calculates ratios between corresponding Yahoo and FNSPID prices.
- Investigates differences caused by stock splits and dividend adjustments.
- Examines corporate action records to understand why historical prices differ.
For example, it examines AAPL, NVDA, MSFT, SPY, and HD.

The analysis showed that Yahoo's OHLC values appear to be adjusted for historical corporate actions, whereas FNSPID's raw OHLC values are on a different adjustment basis.

Why it is important: Combining price data with inconsistent adjustment conventions could introduce artificial price movements and incorrect return calculations. This script helps establish which source should be used consistently.
Script 04 — 04_validate_yahoo_calendar.py
Purpose: Verify that Yahoo Finance records align with the official NYSE trading calendar.

This script compares the available Yahoo Finance price records against the project's NYSE trading calendar.

It:
- Loads the official XNYS trading calendar.
- Compares expected trading sessions against actual price records.
- Identifies missing trading sessions.
- Identifies unexpected dates that are not valid trading sessions.
- Checks for duplicate records.
- Checks for invalid OHLC relationships.
- Reports the number of expected and observed sessions for each ticker.
The validation confirmed:
- 1,715 NYSE sessions through October 28, 2026.
- 49 tickers with complete coverage across those sessions.
- PLTR with 1,527 sessions because it began trading later.
- No missing or unexpected sessions.
A floating-point precision issue was also identified in one HD record and distinguished from a genuine invalid price.

The output is:

processed_fnspid\yahoo_calendar_validation.csv

Why it is important: Financial datasets must use consistent trading dates. This ensures that the 30-day historical windows and five-session prediction targets are based on actual NYSE trading sessions rather than calendar days.
Script 05 — 05_build_30day_dataset.py
Purpose: Construct the final 30-trading-session stock prediction dataset.

This is the primary dataset-generation script.

It combines the validated price data with the NYSE trading calendar to create one prediction example for each company and eligible prediction date.

For every example, it:
1. Identifies the prediction date \(t\).
2. Retrieves the previous 30 completed NYSE trading sessions, including session \(t\).
3. Extracts Open, High, Low, Close, and Volume for each session.
4. Calculates the company's forward five-session return.
5. Calculates SPY's forward five-session return.
6. Calculates the company's excess return relative to SPY.
7. Saves the resulting dataset and summary statistics.
The target calculations are:

Company return:
\[
R_{\text{company}}=
\frac{Close_{t+5}}{Open_{t+1}}-1
\]

SPY return:
\[
R_{\text{SPY}}=
\frac{Close_{\text{SPY},t+5}}{Open_{\text{SPY},t+1}}-1
\]

Excess return:
\[
R_{\text{excess}}=
R_{\text{company}}-R_{\text{SPY}}
\]

The script produces:
- final_dataset\quant_dataset_30day.csv
- final_dataset\dataset_summary.csv
The resulting dataset contains:
Component	Number
Company tickers	50
Examples	83,862
Historical input sessions	30
Features per session	5
Total input features	150
Target variables	3
Metadata columns	4
Total columns	157

The five input features are Open, High, Low, Close, and Volume.

Why it is important: This script transforms the raw financial data into a structured machine-learning dataset. Each row represents a prediction opportunity using historical information to predict future performance relative to the overall market.
Script 06 — 06_verify_dataset.py
Purpose: Independently verify the correctness of the generated dataset.

This script checks whether the dataset generated by Script 05 accurately represents the original price data.

It:
- Selects sample records from different companies and dates.
- Reconstructs the corresponding 30-session historical windows.
- Compares all 150 input features against the original Yahoo Finance records.
- Independently recalculates company returns.
- Independently recalculates SPY returns.
- Verifies the excess-return calculation.
- Checks that entry and exit dates correspond to the correct trading sessions.
It tests five examples, including an NVDA example that crosses its June 2024 10-for-1 stock split.

The verification achieved:
- 750 out of 750 historical feature comparisons passed.
- 15 out of 15 target calculations matched.
- 5 out of 5 entry and exit date checks passed.
Why it is important: A dataset can contain thousands of rows but still have indexing errors, incorrect target calculations, or mismatched dates. This script independently confirms that the dataset-generation logic is producing the intended results.
Script 07 — 07_audit_split_volume.py
Purpose: Investigate how stock splits affect historical trading volume.

This script examines trading volume around recorded stock split events.

It:
- Identifies non-unit stock split events in the corporate actions dataset.
- Retrieves trading volume from the sessions surrounding each event.
- Calculates median volume before and after each split.
- Compares the volume levels to identify potential adjustments.
- Saves the results for further analysis.
The output is:

processed_fnspid\split_volume_audit.csv

The audit identified 17 non-unit corporate action events.

Why it is important: Stock splits change the number of shares outstanding and the number of shares traded per unit of stock price. Historical volume may therefore need to be adjusted to make volume measurements comparable across different share structures.

This investigation is particularly important because Volume is one of the 150 machine-learning input features.

The audit does not independently prove whether Yahoo's volume values are already split-adjusted. That still needs to be established before applying additional adjustments.
Script 08 — 08_inspect_corporate_actions.py
Purpose: Examine the complete corporate action history for companies with non-unit split events.

This script provides a more detailed examination of the corporate action records identified in Script 07.

It:
- Identifies all non-unit stock split ratios.
- Groups split events by ticker.
- Prints the date and ratio of each event.
- Displays the complete dividend and stock split history for the affected companies.
- Helps distinguish ordinary stock splits from unusual corporate actions.
For example, it identifies:
- AAPL's 4-for-1 split.
- AMZN's 20-for-1 split.
- NVDA's 4-for-1 and 10-for-1 splits.
- TSLA's 5-for-1 and 3-for-1 splits.
- Several unusual GE, IBM, MRK, and RTX corporate action ratios requiring further investigation.
Why it is important: Not every non-unit corporate action ratio should automatically be treated as an ordinary stock split. Understanding the event history helps prevent incorrect adjustments to historical volume.