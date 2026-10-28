# ModelWatch

A dark-first, open-source batch monitoring application for exploring tabular data quality, detecting distribution shifts and evaluating predictions. Demo data is synthetic and deterministic; results are calculated from the selected data.

## Run locally

Use Python 3.10–3.12.

```sh
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Select **Datasets & analysis**, choose synthetic demo or two UTF-8 CSV files, inspect the schema, select features, then run the analysis. Optionally select actual and prediction columns for evaluation. Download JSON, CSV or HTML reports. Uploaded rows are not deliberately saved; summaries are stored in `data/modelwatch.sqlite3`. Summaries include column names and statistics, so treat them as sensitive. Delete history in the app or remove the database.

## Deploy on Streamlit Community Cloud

Push this repository to GitHub, connect that repository in Streamlit Community Cloud, select `app.py` as the entry point, and deploy. This environment is intended for demonstration: local SQLite storage on hosted instances is not guaranteed to survive restarts and is shared across users of that instance. Do not upload sensitive data to a public shared deployment. For private durable multi-user use, add authentication, isolation and managed storage first.

## Method and limitations

Numeric features use two-sample KS tests, categorical features Pearson chi-square. The chosen p-value threshold flags features individually; no multiple-testing correction is applied. Small expected categorical counts are flagged, and features that cannot be analyzed remain unavailable rather than counted as healthy. Statistical drift does not prove that a model is worse. Performance needs actual labels and predictions, with missing pairs excluded and counted. Weighted classification metrics may obscure minority-class behavior. Data are held in memory, limited to 20 MB, 200,000 rows and 500 columns per dataset.

This is an early functional subset of the supplied PRD. Evidently, SQLAlchemy, Parquet, configurable alert rules, comparison of historical runs, feature distribution charts, prediction-specific workflow, baseline training and PDF exports are not implemented. SQLite is used directly. Not intended for high-stakes model validation.

## Tests

```sh
python -m unittest discover -s tests -v
```

MIT licensed; see LICENSE.
