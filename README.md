# Urgent Appointments Forecast

Streamlit app for rerunning the saved N-BEATS urgent appointments model.

## Files

Keep these model artifacts in `files/`:

- `nbeats_model.pt`
- `nbeats_model.pt.ckpt`
- `nbeats_scaler.pkl`

Darts saves the model metadata and the learned PyTorch weights separately. If
`nbeats_model.pt.ckpt` is missing, the app can load the model definition but
cannot run `predict()`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Sheet persistence

The app reads and writes persistent data from Google Sheets.

The default Google Sheet target is:

https://docs.google.com/spreadsheets/d/1oZrb2bkqMDwuVyPjk-yurL45htbNU_O3lJBgjjVUf5k/edit?usp=sharing

The app writes:

- forecast log data to a worksheet named `nbeats_forecast_log`
- actuals time-series data to a worksheet named `ts`
- calendar exclusion data to a worksheet named `nbeats_calendar`

### Google Sheet setup

The Streamlit public Google Sheet guide supports reading a public sheet with
`st.connection` and `st-gsheets-connection`. Public links are read-only for the
app. To save `nbeats_forecast_log` back to Google Sheets, use
`st-gsheets-connection` CRUD mode:

1. Create a Google Cloud service account.
2. Enable the Google Drive API and Google Sheets API.
3. Share the spreadsheet with the service account `client_email`.
4. Add the service account fields to `.streamlit/secrets.toml`.
