from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from streamlit_gsheets import GSheetsConnection

try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
except ImportError as exc:
    TimeSeries = None
    NBEATSModel = None
    DARTS_IMPORT_ERROR = exc
else:
    DARTS_IMPORT_ERROR = None


APP_DIR = Path(__file__).resolve().parent
FILES_DIR = APP_DIR / "files"
TS_FILE = FILES_DIR / "ts.csv"
FORECAST_LOG_FILE = FILES_DIR / "nbeats_forecast_log.csv"
MODEL_FILE = FILES_DIR / "nbeats_model.pt"
MODEL_CHECKPOINT_FILE = FILES_DIR / "nbeats_model.pt.ckpt"
SCALER_FILE = FILES_DIR / "nbeats_scaler.pkl"
CALENDAR_FILE = FILES_DIR / "nbeats_calendar.pkl"
FORECAST_HORIZON = 5
PL_TRAINER_KWARGS = {
    "accelerator": "cpu",
    "devices": 1,
    "logger": False,
    "enable_checkpointing": False,
}
GOOGLE_SHEET_ID = "1oZrb2bkqMDwuVyPjk-yurL45htbNU_O3lJBgjjVUf5k"
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    f"{GOOGLE_SHEET_ID}/edit?usp=sharing"
)
GOOGLE_WORKSHEET_NAME = "nbeats_forecast_log"
GOOGLE_TS_WORKSHEET_NAME = "ts"
GOOGLE_CALENDAR_WORKSHEET_NAME = "nbeats_calendar"

st.set_page_config(
    page_title="Urgent Appointments Forecast",
    layout="wide",
)


def require_files() -> list[str]:
    required = [
        MODEL_FILE,
        MODEL_CHECKPOINT_FILE,
        SCALER_FILE,
    ]
    return [str(path.relative_to(APP_DIR)) for path in required if not path.exists()]


def get_google_sheet_connection() -> GSheetsConnection:
    return st.connection("gsheets", type=GSheetsConnection)


def refresh_google_sheet_connection() -> None:
    try:
        get_google_sheet_connection().reset()
    except Exception:
        st.cache_data.clear()


def get_service_account_credentials() -> dict[str, Any]:
    config = st.secrets.get("connections", {}).get("gsheets", {})
    excluded_keys = {
        "spreadsheet",
        "spreadsheet_id",
        "worksheet",
        "worksheet_name",
        "ts_worksheet",
        "ts_worksheet_name",
        "calendar_worksheet",
        "calendar_worksheet_name",
    }
    return {key: value for key, value in config.items() if key not in excluded_keys}


def get_google_spreadsheet_url() -> str:
    return google_connection_secret("spreadsheet") or GOOGLE_SHEET_URL


def update_worksheet_raw(
    worksheet_name: str,
    data: pd.DataFrame,
    text_columns: list[str] | None = None,
) -> None:
    client = gspread.service_account_from_dict(get_service_account_credentials())
    spreadsheet = client.open_by_url(get_google_spreadsheet_url())

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name,
            rows=max(len(data) + 1, 1000),
            cols=max(len(data.columns), 1),
        )

    worksheet.clear()
    if text_columns:
        for column in text_columns:
            column_index = data.columns.get_loc(column) + 1
            column_letter = _column_index_to_letter(column_index)
            worksheet.format(
                f"{column_letter}:{column_letter}",
                {"numberFormat": {"type": "TEXT"}},
            )

    values = [data.columns.tolist()] + data.fillna("").astype(str).values.tolist()
    worksheet.update(values=values, range_name="A1", raw=True)


def _column_index_to_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


@st.cache_resource(show_spinner=False)
def load_model_assets() -> tuple[Any, Any]:
    if DARTS_IMPORT_ERROR is not None:
        raise ImportError(
            "The `darts` package is required to load the N-BEATS model. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from DARTS_IMPORT_ERROR

    model = NBEATSModel.load(
        str(MODEL_FILE),
        pl_trainer_kwargs=PL_TRAINER_KWARGS,
    )
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


def read_ts() -> pd.DataFrame:
    conn = get_google_sheet_connection()
    ts = conn.read(
        spreadsheet=GOOGLE_SHEET_URL,
        worksheet=google_ts_worksheet_name(),
        ttl=0,
    )
    ts.columns = [str(column).strip() for column in ts.columns]
    if not {"ds", "y"}.issubset(ts.columns):
        raise ValueError("Google Sheet worksheet `ts` must contain `ds` and `y` columns.")

    ts = ts[["ds", "y"]].copy()
    ts["ds"] = pd.to_datetime(ts["ds"], errors="raise")
    ts["y"] = pd.to_numeric(ts["y"], errors="coerce").fillna(0).astype(np.float32)
    return ts.sort_values("ds").reset_index(drop=True)


def read_forecast_log() -> pd.DataFrame:
    conn = get_google_sheet_connection()
    log = conn.read(
        spreadsheet=GOOGLE_SHEET_URL,
        worksheet=google_worksheet_name(),
        ttl=0,
    )
    log.columns = [str(column).strip() for column in log.columns]
    required_columns = ["forecast_date", "target_date", "horizon", "predicted", "actual"]
    if log.empty:
        return pd.DataFrame(
            columns=required_columns
        )

    missing_columns = [column for column in required_columns if column not in log.columns]
    if missing_columns:
        raise ValueError(
            "Google Sheet worksheet "
            f"`{google_worksheet_name()}` is missing columns: {', '.join(missing_columns)}."
        )

    log = log[required_columns].copy()
    log["forecast_date"] = pd.to_datetime(log["forecast_date"], errors="raise")
    log["target_date"] = pd.to_datetime(log["target_date"], errors="raise")
    log["horizon"] = pd.to_numeric(log["horizon"], errors="coerce")
    for column in ["predicted", "actual"]:
        log[column] = pd.to_numeric(log[column], errors="coerce")
    return log


def read_calendar() -> pd.DataFrame:
    conn = get_google_sheet_connection()
    calendar = conn.read(
        spreadsheet=GOOGLE_SHEET_URL,
        worksheet=google_calendar_worksheet_name(),
        ttl=0,
    )
    calendar.columns = [str(column).strip() for column in calendar.columns]
    if not {"ds", "holiday"}.issubset(calendar.columns):
        raise ValueError(
            "Google Sheet worksheet "
            f"`{google_calendar_worksheet_name()}` must contain `ds` and `holiday` columns."
        )

    calendar = calendar[["ds", "holiday"]].copy()
    calendar["ds"] = pd.to_datetime(calendar["ds"], errors="raise")
    calendar["holiday"] = calendar["holiday"].astype(str)
    return calendar.sort_values("ds").reset_index(drop=True)


def save_ts(ts: pd.DataFrame) -> None:
    ok, message = sync_ts_to_google_sheet(ts)
    if not ok:
        raise RuntimeError(message)


def save_forecast_log(log: pd.DataFrame) -> None:
    ok, message = sync_forecast_log_to_google_sheet(log)
    if not ok:
        raise RuntimeError(message)


def save_calendar(calendar: pd.DataFrame) -> None:
    ok, message = sync_calendar_to_google_sheet(calendar)
    if not ok:
        raise RuntimeError(message)


def excluded_dates(calendar: pd.DataFrame) -> set[pd.Timestamp]:
    return set(pd.to_datetime(calendar["ds"]).dt.normalize())


def get_next_working_days(
    last_date: pd.Timestamp,
    n: int,
    excluded: set[pd.Timestamp],
) -> pd.DatetimeIndex:
    future_dates: list[pd.Timestamp] = []
    current_date = pd.Timestamp(last_date)

    while len(future_dates) < n:
        current_date += pd.Timedelta(days=1)
        if current_date.normalize() not in excluded:
            future_dates.append(current_date)

    return pd.DatetimeIndex(future_dates)


def build_working_day_series(
    ts: pd.DataFrame,
    excluded: set[pd.Timestamp],
) -> tuple[pd.DataFrame, Any]:
    if DARTS_IMPORT_ERROR is not None:
        raise ImportError(
            "The `darts` package is required to create the forecasting series."
        ) from DARTS_IMPORT_ERROR

    ts_nbeats = ts[["ds", "y"]].copy()
    ts_nbeats["ds"] = pd.to_datetime(ts_nbeats["ds"])
    ts_nbeats["y"] = pd.to_numeric(ts_nbeats["y"], errors="coerce").fillna(0).astype(
        np.float32
    )
    ts_nbeats = (
        ts_nbeats[~ts_nbeats["ds"].dt.normalize().isin(excluded)]
        .sort_values("ds")
        .reset_index(drop=True)
    )
    ts_nbeats["working_day"] = range(len(ts_nbeats))

    series = TimeSeries.from_dataframe(
        ts_nbeats,
        time_col="working_day",
        value_cols="y",
    ).astype(np.float32)
    return ts_nbeats, series


def run_forecast(ts: pd.DataFrame) -> pd.DataFrame:
    model, scaler = load_model_assets()
    calendar = read_calendar()
    excluded = excluded_dates(calendar)
    ts_nbeats, series = build_working_day_series(ts, excluded)

    if ts_nbeats.empty:
        raise ValueError("No working-day rows remain after applying the calendar.")

    series_scaled = scaler.transform(series)
    forecast_scaled = model.predict(
        n=FORECAST_HORIZON,
        series=series_scaled,
        verbose=False,
    )
    forecast = scaler.inverse_transform(forecast_scaled)

    future_dates = get_next_working_days(
        last_date=ts_nbeats["ds"].max(),
        n=FORECAST_HORIZON,
        excluded=excluded,
    )
    return pd.DataFrame(
        {
            "target_date": future_dates,
            "horizon": range(1, FORECAST_HORIZON + 1),
            "predicted": forecast.values().flatten(),
        }
    )


def upsert_actual(ts: pd.DataFrame, date: pd.Timestamp, urgent_slots: int) -> pd.DataFrame:
    updated = ts.copy()
    normalized = pd.Timestamp(date).normalize()
    matches = updated["ds"].dt.normalize() == normalized

    if matches.any():
        updated.loc[matches, "y"] = urgent_slots
    else:
        updated = pd.concat(
            [
                updated,
                pd.DataFrame({"ds": [normalized], "y": [urgent_slots]}),
            ],
            ignore_index=True,
        )

    return updated.sort_values("ds").reset_index(drop=True)


def append_forecast_log(
    existing_log: pd.DataFrame,
    forecast_date: pd.Timestamp,
    urgent_slots: int,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    log = existing_log.copy()
    forecast_date = pd.Timestamp(forecast_date).normalize()

    if not log.empty:
        target_matches = pd.to_datetime(log["target_date"]).dt.normalize() == forecast_date
        log.loc[target_matches, "actual"] = urgent_slots

    new_rows = forecast.copy()
    new_rows.insert(0, "forecast_date", forecast_date)
    new_rows["actual"] = np.nan

    return pd.concat([log, new_rows], ignore_index=True)


def load_uploaded_rows(uploaded_file: Any) -> pd.DataFrame:
    uploaded = pd.read_csv(uploaded_file)
    uploaded.columns = [col.strip().lower() for col in uploaded.columns]

    if not {"ds", "y"}.issubset(uploaded.columns):
        raise ValueError("CSV must contain `ds` and `y` columns.")

    rows = uploaded[["ds", "y"]].copy()
    rows["ds"] = pd.to_datetime(rows["ds"], errors="raise")
    rows["y"] = pd.to_numeric(rows["y"], errors="raise").round().astype(int)
    return rows.sort_values("ds").reset_index(drop=True)


def google_connection_secret(key: str) -> Any | None:
    try:
        return st.secrets.get("connections", {}).get("gsheets", {}).get(key)
    except Exception:
        return None


def has_google_sheet_write_auth() -> bool:
    return google_connection_secret("type") == "service_account"


def google_sheet_id() -> str:
    return (
        google_connection_secret("spreadsheet_id")
        or os.getenv("GOOGLE_SHEET_ID")
        or GOOGLE_SHEET_ID
    )


def google_worksheet_name() -> str:
    return (
        google_connection_secret("worksheet")
        or google_connection_secret("worksheet_name")
        or os.getenv("GOOGLE_WORKSHEET_NAME")
        or GOOGLE_WORKSHEET_NAME
    )


def google_ts_worksheet_name() -> str:
    return (
        google_connection_secret("ts_worksheet")
        or google_connection_secret("ts_worksheet_name")
        or os.getenv("GOOGLE_TS_WORKSHEET_NAME")
        or GOOGLE_TS_WORKSHEET_NAME
    )


def google_calendar_worksheet_name() -> str:
    return (
        google_connection_secret("calendar_worksheet")
        or google_connection_secret("calendar_worksheet_name")
        or os.getenv("GOOGLE_CALENDAR_WORKSHEET_NAME")
        or GOOGLE_CALENDAR_WORKSHEET_NAME
    )


def format_ts_for_sheet(ts: pd.DataFrame) -> pd.DataFrame:
    sheet_ts = ts.copy()
    sheet_ts["ds"] = pd.to_datetime(sheet_ts["ds"]).dt.strftime("%Y-%m-%d")
    sheet_ts["y"] = pd.to_numeric(sheet_ts["y"], errors="coerce").fillna(0).round().astype(int)
    return sheet_ts[["ds", "y"]]


def format_calendar_for_sheet(calendar: pd.DataFrame) -> pd.DataFrame:
    sheet_calendar = calendar.copy()
    sheet_calendar["ds"] = pd.to_datetime(sheet_calendar["ds"]).dt.strftime("%Y-%m-%d")
    sheet_calendar["holiday"] = sheet_calendar["holiday"].fillna("").astype(str)
    return sheet_calendar[["ds", "holiday"]]


def format_forecast_log_for_sheet(forecast_log: pd.DataFrame) -> pd.DataFrame:
    sheet_log = forecast_log.copy()
    sheet_log["forecast_date"] = pd.to_datetime(sheet_log["forecast_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    sheet_log["target_date"] = pd.to_datetime(sheet_log["target_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    sheet_log["horizon"] = pd.to_numeric(sheet_log["horizon"], errors="coerce")
    sheet_log["predicted"] = pd.to_numeric(sheet_log["predicted"], errors="coerce")
    sheet_log["actual"] = pd.to_numeric(sheet_log["actual"], errors="coerce")
    return sheet_log.replace({np.nan: ""})


def assert_sheet_date_format(sheet_log: pd.DataFrame) -> None:
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    for column in ["forecast_date", "target_date"]:
        invalid = ~sheet_log[column].astype(str).str.match(date_pattern)
        if invalid.any():
            raise ValueError(f"`{column}` must be formatted as YYYY-MM-DD.")


def assert_ts_date_format(sheet_ts: pd.DataFrame) -> None:
    invalid = ~sheet_ts["ds"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    if invalid.any():
        raise ValueError("`ds` must be formatted as YYYY-MM-DD.")


def assert_calendar_date_format(sheet_calendar: pd.DataFrame) -> None:
    invalid = ~sheet_calendar["ds"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    if invalid.any():
        raise ValueError("Calendar `ds` must be formatted as YYYY-MM-DD.")


def sync_forecast_log_to_google_sheet(forecast_log: pd.DataFrame) -> tuple[bool, str]:
    if not has_google_sheet_write_auth():
        return (
            False,
            "Google Sheet write sync needs service-account credentials in `.streamlit/secrets.toml`.",
        )

    worksheet_name = google_worksheet_name()
    sheet_log = format_forecast_log_for_sheet(forecast_log)
    assert_sheet_date_format(sheet_log)
    update_worksheet_raw(
        worksheet_name=worksheet_name,
        data=sheet_log,
        text_columns=["forecast_date", "target_date"],
    )
    return True, f"Synced {len(sheet_log)} row(s) to `{worksheet_name}`."


def sync_ts_to_google_sheet(ts: pd.DataFrame) -> tuple[bool, str]:
    if not has_google_sheet_write_auth():
        return (
            False,
            "Google Sheet write sync needs service-account credentials in `.streamlit/secrets.toml`.",
        )

    conn = st.connection("gsheets", type=GSheetsConnection)
    worksheet_name = google_ts_worksheet_name()
    sheet_ts = format_ts_for_sheet(ts)
    assert_ts_date_format(sheet_ts)
    conn.update(
        spreadsheet=GOOGLE_SHEET_URL,
        worksheet=worksheet_name,
        data=sheet_ts,
    )
    return True, f"Synced {len(sheet_ts)} row(s) to `{worksheet_name}`."


def sync_calendar_to_google_sheet(calendar: pd.DataFrame) -> tuple[bool, str]:
    if not has_google_sheet_write_auth():
        return (
            False,
            "Google Sheet write sync needs service-account credentials in `.streamlit/secrets.toml`.",
        )

    conn = st.connection("gsheets", type=GSheetsConnection)
    worksheet_name = google_calendar_worksheet_name()
    sheet_calendar = format_calendar_for_sheet(calendar)
    assert_calendar_date_format(sheet_calendar)
    conn.update(
        spreadsheet=GOOGLE_SHEET_URL,
        worksheet=worksheet_name,
        data=sheet_calendar,
    )
    return True, f"Synced {len(sheet_calendar)} row(s) to `{worksheet_name}`."


def sync_all_to_google_sheet(
    ts: pd.DataFrame,
    forecast_log: pd.DataFrame,
) -> tuple[bool, str]:
    forecast_ok, forecast_message = sync_forecast_log_to_google_sheet(forecast_log)
    ts_ok, ts_message = sync_ts_to_google_sheet(ts)
    return forecast_ok and ts_ok, f"{forecast_message} {ts_message}"


def calculate_forecast_metrics(forecast_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = forecast_log.copy()
    scored["actual"] = pd.to_numeric(scored["actual"], errors="coerce")
    scored["predicted"] = pd.to_numeric(scored["predicted"], errors="coerce")
    scored = scored.dropna(subset=["actual", "predicted"]).copy()

    if scored.empty:
        return scored, pd.DataFrame()

    scored["error"] = scored["actual"] - scored["predicted"]
    scored["abs_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    scored["pct_error"] = np.where(
        scored["actual"] != 0,
        scored["abs_error"] / scored["actual"].abs(),
        np.nan,
    )

    by_horizon = (
        scored.groupby("horizon", dropna=False)
        .agg(
            rows=("actual", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            mape=("pct_error", lambda values: float(values.mean() * 100)),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    return scored, by_horizon


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


st.title("Urgent Appointments Forecast")

missing_files = require_files()
if missing_files:
    st.error("Missing required files: " + ", ".join(missing_files))
    st.stop()

ts = read_ts()
forecast_log = read_forecast_log()
last_actual_date = ts["ds"].max()

with st.sidebar:
    st.header("Data")
    st.caption(f"Time series source: Google Sheet `{google_ts_worksheet_name()}`")
    st.caption(f"Forecast log source: Google Sheet `{google_worksheet_name()}`")
    st.divider()
    st.header("Google Sheet")
    st.link_button("Open target sheet", GOOGLE_SHEET_URL)
    st.caption(f"Forecast worksheet: `{google_worksheet_name()}`")
    st.caption(f"Actuals worksheet: `{google_ts_worksheet_name()}`")
    st.caption(f"Calendar worksheet: `{google_calendar_worksheet_name()}`")
    can_sync_google_sheet = has_google_sheet_write_auth()
    if can_sync_google_sheet:
        st.success("Streamlit GSheets service account configured")
    else:
        st.info("Public sheet links are read-only. Add service account fields to `[connections.gsheets]` to save.")

top_cols = st.columns(4)
with top_cols[0]:
    metric_card("Latest actual date", last_actual_date.strftime("%Y-%m-%d"))
with top_cols[1]:
    metric_card("Latest used slots", f"{int(ts.iloc[-1]['y']):,}")
with top_cols[2]:
    metric_card("Rows in ts.csv", f"{len(ts):,}")
with top_cols[3]:
    metric_card("Forecast log rows", f"{len(forecast_log):,}")

tab_single, tab_upload, tab_history, tab_metrics = st.tabs(
    ["Run prediction", "Upload actuals", "History", "Metrics"]
)

with tab_single:
    st.subheader("Add one day and forecast the next 5 working days")
    with st.form("single_day_form"):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            input_date = st.date_input(
                "Date",
                value=(last_actual_date + pd.Timedelta(days=1)).date(),
            )
        with col_b:
            urgent_slots = st.number_input(
                "Urgent used slots",
                min_value=0,
                max_value=1000,
                step=1,
                value=0,
            )
        sync_sheet = st.checkbox(
            "Save updates to Google Sheet",
            value=can_sync_google_sheet,
            disabled=not can_sync_google_sheet,
        )
        if not can_sync_google_sheet:
            st.caption("Google sync is disabled until service-account credentials are configured.")
        submitted = st.form_submit_button("Save actual and run forecast", type="primary")

    if submitted:
        selected_date = pd.Timestamp(input_date)
        calendar = read_calendar()
        if selected_date.normalize() in excluded_dates(calendar):
            st.warning(
                "This date is excluded from the model calendar, so it will be saved "
                "to ts.csv but not used as a working-day model observation."
            )

        updated_ts = upsert_actual(ts, selected_date, int(urgent_slots))
        forecast = run_forecast(updated_ts)
        updated_log = append_forecast_log(
            forecast_log,
            selected_date,
            int(urgent_slots),
            forecast,
        )

        if sync_sheet:
            try:
                ok, message = sync_all_to_google_sheet(updated_ts, updated_log)
                if ok:
                    st.success(f"Saved to Google Sheets. {message}")
                else:
                    st.info(message)
            except Exception as exc:
                st.error(f"Google Sheet sync failed: {exc}")
                st.stop()
        else:
            st.warning("Changes were calculated but not saved because Google Sheet sync is disabled.")

        st.success("Appended a new 5 working-day forecast.")
        st.dataframe(
            forecast.assign(target_date=forecast["target_date"].dt.strftime("%Y-%m-%d")),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download updated forecast log",
            format_forecast_log_for_sheet(updated_log).to_csv(index=False).encode("utf-8"),
            file_name="nbeats_forecast_log.csv",
            mime="text/csv",
        )

with tab_upload:
    st.subheader("Upload one or more actual rows")
    st.caption("Upload a CSV with `ds` and `y` columns. The last uploaded date drives the forecast run.")
    uploaded_file = st.file_uploader("Actuals CSV", type=["csv"])
    sync_uploaded_sheet = st.checkbox(
        "Save updates to Google Sheet after upload",
        value=can_sync_google_sheet,
        disabled=not can_sync_google_sheet,
    )
    run_upload = st.button("Save uploaded actuals and run forecast", type="primary")

    if uploaded_file and run_upload:
        try:
            uploaded_rows = load_uploaded_rows(uploaded_file)
            updated_ts = ts.copy()
            updated_log = forecast_log.copy()

            for row in uploaded_rows.itertuples(index=False):
                updated_ts = upsert_actual(updated_ts, row.ds, int(row.y))
                updated_log = append_forecast_log(
                    updated_log,
                    pd.Timestamp(row.ds),
                    int(row.y),
                    pd.DataFrame(
                        columns=["target_date", "horizon", "predicted"]
                    ),
                )

            forecast = run_forecast(updated_ts)
            final_date = pd.Timestamp(uploaded_rows["ds"].max())
            updated_log = pd.concat(
                [
                    updated_log,
                    forecast.assign(forecast_date=final_date, actual=np.nan)[
                        ["forecast_date", "target_date", "horizon", "predicted", "actual"]
                    ],
                ],
                ignore_index=True,
            )

            if sync_uploaded_sheet:
                ok, message = sync_all_to_google_sheet(updated_ts, updated_log)
                if ok:
                    st.success(f"Saved to Google Sheets. {message}")
                else:
                    st.info(message)
            else:
                st.warning("Changes were calculated but not saved because Google Sheet sync is disabled.")

            st.success(f"Processed {len(uploaded_rows)} actual row(s) and appended a forecast.")
            st.dataframe(
                forecast.assign(target_date=forecast["target_date"].dt.strftime("%Y-%m-%d")),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

with tab_history:
    st.subheader("Recent actuals")
    recent_actuals = ts[ts["ds"] >= pd.Timestamp("2026-01-01")].copy()
    st.line_chart(recent_actuals.set_index("ds")["y"])
    st.dataframe(recent_actuals, width="stretch", hide_index=True)

    forecast_header, refresh_col = st.columns([1, 0.25], vertical_alignment="center")
    with forecast_header:
        st.subheader("Forecast log")
    with refresh_col:
        if st.button("Refresh", key="refresh_forecast_log"):
            refresh_google_sheet_connection()
            st.rerun()

    st.dataframe(
        forecast_log.sort_values(["forecast_date", "horizon"], ascending=[False, True]),
        width="stretch",
        hide_index=True,
    )

with tab_metrics:
    st.subheader("Forecast Accuracy")
    scored_log, metrics_by_horizon = calculate_forecast_metrics(forecast_log)

    if scored_log.empty:
        st.info("No completed forecast rows yet. Metrics need rows with both `predicted` and `actual` values.")
    else:
        overall_mae = scored_log["abs_error"].mean()
        overall_rmse = float(np.sqrt(scored_log["squared_error"].mean()))
        overall_mape = scored_log["pct_error"].mean() * 100
        overall_bias = scored_log["error"].mean()

        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("Scored rows", f"{len(scored_log):,}")
        with metric_cols[1]:
            st.metric("MAE", f"{overall_mae:.1f}")
        with metric_cols[2]:
            st.metric("RMSE", f"{overall_rmse:.1f}")
        with metric_cols[3]:
            st.metric("MAPE", "N/A" if pd.isna(overall_mape) else f"{overall_mape:.1f}%")
        with metric_cols[4]:
            st.metric("Bias", f"{overall_bias:.1f}")

        st.subheader("By Horizon")
        st.dataframe(
            metrics_by_horizon.round(
                {"mae": 1, "rmse": 1, "mape": 1, "bias": 1}
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Scored Forecast Rows")
        display_scored = scored_log[
            [
                "forecast_date",
                "target_date",
                "horizon",
                "predicted",
                "actual",
                "error",
                "abs_error",
                "pct_error",
            ]
        ].copy()
        display_scored["forecast_date"] = pd.to_datetime(
            display_scored["forecast_date"]
        ).dt.strftime("%Y-%m-%d")
        display_scored["target_date"] = pd.to_datetime(
            display_scored["target_date"]
        ).dt.strftime("%Y-%m-%d")
        display_scored["pct_error"] = display_scored["pct_error"] * 100
        st.dataframe(
            display_scored.sort_values(["target_date", "horizon"], ascending=[False, True]).round(
                {
                    "predicted": 1,
                    "actual": 1,
                    "error": 1,
                    "abs_error": 1,
                    "pct_error": 1,
                }
            ),
            width="stretch",
            hide_index=True,
        )
