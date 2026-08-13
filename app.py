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
GOOGLE_SHEET_READ_TTL_SECONDS = 60
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
        pass
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


@st.cache_resource(show_spinner=True)
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
        ttl=GOOGLE_SHEET_READ_TTL_SECONDS,
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
        ttl=GOOGLE_SHEET_READ_TTL_SECONDS,
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
        ttl=GOOGLE_SHEET_READ_TTL_SECONDS,
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
    refresh_google_sheet_connection()


def save_forecast_log(log: pd.DataFrame) -> None:
    ok, message = sync_forecast_log_to_google_sheet(log)
    if not ok:
        raise RuntimeError(message)
    refresh_google_sheet_connection()


def save_calendar(calendar: pd.DataFrame) -> None:
    ok, message = sync_calendar_to_google_sheet(calendar)
    if not ok:
        raise RuntimeError(message)
    refresh_google_sheet_connection()


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


def calculate_daily_forecast_metrics(scored_log: pd.DataFrame) -> pd.DataFrame:
    daily_metrics = (
        scored_log.groupby("target_date", dropna=False)
        .agg(
            rows=("actual", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            mape=("pct_error", lambda values: float(values.mean() * 100)),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    daily_metrics["target_date"] = pd.to_datetime(daily_metrics["target_date"])
    return daily_metrics.sort_values("target_date")


def calculate_daily_metrics_by_horizon(scored_log: pd.DataFrame) -> pd.DataFrame:
    daily_by_horizon = (
        scored_log.groupby(["target_date", "horizon"], dropna=False)
        .agg(
            rows=("actual", "size"),
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            mape=("pct_error", lambda values: float(values.mean() * 100)),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    daily_by_horizon["target_date"] = pd.to_datetime(daily_by_horizon["target_date"])
    return daily_by_horizon.sort_values(["target_date", "horizon"])


def highlight_next_day_forecasts(row: pd.Series) -> list[str]:
    next_day = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    target_date = pd.to_datetime(row.get("target_date"), errors="coerce")
    if pd.notna(target_date) and target_date.normalize() == next_day:
        return ["background-color: #FFE8B3; color: #111827"] * len(row)
    return [""] * len(row)


def date_only_column_config(data: pd.DataFrame) -> dict[str, Any]:
    date_columns = ["ds", "forecast_date", "target_date"]
    return {
        column: st.column_config.DatetimeColumn(column, format="YYYY-MM-DD")
        for column in date_columns
        if column in data.columns
    }


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


st.title(":material/zone_person_urgent: Urgent Appointments Forecast - SMW")

missing_files = require_files()
if missing_files:
    st.error("Missing required files: " + ", ".join(missing_files))
    st.stop()

with st.sidebar:
    st.header(":material/database: Data")
    st.caption(f"Time series source: Google Sheet `{google_ts_worksheet_name()}`")
    st.caption(f"Forecast log source: Google Sheet `{google_worksheet_name()}`")
    if st.button(
        "Reload app data",
        icon=":material/refresh:",
        width="stretch",
        key="reload_app_data",
    ):
        refresh_google_sheet_connection()
        st.rerun()
    st.divider()
    st.header(":material/apk_document: Google Sheet")
    st.link_button(
        "Open target sheet",
        GOOGLE_SHEET_URL,
        icon=":material/open_in_new:",
        width="stretch",
    )
    st.caption(f"Forecast worksheet: `{google_worksheet_name()}`")
    st.caption(f"Actuals worksheet: `{google_ts_worksheet_name()}`")
    st.caption(f"Calendar worksheet: `{google_calendar_worksheet_name()}`")
    st.divider()
    can_sync_google_sheet = has_google_sheet_write_auth()
    if can_sync_google_sheet:
        st.success(":material/cloud_done: GSheets connection **Live**")
    else:
        st.info("Public sheet links are read-only. Add service account fields to `[connections.gsheets]` to save.")

ts = read_ts()
forecast_log = read_forecast_log()
last_actual_date = ts["ds"].max()

top_cols = st.columns(4)
with top_cols[0]:
    metric_card("Latest actual date", last_actual_date.strftime("%Y-%m-%d"))
with top_cols[1]:
    metric_card("Latest used slots", f"{int(ts.iloc[-1]['y']):,}")
with top_cols[2]:
    metric_card("Rows in ts.csv", f"{len(ts):,}")
with top_cols[3]:
    metric_card("Forecast log rows", f"{len(forecast_log):,}")

tab_single, tab_weekend, tab_history, tab_metrics = st.tabs(
    ["Run prediction", "Add weekend to TS", "History", "Metrics"]
)

with tab_single:
    st.subheader(":shimmer[Add one day and forecast the next 5 working days]")
    st.caption("Update `ts.csv` with weekend data as 0 directly in the Gsheet skipping prediction.")
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
                    refresh_google_sheet_connection()
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
            forecast,
            column_config=date_only_column_config(forecast),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download updated forecast log",
            format_forecast_log_for_sheet(updated_log).to_csv(index=False).encode("utf-8"),
            file_name="nbeats_forecast_log.csv",
            mime="text/csv",
        )

with tab_weekend:
    st.subheader("Add weekend to TS")
    st.caption("Adds zero-valued weekend rows only when the latest `ts` date is a Friday.")

    with st.form("add_weekend_form"):
        add_saturday = st.checkbox("Saturday", value=True)
        add_sunday = st.checkbox("Sunday", value=True)
        add_weekend = st.form_submit_button(
            "Add weekend to TS",
            type="primary",
        )

    if add_weekend:
        if not can_sync_google_sheet:
            st.error("Google Sheet write sync needs service-account credentials before weekend rows can be added.")
            st.stop()

        if not add_saturday and not add_sunday:
            st.warning("Select Saturday, Sunday, or both before adding weekend rows.")
            st.stop()

        try:
            refresh_google_sheet_connection()
            latest_ts = read_ts()
            latest_date = pd.Timestamp(latest_ts["ds"].max()).normalize()
            latest_day_name = latest_date.day_name()

            if latest_date.weekday() != 4:
                st.warning(
                    "Not appropriate to apply weekend: "
                    f"last date is a {latest_day_name}."
                )
                st.stop()

            weekend_dates: list[pd.Timestamp] = []
            if add_saturday:
                weekend_dates.append(latest_date + pd.Timedelta(days=1))
            if add_sunday:
                weekend_dates.append(latest_date + pd.Timedelta(days=2))

            updated_ts = latest_ts.copy()
            for weekend_date in weekend_dates:
                updated_ts = upsert_actual(updated_ts, weekend_date, 0)

            save_ts(updated_ts)
            added_rows = updated_ts[
                updated_ts["ds"].dt.normalize().isin(weekend_dates)
            ].copy()
            st.success(f"Added {len(added_rows)} weekend row(s) to `{google_ts_worksheet_name()}`.")
            st.dataframe(
                added_rows,
                column_config=date_only_column_config(added_rows),
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Weekend update failed: {exc}")

with tab_history:
    st.subheader("Recent actuals")
    recent_actuals = ts.copy()
    show_recent_actuals_window = st.toggle(
        "Show actuals from 2026-04-01",
        value=False,
        key="show_recent_actuals_window",
    )
    actuals_chart = recent_actuals
    if show_recent_actuals_window:
        actuals_chart = actuals_chart[
            actuals_chart["ds"] >= pd.Timestamp("2026-04-01")
        ]
    st.line_chart(actuals_chart.set_index("ds")["y"])
    st.subheader("ts")
    recent_actuals_display = recent_actuals.tail(20)
    st.dataframe(
        recent_actuals_display,
        column_config=date_only_column_config(recent_actuals_display),
        width="stretch",
        hide_index=True,
    )

    forecast_header, refresh_col = st.columns([1, 0.25], vertical_alignment="center")
    with forecast_header:
        st.subheader("Forecast log")
    with refresh_col:
        if st.button("Refresh", key="refresh_forecast_log", icon=":material/refresh:"):
            refresh_google_sheet_connection()
            st.rerun()

    forecast_log_display = forecast_log.sort_values(
        ["forecast_date", "horizon"],
        ascending=[False, True],
    )
    st.dataframe(
        forecast_log_display.style.apply(highlight_next_day_forecasts, axis=1),
        column_config=date_only_column_config(forecast_log_display),
        width="stretch",
        hide_index=True,
    )

with tab_metrics:
    st.subheader("Forecasting Accuracy Metrics")
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
            st.metric("Scored rows", f"{len(scored_log):,}", icon=":material/show_chart:")
        with metric_cols[1]:
            st.metric("MAE", f"{overall_mae:.1f}", icon=":material/show_chart:")
        with metric_cols[2]:
            st.metric("RMSE", f"{overall_rmse:.1f}", icon=":material/show_chart:")
        with metric_cols[3]:
            st.metric("MAPE", "N/A" if pd.isna(overall_mape) else f"{overall_mape:.1f}%", icon=":material/show_chart:")
        with metric_cols[4]:
            st.metric("Bias", f"{overall_bias:.1f}", icon=":material/show_chart:")

        daily_metrics = calculate_daily_forecast_metrics(scored_log)
        daily_metrics_by_horizon = calculate_daily_metrics_by_horizon(scored_log)
        metric_colors = {
            "RMSE": "#252f3d",
            "Bias": "#306f83",
            "MAE": "#FFD814",
        }
        metric_descriptions = {
            "MAPE": (
                "MAPE shows the average forecast error as a percentage of the actual value. "
                "Lower is better. It is useful for judging relative accuracy, but can be noisy "
                "when actual values are very small."
            ),
            "RMSE": (
                "RMSE shows the typical forecast miss in urgent slots while giving extra weight "
                "to larger misses. Lower is better. A rising RMSE means the model is making bigger "
                "absolute mistakes."
            ),
            "MAE": (
                "MAE shows the average size of the forecast miss in urgent slots. Lower is better. "
                "It is easy to interpret because it stays in the same unit as the forecast."
            ),
            "Bias": (
                "Bias shows whether the model is usually too high or too low. Values near zero are "
                "balanced. Positive bias means under-forecasting; negative bias means over-forecasting."
            ),
        }

        st.subheader("Performance Trend")
        trend_metric = st.segmented_control(
            "Metric",
            ["MAPE", "RMSE", "MAE", "Bias"],
            default="MAPE",
            key="performance_trend_metric",
        )
        st.markdown(f"`{metric_descriptions[trend_metric]}`")
        trend_column = trend_metric.lower()
        trend_display = daily_metrics.set_index("target_date")[[trend_column]]
        st.line_chart(
            trend_display,
            color=metric_colors.get(trend_metric),
            height=280,
        )

        st.subheader("Horizon Diagnostics")
        horizon_cols = st.columns(3)
        with horizon_cols[0]:
            st.caption("MAPE by horizon")
            st.bar_chart(
                metrics_by_horizon.set_index("horizon")["mape"],
                height=260,
            )
        with horizon_cols[1]:
            st.caption("RMSE by horizon")
            st.bar_chart(
                metrics_by_horizon.set_index("horizon")["rmse"],
                color="#252f3d",
                height=260,
            )
        with horizon_cols[2]:
            st.caption("Bias by horizon")
            st.bar_chart(
                metrics_by_horizon.set_index("horizon")["bias"],
                color="#306f83",
                height=260,
            )

        st.subheader("Trend by Horizon")
        horizon_trend_metric = st.segmented_control(
            "Trend metric",
            ["MAPE", "RMSE", "Bias"],
            default="MAPE",
            key="horizon_trend_metric",
        )
        horizon_trend_column = horizon_trend_metric.lower()
        horizon_trend = daily_metrics_by_horizon.pivot_table(
            index="target_date",
            columns="horizon",
            values=horizon_trend_column,
            aggfunc="mean",
        ).sort_index()
        horizon_trend.columns = [f"H{int(column)}" for column in horizon_trend.columns]
        st.line_chart(horizon_trend, height=300)


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
            column_config=date_only_column_config(display_scored),
            width="stretch",
            hide_index=True,
        )
    st.divider()

    with st.expander("Model Training Architecture"):
        st.subheader("Model Training Accuracy Metrics")
        st.code("""# NBeats Model Training Architecture
    model_nbeats = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=5,

        generic_architecture=True,

        num_stacks=10,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,

        n_epochs=85,
        batch_size=32,

        random_state=42,
        force_reset=True,

        pl_trainer_kwargs={
            "accelerator": "mps",
            "devices": 1
        }
    )""")

        st.image('metric1.png')
        with st.expander("SMAPE v MDAPE"):
            st.markdown("""### 1. What is SMAPE?
    SMAPE stands for **Symmetric Mean Absolute Percentage Error**. It is a metric used to measure the accuracy of a forecast.

    Why "Percentage Error"? Most error metrics (like MAE or RMSE) are expressed in the same units as the data (e.g., dollars, liters, or people). A percentage error is useful because it tells you the error relative to the size of the numbers. An error of 10 units is huge if you are predicting 100, but tiny if you are predicting 1,000,000.
    Why "Symmetric"? Standard MAPE (Mean Absolute Percentage Error) has a flaw: it treats "under-forecasting" and "over-forecasting" differently. If the actual value is 0, MAPE becomes undefined. SMAPE was designed to fix this by putting both the actual value and the forecasted value in the denominator, creating a "symmetric" calculation.
    The Scale: SMAPE results are typically expressed as a decimal between 0 and 2 (or 0% and 200%). In your plot, the values are very small (e.g., 0.025), which means the error is extremely low (2.5%), indicating a very accurate model.
    2. How to Interpret the Axes
    To read this plot, you must look at the relationship between the two axes:

    X-Axis: Forecast Horizon (working days)
    This represents how far into the future the model is looking.

    Day 1: The model is predicting what will happen tomorrow.
    Day 5: The model is predicting what will happen five days from now.
    General Rule: Usually, as the horizon increases, error goes up because the future is harder to predict.
    Y-Axis: SMAPE (Error Rate)
    This represents the "penalty" or the amount of error.

    Lower is better. A value of 0.025 means a 2.5% error.
    3. Interpreting This Specific Plot
    When you look at this specific line, you are seeing the trade-off between time and accuracy.

    The Descent (Days 1 to 3): Interestingly, your error decreases as you move from Day 1 to Day 3. This suggests that the model is actually more accurate at a 3-day horizon than it is for a 1-day horizon. This can happen if the model is capturing a weekly trend or if the 1-day volatility is higher than the 3-day average.
    The "Sweet Spot" (Day 3): The lowest point on the graph is at Day 3 (SMAPE
    ≈
    ≈ 0.024). This is the model's peak performance. At this specific horizon, the model's predictions are most reliable.
    The Ascent (Days 3 to 5): After Day 3, the error begins to climb. This is the "natural" behavior of forecasting. As the model tries to look further into the future (Day 4 and Day 5), uncertainty grows, and the error increases.
    4. Summary Checklist for Interpretation
    When you see this type of plot in the future, ask these three questions:

    Where is the minimum? That is your most accurate forecast horizon.
    How steep is the curve? A very steep rise after the minimum suggests that the model's accuracy degrades rapidly as you look further ahead.
    Is the error magnitude acceptable? In your plot, the error is very low (maxing out around 5.4%). This would be considered an excellent model in almost any industry.
    In short: Your plot shows that your model is most effective at a 3-day forecast horizon, with error rates remaining very low (between 2.4% and 5.4%) across all tested days.

    ### 1. What is MDAPE?
    MDAPE stands for **Median Absolute Percentage Error**.

    The "Median" Difference: Unlike SMAPE (which is an average), MDAPE uses the median. This makes it much more robust to outliers.
    Why use it? If your data has a few extreme "wild" days where the error was massive, an average (SMAPE) would be pulled upward by those outliers. The median (MDAPE) ignores those extreme swings and tells you where the "typical" or "middle" error lies for the majority of your data points.
    The Scale: Like SMAPE, it is a percentage-based error.
    2. Comparing the Two Plots (The "Conflict")
    You might notice that the shapes of the two plots are different. This is the most important part of your analysis.

    Feature	SMAPE Plot (The "Average" View)	MDAPE Plot (The "Typical" View)
    Lowest Point	Day 3 (Error
    ≈
    ≈ 0.024)	Day 2 (Error
    ≈
    ≈ 0.015)
    The Story	Tells you the average error magnitude across all predictions.	Tells you where the middle of your error distribution sits.
    Interpretation	Suggests the model is best at a 3-day horizon.	Suggests the model is most "typical" or consistent at a 2-day horizon.
    3. Why are they different?
    The difference in the "best day" (Day 3 vs. Day 2) reveals something critical about your model's error distribution:

    The Day 2 "Dip" in MDAPE: The fact that MDAPE is much lower at Day 2 suggests that for the majority of your data, the model is incredibly accurate at the 2-day horizon.
    The Day 3 "Dip" in SMAPE: The reason the SMAPE (average) doesn't drop as low at Day 2 as the MDAPE does is likely because there are outliers at Day 2.
    Example: On Day 2, you might have 90% of your predictions being perfect (low error), but 10% being huge errors. The MDAPE would ignore those huge errors and show a low value. However, the SMAPE (the average) would be pulled up by those huge errors.
    4. Final Conclusion for your Report
    If you are presenting this, here is how you should interpret the relationship:

    "The model shows a discrepancy between average error (SMAPE) and typical error (MDAPE). While the average error is lowest at a 3-day horizon, the median error is lowest at a 2-day horizon. This indicates that while the model is highly accurate for the majority of cases at Day 2, there are occasional large errors (outliers) at that horizon that pull the average error up. The 3-day horizon provides a more stable average performance."

    Which one should you trust?

    If your business goal is to minimize the total sum of errors, look at SMAPE.
    If your business goal is to ensure the most common/typical forecast is accurate, look at MDAPE.""")


        with st.expander("Comparative Analysis of Forecast Model Performance"):
            st.markdown("""# Comparative Analysis of Forecast Model Performance

    This analysis compares two different error metrics—**SMAPE** (Symmetric Mean Absolute Percentage Error) and **MDAPE** (Median Absolute Percentage Error)—to evaluate a forecasting model across different horizons (1 to 5 working days).

    ### 1. Metric Definitions
    *   **SMAPE (The "Average" View):** Measures the average percentage error. It is sensitive to all data points, meaning large errors (outliers) will significantly pull this value up.
    *   **MDAPE (The "Typical" View):** Measures the median percentage error. It is robust to outliers, showing the error level for the "middle" of your data.

    ---

    ### 2. Data Summary Table

    | Forecast Horizon | SMAPE (Average Error) | MDAPE (Typical Error) |
    | :--- | :--- | :--- |
    | **Day 1** | ~0.054 | ~0.025 |
    | **Day 2** | ~0.042 | **~0.015 (Minimum)** |
    | **Day 3** | **~0.024 (Minimum)** | ~0.023 |
    | **Day 4** | ~0.027 | ~0.022 |
    | **Day 5** | ~0.047 | ~0.020 |

    ---

    ### 3. Key Findings & Interpretation

    #### The "Optimal" Horizon Discrepancy
    There is a notable difference in where the "best" performance occurs depending on which metric you use:
    *   **SMAPE suggests Day 3 is best:** The average error is minimized at the 3-day horizon.
    *   **MDAPE suggests Day 2 is best:** The typical error is minimized at the 2-day horizon.

    #### Why are they different?
    The discrepancy reveals the presence of **outliers at the 2-day horizon**.
    At **Day 2**, the MDAPE is extremely low (~0.015), meaning that for the vast majority of your data, the model is incredibly accurate. However, the SMAPE is much higher (~0.042) because there are likely a few significant "misses" (outliers) at Day 2 that are pulling the average up.

    At **Day 3**, the error is more "balanced." The errors are spread more evenly, which is why the average (SMAPE) reaches its lowest point there.

    ### 4. Final Conclusion for Decision Making

    *   **If you want to minimize total aggregate error:** The model is most effective at a **3-day horizon**. This is your "safest" bet for overall stability.
    *   **If you want to know how the model performs for a typical case:** The model is most accurate at a **2-day horizon**, but you must be aware that it is prone to occasional large errors at this specific timeframe.

    **Summary Statement:**
    *"The model demonstrates high accuracy across all horizons (all errors < 6%). The discrepancy between the 2-day MDAPE minimum and the 3-day SMAPE minimum indicates that while the model is highly precise for most cases at 2 days, it is susceptible to intermittent outliers at that horizon. For stable, generalized performance, the 3-day horizon is the optimal target."*
    """)

        st.image('metric2.png')
