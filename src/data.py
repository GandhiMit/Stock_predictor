import yfinance as yf
from datetime import timedelta

def load_data(company, start, end):
    return yf.download(company, start=start, end=end)

def generate_prediction_dates(start_date, num_days):
    dates = []
    current_date = start_date
    while len(dates) < num_days:
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates
