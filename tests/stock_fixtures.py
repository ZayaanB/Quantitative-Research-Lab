import csv
from datetime import date, timedelta
from hashlib import sha256


def weekday_dates(count, start=date(2020, 1, 1)):
    dates = []
    day = start
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += timedelta(days=1)
    return dates


def write_price_asset(directory, symbol, dates, prices):
    history = directory / f"{symbol}.csv"
    with history.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Date", "Adj Close"])
        writer.writerows(zip(dates, prices))
    metadata = directory / f"{symbol}.meta"
    metadata.write_text(f"provider=synthetic\nsymbol={symbol}\nadjustment=none\n"
                        f"retrieved_on={dates[-1]}\nsource_url=local:fixture\n"
                        "price_column=Adj Close\nfrequency=daily\n", encoding="utf-8")
    return {"symbol": symbol, "csv": history.name, "metadata": metadata.name,
            "csv_sha256": sha256(history.read_bytes()).hexdigest(),
            "metadata_sha256": sha256(metadata.read_bytes()).hexdigest()}
