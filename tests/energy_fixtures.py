import csv
from datetime import date, timedelta
import io


def report_bytes(start=date(2023, 1, 1), days=30, constant=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["\\\\Hourly Demand Report", "", "", ""])
    writer.writerow(["Date", "Hour", "Market Demand", "Ontario Demand"])
    for index in range(days):
        day = start + timedelta(days=index)
        for hour in range(1, 25):
            load = constant if constant is not None else 10000 + index * 10 + hour * 50 + day.weekday() * 100
            writer.writerow([day.isoformat(), hour, load + 1000, load])
    return stream.getvalue().encode()
