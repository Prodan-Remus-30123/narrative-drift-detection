PERIOD_ORDER = {
    "01_02": 1,
    "03_04": 2,
    "05_06": 3,
    "07_08": 4,
    "09_10": 5,
    "11_12": 6
}


def sort_period_key(period):

    if "->" in period:
        period = period.split("->")[0]

    year = int(period[:4])
    block = period[5:]

    return (
        year,
        PERIOD_ORDER[block]
    )