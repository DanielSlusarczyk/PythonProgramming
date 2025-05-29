from datetime import date, timedelta
import pandas as pd

def DescribeData(df: pd.DataFrame):
    nmbOfRows = df.shape[0]
    nmbOfColumns = df.shape[1]
    
    print(f'Size: {nmbOfRows} x {nmbOfColumns}\n')
    display(pd.DataFrame(df.isnull().sum(), columns=['Brakujące dane']))
    display(df.describe().T.rename(columns={
        'min' : 'Minimum',
        'max' : 'Maksium',
        'count' : 'Liczebność',
        'std' : 'Odch. Stand.',
        'mean' : 'Średnia'}))

    display(df.head(3))

def SplitDateColumn(df: pd.DataFrame, column: str, suffix = '', replace=False):
    
    if suffix == '':
        suffix = column

    df[suffix + '_minute'] = df[column].dt.minute
    df[suffix + '_hour'] = df[column].dt.hour
    df[suffix + '_day'] = df[column].dt.day
    df[suffix + '_month'] = df[column].dt.month
    df[suffix + '_year'] = df[column].dt.year
    df[suffix + '_datetime'] = df[column]
    df[suffix + '_time'] = df[column].dt.time
    df[suffix + '_date'] = df[column].dt.date

    if replace:
        df.drop(column, axis = 1, inplace=True)

    return df

def AddPrefixToColumns(df: pd.DataFrame, columns: [str], preffix: str):
    for column in columns:
        df.rename(columns={column : preffix + column}, inplace=True)

def IsEstionianHoliday(day: date):
    year = day.year
    easter = EasterSunday(year)

    if (
        # New Year's Day
        day == date(year, 1, 1) or
        # Independence Day
        day == date(year, 2, 24) or
        # Good Friday
        day == easter - timedelta(days=2) or
        # Easter Sunday
        day == easter or
        # Spring Day
        day == date(year, 5, 1) or
        # Pentecost
        day == easter + timedelta(days=49) or
        # Victory Day
        day == date(year, 6, 23) or
        # Midsummer Day
        day == date(year, 6, 24) or
        # Day of Restoration of Independence
        day == date(year, 8, 20) or
        # Christmas Eve
        day == date(year, 12, 24) or
        # Christmas Day
        day == date(year, 12, 25) or
        # Boxing day
        day == date(year, 12, 26)
        ):
        return True
    
    return False

def IsWeekend(day: date):
    return day.weekday() >= 5

def EasterSunday(year):
    g = year % 19
    c = year // 100
    h = (c - (c // 4) - ((8 * c + 13) // 25) + 19 * g + 15) % 30
    i = h - (h // 28) * (1 - (h // 28) * (29 // (h + 1)) * ((21 - g) // 11))

    day = i - ((year + (year // 4) + i + 2 - c + (c // 4)) % 7) + 28
    month = 3

    if day > 31:
        month += 1
        day -= 31

    return date(year, month, day)