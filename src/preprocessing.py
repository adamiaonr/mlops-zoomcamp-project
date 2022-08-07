from typing import Union
from pathlib import Path

import pandas as pd


def add_extra_columns(data: pd.DataFrame):
    # - date of record in format 'YYYY-MM-DD'
    data['Day'] = data['RecordedAtTime'].dt.date

    # - seconds since start of day
    data['TimeOfDayInSeconds'] = (
        data['RecordedAtTime'] - data['RecordedAtTime'].dt.normalize()
    ).dt.total_seconds()

    # - weekday
    data['DayOfWeek'] = data['RecordedAtTime'].dt.dayofweek

    # - difference in seconds between 'RecordedAtTime' and 'ScheduledArrivalTime'
    #   when 'ArrivalProximityText' == 'at stop'
    mask = data['ArrivalProximityText'] == 'at stop'
    data['DelayAtStop'] = data[mask].apply(
        lambda r: (r['RecordedAtTime'] - r['ScheduledArrivalTime']).total_seconds(),
        axis=1,
    )


def fix_scheduled_arrival_time_column(data: pd.DataFrame, threshold: float = 12.0):
    # drop rows with 'ScheduledArrivalTime' with nan values
    data.dropna(subset=['ScheduledArrivalTime'], inplace=True)

    # transform 'ScheduledArrivalTime' to pd.datetime : use date from 'RecordedAtTime'
    data['ScheduledArrivalTime'] = data[
        'RecordedAtTime'
    ].dt.normalize() + pd.to_timedelta(data['ScheduledArrivalTime'].astype(str))

    # add or subtract 1 day to 'ScheduledArrivalTime' based on time difference
    # between 'RecordedAtTime' and 'ScheduledArrivalTime'
    time_diffs = (
        data['RecordedAtTime'] - data['ScheduledArrivalTime']
    ).dt.total_seconds() / 3600.0

    data.loc[time_diffs > threshold, 'ScheduledArrivalTime'] = data.loc[
        time_diffs > threshold, 'ScheduledArrivalTime'
    ].apply(lambda sat: sat + pd.to_timedelta(1, unit='d'))
    data.loc[time_diffs < -threshold, 'ScheduledArrivalTime'] = data.loc[
        time_diffs < -threshold, 'ScheduledArrivalTime'
    ].apply(lambda sat: sat - pd.to_timedelta(1, unit='d'))


def fix_datetime_columns(data: pd.DataFrame):
    # ensure no nan rows for 'ExpectedArrivalTime'
    mask = data['ExpectedArrivalTime'].isnull()
    data.loc[mask, 'ExpectedArrivalTime'] = data.loc[mask, 'RecordedAtTime']

    # ensure 'RecordedAtTime' and 'ExpectedArrivalTime' in datetime format
    data[['RecordedAtTime', 'ExpectedArrivalTime']] = data[
        ['RecordedAtTime', 'ExpectedArrivalTime']
    ].apply(pd.to_datetime, infer_datetime_format=True)


def load_data(
    data_dir: Union[str, Path], months: list[int], pre_process=True, **kws
) -> pd.DataFrame:

    data = pd.DataFrame()
    for file in [Path(data_dir) / f"mta_17{month:02d}.csv" for month in months]:
        # ignore lines with errors with on_bad_lines='skip'
        data = pd.concat(
            [pd.read_csv(file, index_col=False, on_bad_lines='skip', **kws), data]
        )

    if pre_process:
        # keep only rows with 'ArrivalProximityText' == 'at stop'
        data = data[data['ArrivalProximityText'] == 'at stop'].reset_index(drop=True)

        # fix datetime columns
        fix_datetime_columns(data)
        fix_scheduled_arrival_time_column(data)

        # add extra columns, which will be useful for delay prediction
        add_extra_columns(data)

        # sort by 'RecordedAtTime'
        data = data.sort_values(by=['RecordedAtTime']).reset_index(drop=True)

    return data
