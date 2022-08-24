import unittest

import pandas as pd

from src.preprocessing import fix_scheduled_arrival_time_column


class TestPreprocessing(unittest.TestCase):
    def test_fix_scheduled_arrival_time_column(self):
        data = pd.DataFrame(
            {
                'RecordedAtTime': pd.to_datetime(
                    [
                        '2022-07-31 23:50:00',
                        '2022-07-31 23:50:00',
                        '2022-07-31 23:50:00',
                        '2022-08-01 00:05:00',
                        '2022-08-01 00:05:00',
                        '2022-08-01 00:05:00',
                    ],
                    infer_datetime_format=True,
                ),
                'ScheduledArrivalTime': [
                    '23:55:00',
                    '24:05:00',
                    '00:05:00',
                    '23:55:00',
                    '24:05:00',
                    '00:05:00',
                ],
            }
        )

        fix_scheduled_arrival_time_column(data)

        # differences between 'RecordedAtTime' and 'ScheduledArrivalTime' must be within 1 hr
        time_diff = abs(
            (data['RecordedAtTime'] - data['ScheduledArrivalTime']).dt.total_seconds()
            / 3600.0
        )
        self.assertLess(max(time_diff), 1.0)
