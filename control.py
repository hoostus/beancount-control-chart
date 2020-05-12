#!/usr/bin/env python3

""" Calculate monthly net worth """

import collections
import datetime
import logging
import time
import argparse
import logging

from dateutil import rrule
from dateutil.parser import parse

import pandas
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
import matplotlib
import seaborn

from beancount.core import data
from beancount.ops import holdings
from beancount import loader
from beancount.reports import holdings_reports
from beancount.query import query

QUERY = """
SELECT
	year(date) AS year,
        month(date) AS month,
        convert(sum(cost(position)), 'USD') AS balance
WHERE
        account ~ 'Expenses:' AND
        account != 'Expenses:Tax'
"""

def parse_args():
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s: %(message)s')
    parser = argparse.ArgumentParser(description=__doc__.strip())

    parser.add_argument('--min-date', action='store',
                        type=lambda string: parse(string).date(),
                        help="Minimum date")
    parser.add_argument('--ignore-account', action='append', default=[])
    parser.add_argument('--currency', help="Currency to report in")
    parser.add_argument('--control-limit', nargs=2, action='append', help="Define a new control limit with a name and a percentage value. e.g. --control-limit danger 0.05")
    parser.add_argument('--control-limit-absolute', nargs=2, action='append', help="Define a new control limit with a name and an absolute value.")
    parser.add_argument('--output', default='process-chart.png', help="The filename to write the graph to")
    parser.add_argument('--display', default=False, action='store_true', help='Display the chart instead of saving it to a file')
    parser.add_argument('--rolling', default=12, type=int, help='How many months to use when calculating the rolling average.')

    parser.add_argument('filename', help='Beancount input filename')
    args = parser.parse_args()
    return args

def get_spending(entries, options_map, min_date):
    query_rows = query.run_query(entries, options_map, QUERY, numberify=True)
    spending = [(datetime.date(n[0], n[1], 1), float(n[2])) for n in query_rows[1]]
    filtered = [n for n in spending if n[0] >= min_date]
    return pandas.Series(dict(filtered))

def get_networth(entries, options_map, args):
    net_worths = []
    index = 0
    current_entries = []

    dtend = datetime.date.today()
    period = rrule.rrule(rrule.MONTHLY, bymonthday=1, dtstart=args.min_date, until=dtend)

    for dtime in period:
        date = dtime.date()

        # Append new entries until the given date.
        while True:
            entry = entries[index]
            if entry.date >= date:
                break
            current_entries.append(entry)
            index += 1

        # Get the list of holdings.
        raw_holdings_list, price_map = holdings_reports.get_assets_holdings(current_entries,
                                                                            options_map)

        # Remove any accounts we don't in our final total
        filtered_holdings_list = [n for n in raw_holdings_list if n.account not in args.ignore_account]

        # Convert the currencies.
        holdings_list = holdings.convert_to_currency(price_map,
                                                        args.currency,
                                                        filtered_holdings_list)

        holdings_list = holdings.aggregate_holdings_by(
            holdings_list, lambda holding: holding.cost_currency)

        holdings_list = [holding
                            for holding in holdings_list
                            if holding.currency and holding.cost_currency]

        # If after conversion there are no valid holdings, skip the currency
        # altogether.
        if not holdings_list:
            continue

        # TODO: How can something have a book_value but not a market_value?
        value = holdings_list[0].market_value or holdings_list[0].book_value
        net_worths.append((date, float(value)))
        logging.debug("{}: {:,.2f}".format(date, value))

    return pandas.Series(dict(net_worths))

def build_dataframe(args):
    logging.info("Loading beancount file")
    entries, errors, options_map = loader.load_file(args.filename)

    if not args.min_date:
        for entry in entries:
            if isinstance(entry, data.Transaction):
                args.min_date = entry.date
                break

    if not args.currency:
        args.currency = options_map['operating_currency'][0]
    logging.info("Using currency {}".format(args.currency))

    logging.info("Building spending history")
    spending_series = get_spending(entries, options_map, args.min_date)

    logging.info("Building net worth history")
    networth_series = get_networth(entries, options_map, args)

    combined_df = pandas.concat([spending_series, networth_series], axis=1, sort=False)
    # we convert the monthly rate to an annual one by multiplying by 12
    spending_rate = (combined_df[0] * 12) / (combined_df[1])
    moving_average = (combined_df[0].rolling(args.rolling).mean() * 12) / combined_df[1].rolling(args.rolling).mean()

    df = pandas.concat([spending_rate, moving_average], axis=1, sort=False)
    df.columns = ['monthly spending rate', 'moving average']

    # define all of the control limits.
    for (name, value) in args.control_limit:
        df[name] = float(value)

    for (name, value) in args.control_limit_absolute:
        df[name] = float(value) / networth_series

    return df

def chart(df, args):
    df['date'] = df.index

    melted = df.melt(id_vars='date')
    stdev = df['monthly spending rate'].std()

    matplotlib.rcParams['font.family'] = 'League Spartan'
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    seaborn.set_context(rc={"lines.linewidth": 2.5})
    seaborn.lineplot(ax=ax, x='date', y='value', hue='variable', data=melted, palette='Set2')
    
    seaborn.despine(ax=ax, left=True, bottom=True, offset=20) 
    plt.xticks(rotation=45)
    # add in stdevs, leave it 1 stdev for now but consider using 2stdevs in future.
    # we really only care about upward variance (i.e. more spending that expected)
    plt.fill_between(df.index,
            df['moving average'],
            df['moving average'] + stdev,
            alpha=0.2)
    ax.set_ylim(bottom=0)
    plt.title('Spending Process Control Chart')

    if args.display:
        plt.show()
    else:
        fig.savefig(args.output)

if __name__ == '__main__':
    args = parse_args()
    df = build_dataframe(args)
    chart(df, args)
