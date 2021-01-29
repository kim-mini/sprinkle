"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""

from kocrawl.dust import DustCrawler
from kocrawl.map import MapCrawler
from kocrawl.retaurant import RestaurantCrawler
from kocrawl.weather import WeatherCrawler
from kochat.app import Scenario
from do import do_call, do_schedule

call = Scenario(
    intent='call',
    api=do_call,
    senario={
        'Target': [],
    }
)

schedule = Scenario(
    intent='schedule',
    api=do_schedule,
    senario={
        'Date':[],
        'Subject':[],
        'Time':[],
        'Action':[],
    }
)

