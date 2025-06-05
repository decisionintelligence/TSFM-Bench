# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.fixed_forecast import FixedForecast
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast
from ts_benchmark.evaluation.strategy.rolling_forecast_for_autotimes import AutoTimesForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
    "autotimes_forecast": AutoTimesForecast,
}