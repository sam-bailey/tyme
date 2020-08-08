import numpy as np
from tyme.base_forecasters.exponential_smoothing import exp_smoothing_filter, exp_smoothing_forecaster
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the basic exponential smoothing filter and forecaster")
    parser.add_argument("input_length", type=int)
    parser.add_argument("output_length", type=int)
    parser.add_argument("alpha", type=float)
    parser.add_argument("beta", type=float)
    parser.add_argument("phi", type=float)

    args = parser.parse_args()

    for i in np.arange(1000):
        x = np.random.rand(args.input_length) + np.arange(args.input_length)
        level, trend = exp_smoothing_filter(x, args.alpha, args.beta, args.phi)
        forecast = exp_smoothing_forecaster(level, trend, args.phi, n_steps_max=args.output_length)

    print(f"Level = {level}\nTrend = {trend}\nForecast = {forecast}")
