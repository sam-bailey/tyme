import numpy as np
from tyme.base_forecasters.exponential_smoothing import ExponentialSmoothing
import argparse
import timeit, functools


def run_model(x, args, with_print=False):
    model = ExponentialSmoothing(args.alpha, args.beta, args.phi)
    model.filter(x)
    forecast = model.forecast(n_steps_max=args.output_length)

    if with_print:
        print("\nCython Version")
        print(model)
        print(forecast)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the basic exponential smoothing filter and forecaster")
    parser.add_argument("input_length", type=int)
    parser.add_argument("output_length", type=int)
    parser.add_argument("n_iterations", type=int)
    parser.add_argument("alpha", type=float)
    parser.add_argument("beta", type=float)
    parser.add_argument("phi", type=float)

    args = parser.parse_args()

    x = np.random.rand(args.input_length) + np.arange(args.input_length)
    run_model(x, args, with_print=True)

    t_cy = timeit.Timer(functools.partial(run_model, x, args))
    print("Time for {} iterations: ".format(args.n_iterations), t_cy.timeit(args.n_iterations))
