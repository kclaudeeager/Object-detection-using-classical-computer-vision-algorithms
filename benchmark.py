import time
import yaml
from main import main

def benchmark(config_path):
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    detectors = ['sift', 'surf', 'orb', 'template']
    matchers = ['flann', 'brute_force']

    results = {}

    for detector in detectors:
        for matcher in matchers:
            config = base_config.copy()
            config['feature_detector'] = detector
            config['matcher'] = matcher

            start_time = time.time()
            main(config)
            end_time = time.time()

            results[f"{detector}_{matcher}"] = end_time - start_time
            print(f"{detector}_{matcher}: {end_time - start_time:.2f} seconds")

    print("Benchmark Results:")
    for method, duration in results.items():
        print(f"{method}: {duration:.2f} seconds")

if __name__ == "__main__":
    benchmark("config.yaml")