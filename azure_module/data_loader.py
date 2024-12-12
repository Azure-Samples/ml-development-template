import argparse
import pandas as pd
from pathlib import Path


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for loading and saving data.

    Methods
    -------
    __init__():
        Initializes the DataLoader instance.

    get_data():
        Loads data from the specified input file and returns it as a DataFrame.

    save_data(df, filename="default.csv"):
        Saves the given DataFrame to a CSV file in the specified output datastore.

    parse_input_arg():
        Parses the input data argument from the command line.

    parse_output_arg():
        Parses the output datastore argument from the command line.
    """

    def __init__(self):
        pass

    def get_data(self):
        logger.info(f"Data loaded from {self.input_data}.")
        df = pd.read_csv(self.input_data)
        return df

    def save_data(self, df, filename="default.csv"):
        output_path = Path(self.output_datastore) / filename
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    def parse_input_arg(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--input_data", dest="input_data", type=str, required=True)
        args = parser.parse_args()
        self.input_data = args.input_data

    def parse_output_arg(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--output_datastore", dest="output_datastore", type=str, required=True
        )
        args = parser.parse_args()
        self.output_datastore = args.output_datastore
