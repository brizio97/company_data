from pathlib import Path
import pandas as pd
import logging
import unittest
import numpy as np

logging.basicConfig(
    level=logging.ERROR,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True) 

from app.utils import confirmation_statement_to_data, incorporation_to_data

logging.basicConfig(
    level=logging.ERROR,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True) 


class TestIncorporationToData(unittest.TestCase):

    def setUp(cls):
        cls.data_dir = Path(__file__).parent/'tests' / 'data'

    def test_incorporation_to_data_live(cls):

        #Test case 1. Koan Records has the standard online filing confirmation. From 2017 so the pdf is an image, not text can be extracted directly from pdf.
        output_1 = incorporation_to_data('10867753').astype({'Number of Shares': 'str'}).sort_values(['Document ID', 'Type of Shares', 'Name']).reset_index(drop=True)
        correct_output_1 = pd.read_csv(cls.data_dir / 'KoanRecordsLTD_incorporation_to_data.csv',
                                dtype={
                                    'Number of Shares': 'str',
                                    'Type of Shares': 'object',
                                    'Name': 'object',
                                    'Document ID': 'object',
                                    'Company Number': 'object',
                                    'Company Name': 'object'
                                },
                                parse_dates=['Document Date'],
                                ).sort_values(['Document ID', 'Type of Shares', 'Name']).reset_index(drop=True)
        pd.testing.assert_frame_equal(output_1 , correct_output_1, check_like = True, check_names = False)


class TestConfirmationStatementToData(unittest.TestCase):

    def setUp(cls):
        cls.data_dir = Path(__file__).parent/'tests' / 'data'

    def test_confirmation_statement_to_data_live(cls):
        #Test case 1. Juniper has 2 confirmation statements with a few shareholders. They follow the modern online filing format, with text directly in the pdf.
        output_1 = confirmation_statement_to_data('14973476').astype({'Number of Shares': 'str'}).sort_values(['Document ID', 'Type of Shares', 'Name']).reset_index(drop=True)
        correct_output_1 = pd.read_csv(cls.data_dir / 'J_confirmation_statement_to_data.csv',
                                dtype={
                                    'Number of Shares': 'str',
                                    'Type of Shares': 'object',
                                    'Name': 'object',
                                    'Document ID': 'object',
                                    'Company Number': 'object',
                                    'Company Name': 'object'
                                },
                                parse_dates=['Document Date'],
                                ).sort_values(['Document ID', 'Type of Shares', 'Name']).reset_index(drop=True)
        pd.testing.assert_frame_equal(output_1 , correct_output_1, check_like = True, check_names = False)



if __name__ == "__main__":
   unittest.main()