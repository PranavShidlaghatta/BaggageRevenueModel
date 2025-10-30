from pdb import main
from dotenv import load_dotenv
import os
from edgar import set_identity, Company, get_filings

def get_incomes(company_ticker):
    """
    Download the last 6 SEC 10-Q filings.
    """
    c = Company(company_ticker)
    filings = c.get_filings(form="10-Q").latest(6)
    return filings

def main():
    load_dotenv() # load env variables

    # Set your identity to retrieve SEC filings
    identity = os.environ.get('EDGAR_IDENTITY')
    email = os.environ.get('EDGAR_EMAIL')

    # Set identity
    set_identity(f"{identity} {email}")

    # Download 10-Q filings for various airline companies
    ticker_symbols = ["LUV", "AAL", "DAL", "UAL", "JBLU", "FLYYQ", "ALK"]
    for ticker in ticker_symbols:
        print(get_incomes(ticker))


if __name__ == "__main__":
    main()