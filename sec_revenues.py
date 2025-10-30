from pdb import main
from dotenv import load_dotenv
import os
from edgar import set_identity, Company, get_filings
from edgar.xbrl import XBRL
import pandas as pd
import re

import pandas as pd


TICKERS = ["LUV", "AAL", "DAL", "UAL", "JBLU", "FLYYQ", "ALK"]

REVENUE_TAGS = ["Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "OperatingRevenue"]
INCOME_TAGS  = ["NetIncomeLoss", "ProfitLoss"]


def get_incomes(company_ticker):
    """
    Download the last 6 SEC 10-Q filings.
    """

    try:
        # print(company_ticker)
        company = Company(company_ticker)
        filing = company.latest("10-Q")         
        xbrl = XBRL.from_filing(filing)
        income_df = xbrl.statements.income_statement()   
        #print(income_df)
        if income_df is None or income_df.empty:
            return pd.DataFrame()
        income_df["ticker"] = company_ticker
        return income_df
    except Exception as e:
        print(f"[{company_ticker}] Error while fetching income statement: {e}")
        return pd.DataFrame()
    # way to get income statement that actually works
    # company = Company("LUV")
    # filing = company.latest("10-Q")
    # xbrl = XBRL.from_filing(filing)
    # print(xbrl.statements.income_statement())
    # return None

def pick_by_precedence(df, tag_list):
    """Return a single numeric value from df for the first tag that exists."""
    for tag in tag_list:
        sub = df[df["fact"] == tag]
        if sub.empty:
            continue
        # prefer latest period end within that tag (if available)
        if "end" in sub.columns:
            sub = sub.assign(_end=pd.to_datetime(sub["end"], errors="coerce")).sort_values("_end")
        return pd.to_numeric(sub["val"], errors="coerce").dropna().iloc[-1] if not sub.empty else None
    return None


def pick_lineitem_regex(df, pattern):
    """Find a value whose label matches a text pattern like 'Passenger ancillary'."""
    if "label" not in df.columns:
        return None
    sub = df[df["label"].astype(str).str.contains(pattern, case=False, na=False)]
    if sub.empty and "fact" in df.columns:
        sub = df[df["fact"].astype(str).str.contains(pattern, case=False, na=False)]
    if sub.empty:
        return None
    return pd.to_numeric(sub["val"], errors="coerce").dropna().iloc[-1]



def build_quarterly_table(tickers):
    rows = []
    for t in tickers:
        df = get_incomes(t)
        if df.empty:
            continue

        # one row per filing
        year = df["fy"].dropna().iloc[0] if "fy" in df else None
        quarter = df["fp"].dropna().iloc[0] if "fp" in df else None
        filed = df["filed"].dropna().iloc[0] if "filed" in df else None
        accn = df["accn"].dropna().iloc[0] if "accn" in df else None

        ancillary = (
            pick_lineitem_regex(df, r"Passenger\s+ancillary\s+sold\s+separately") or
            pick_lineitem_regex(df, r"\bPassenger\s+ancillary\b") or
            pick_lineitem_regex(df, r"\bAncillary\b")
        )

        rows.append({
            "company": t,
            "year_quarter": f"{year}-{quarter}",
            "passenger_ancillary": ancillary,
            "filed_date": filed,
            "accession": accn
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["company", "year_quarter"], ascending=[True, False]).groupby("company").head(6)



def main():
    load_dotenv() # load env variables

    # Set your identity to retrieve SEC filings
    identity = os.environ.get('EDGAR_IDENTITY')
    email = os.environ.get('EDGAR_EMAIL')

    # Set identity
    set_identity(f"{identity} {email}")

    # Download 10-Q filings for various airline companies and build the quarterly table
    df = build_quarterly_table(TICKERS)
    print(df.head(20))
    
    df.to_csv("airlines_10Q_income+revenue.csv", index=False) # save to csv


if __name__ == "__main__":
    main()