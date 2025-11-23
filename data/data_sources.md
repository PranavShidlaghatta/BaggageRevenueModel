# data_sources.md

## **`combined_bag_revenue.csv`**

* **Content**

  * Aggregated baggage revenues for major US airlines over the last 11 years.
  * Quarterly revenue data collected for each year (2014–2024).
  * Includes **41 columns**, where:

    * **`Airlines`** is the primary identifier.
    * All remaining columns follow the naming pattern **`Year-Quarter`** (e.g., `2015-Q1`).
* **Source**

  * Bureau of Transportation Statistics (BTS)

    * [https://www.bts.gov/topics/airlines-and-airports/baggage-fees-airline-2024](https://www.bts.gov/topics/airlines-and-airports/baggage-fees-airline-2024)
* **Preprocessing**

  * Python script used to:

    * Join annual BTS files.
    * Rename columns to standardized `Year-Quarter` format.
    * Drop rows containing null values.


## **bag-rev-2015-2024 Folder**

* **Content**

  * Contains individual yearly CSV files of quarterly baggage revenue data.
  * Each file represents **one full year** beginning with **2014**.
* **Source**

  * Bureau of Transportation Statistics (BTS)

    * [https://www.bts.gov/topics/airlines-and-airports/baggage-fees-airline-2024](https://www.bts.gov/topics/airlines-and-airports/baggage-fees-airline-2024)


## **`CPI_quarterly_only.csv`**

* **Columns**

  * `Date`
  * `CPI`
* **Source**

  * Bureau of Labor Statistics (BLS) API.
* **Preprocessing**

  * Monthly CPI values aggregated into **quarterly** values per year.
  * Used as an inflation-normalizing metric for other datasets.

## **`jet_fuel_prices.csv`**

* **Columns**

  * `Month of Date`
  * `Fuel Type`
  * `Price (cents per Gallon)`
* **Source**

  * Bureau of Transportation Statistics (BTS)

    * [https://www.bts.gov/browse-statistical-products-and-data/freight-facts-and-figures/diesel-and-jet-fuel-prices](https://www.bts.gov/browse-statistical-products-and-data/freight-facts-and-figures/diesel-and-jet-fuel-prices)
* **Preprocessing**

  * Monthly dates standardized to **quarterly** format.
  * Fuel price converted to floating-point representation.

## **`US Unemployment Data 2014-2024.csv`**

* **Columns**

  * `Month`
  * `Total`
* **Source**

  * Bureau of Labor Statistics (BLS)

    * [https://www.bls.gov/charts/employment-situation/civilian-unemployment-rate.htm](https://www.bls.gov/charts/employment-situation/civilian-unemployment-rate.htm)
* **Preprocessing**

  * Monthly values standardized into **quarterly** format (same method as jet fuel prices dataset).


## **`GDP_per_capita 1960-2024.csv`**

* **Columns**

  * ~70 columns representing GDP-per-capita by year.
  * Includes country name and country code.
* **Source**

  * Federal Reserve Economic Data (FRED):

    * [https://fred.stlouisfed.org/series/A939RX0Q048SBEA](https://fred.stlouisfed.org/series/A939RX0Q048SBEA)
* **Preprocessing**

  * Extracted the **United States** row (complete with no missing values for relevant years).
  * Rotated dataset so years became rows rather than columns.
  * Standardized year/month values to **quarterly** format.