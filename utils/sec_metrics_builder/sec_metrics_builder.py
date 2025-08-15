#!/usr/bin/env python3
"""
SEC Metrics Builder - Comprehensive Financial Data Extraction with Warren Buffett-Style Analysis
===============================================================================================

Fetches all 104 working SEC-available financial metrics + 22 derived Warren Buffett-style metrics
for companies from 2015-2025. Creates standardized CSV files with institutional-grade analysis.

Features:
- 102 comprehensive SEC XBRL metrics (30 empty metrics removed)
- 2 market price metrics via Financial Modeling Prep API (optional)
- 22 derived Warren Buffett-style metrics (EPS, P/E, ROE, FCF, etc.)
- 126 total metrics per company per year
- 10,069 companies from SEC database
- 11 years of historical data (2015-2025)
- Respectful rate limiting (default 0.5s between companies)
- Standardized CSV format with validation
- Balance sheet equation validation
- Comprehensive failure reporting (JSON + CSV formats)
- ETF filtering options (--skip-etf / --etf-only)

Usage:
    export FMP_API_KEY="your_fmp_api_key"
    python sec_metrics_builder.py --companies all --years 2015 2025 --skip-etf
    python sec_metrics_builder.py --ticker AAPL --years 2020 2025 --fmp-api-key YOUR_KEY
    python sec_metrics_builder.py --top 100 --years 2015 2025 --rate-limit 1.0
    python sec_metrics_builder.py --top 50 --years 2015 2025 --etf-only
"""

import argparse
import csv
import json
import logging
import os
import pandas as pd
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sec_metrics_builder.log')
    ]
)
logger = logging.getLogger(__name__)

# SEC API Configuration
SEC_BASE_URL = "https://data.sec.gov/api/xbrl"
SEC_COMPANIES_FILE = "sec_companies.json"

# Request session with proper headers for SEC compliance
session = requests.Session()
session.headers.update({
    'User-Agent': 'SEC Financial Metrics Builder (luluventures.ivy@gmail.com)',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
})

# Price data sources configuration
PRICE_CACHE_FILE = "stock_price_history_universal.json"  # Use universal cache with proper split adjustments
PRICE_CACHE = None  # Will be loaded on initialization

# Financial Modeling Prep API configuration (fallback)
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_API_KEY = None  # Will be set from environment variable

# Comprehensive SEC XBRL tag mappings (102 metrics total) - ORDERED (30 empty metrics removed)
SEC_METRICS = {
    # BALANCE SHEET - Assets (15 metrics)
    'Assets': ['Assets', 'AssetsTotal', 'AssetsTotalCurrentAndNoncurrent'],
    'AssetsCurrent': ['AssetsCurrent'],
    'AssetsNoncurrent': ['AssetsNoncurrent'],
    'CashAndCashEquivalents': ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents', 'CashAndCashEquivalents'],
    'CashAndRestrictedCash': ['CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'MarketableSecurities': ['MarketableSecurities', 'AvailableForSaleSecurities'],
    'AccountsReceivableNet': ['AccountsReceivableNetCurrent', 'AccountsReceivableNet'],
    'Inventory': ['InventoryNet', 'Inventory'],
    'PropertyPlantAndEquipmentNet': ['PropertyPlantAndEquipmentNet'],
    'PropertyPlantAndEquipmentGross': ['PropertyPlantAndEquipmentGross'],
    'AccumulatedDepreciation': ['AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'],
    'Goodwill': ['Goodwill'],
    'IntangibleAssetsNet': ['IntangibleAssetsNetExcludingGoodwill', 'IntangibleAssetsNet'],
    'DeferredTaxAssetsNet': ['DeferredTaxAssetsNet'],
    'OtherAssets': ['OtherAssets', 'OtherAssetsNoncurrent'],

    # BALANCE SHEET - Liabilities (17 metrics)
    'Liabilities': ['Liabilities', 'LiabilitiesTotal'],
    'LiabilitiesCurrent': ['LiabilitiesCurrent'],
    'LiabilitiesNoncurrent': ['LiabilitiesNoncurrent'],
    'AccountsPayable': ['AccountsPayableCurrent', 'AccountsPayable'],
    'AccruedLiabilities': ['AccruedLiabilitiesCurrent', 'AccruedLiabilities'],
    'ShortTermBorrowings': ['ShortTermBorrowings', 'DebtCurrent'],
    'LongTermDebt': ['LongTermDebt', 'LongTermDebtNoncurrent'],
    'LongTermDebtCurrent': ['LongTermDebtCurrent'],
    'DebtCurrent': ['DebtCurrent'],
    'DebtNoncurrent': ['DebtNoncurrent', 'LongTermDebt'],
    'DeferredRevenue': ['DeferredRevenue', 'ContractWithCustomerLiabilityCurrent'],
    'DeferredRevenueNoncurrent': ['DeferredRevenueNoncurrent', 'ContractWithCustomerLiabilityNoncurrent'],
    'DeferredTaxLiabilities': ['DeferredTaxLiabilitiesNoncurrent', 'DeferredTaxLiabilities'],
    'EmployeeRelatedLiabilities': ['EmployeeRelatedLiabilitiesCurrent'],
    'OperatingLeaseLiability': ['OperatingLeaseLiability'],
    'FinanceLeaseLiability': ['FinanceLeaseLiability'],
    'OtherLiabilities': ['OtherLiabilities', 'OtherLiabilitiesNoncurrent'],

    # BALANCE SHEET - Equity (8 metrics)
    'StockholdersEquity': ['StockholdersEquity', 'StockholdersEquityTotal'],
    'StockholdersEquityIncludingNCI': ['StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'CommonStockValue': ['CommonStockValue'],
    'PreferredStockValue': ['PreferredStockValue'],
    'AdditionalPaidInCapital': ['AdditionalPaidInCapital'],
    'RetainedEarnings': ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings'],
    'TreasuryStock': ['TreasuryStockValue'],
    'NoncontrollingInterest': ['MinorityInterest', 'NoncontrollingInterest'],

    # INCOME STATEMENT - Revenue & Sales (4 metrics)
    'Revenues': ['Revenues', 'Revenue', 'TotalRevenues'],
    'RevenueFromContracts': ['RevenueFromContractWithCustomerExcludingAssessedTax'],
    'SalesRevenueNet': ['SalesRevenueNet'],
    'ProductSales': ['ProductSales', 'RevenueFromRelatedParties'],

    # INCOME STATEMENT - Cost & Expenses (13 metrics)
    'CostOfRevenue': ['CostOfRevenue', 'CostOfGoodsAndServicesSold'],
    'CostOfSales': ['CostOfSales', 'CostOfGoodsSold'],
    'ResearchAndDevelopmentExpense': ['ResearchAndDevelopmentExpense'],
    'SellingGeneralAndAdministrativeExpense': ['SellingGeneralAndAdministrativeExpense'],
    'GeneralAndAdministrativeExpense': ['GeneralAndAdministrativeExpense'],
    'SellingAndMarketingExpense': ['SellingAndMarketingExpense'],
    'DepreciationDepletionAndAmortization': ['DepreciationDepletionAndAmortization'],
    'AmortizationOfIntangibleAssets': ['AmortizationOfIntangibleAssets'],
    'RestructuringCosts': ['RestructuringCosts', 'RestructuringCharges'],
    'OperatingLeaseExpense': ['OperatingLeaseExpense'],
    'StockBasedCompensation': ['ShareBasedCompensation', 'StockBasedCompensation'],
    'OtherOperatingExpenses': ['OtherOperatingIncomeExpenseNet'],
    'TotalOperatingExpenses': ['OperatingExpenses'],

    # INCOME STATEMENT - Income & Profit (12 metrics)
    'GrossProfit': ['GrossProfit'],
    'OperatingIncomeLoss': ['OperatingIncomeLoss', 'IncomeLossFromOperations'],
    'IncomeTaxExpenseBenefit': ['IncomeTaxExpenseBenefit', 'IncomeTaxExpense'],
    'NetIncomeLoss': ['NetIncomeLoss', 'NetIncome'],
    'NetIncomeAvailableToCommonShareholders': ['NetIncomeLossAvailableToCommonStockholdersBasic'],
    'InterestExpense': ['InterestExpense', 'InterestExpenseDebt'],
    'OtherNonoperatingIncomeExpense': ['NonoperatingIncomeExpense', 'OtherNonoperatingIncomeExpense'],
    'GainLossOnSaleOfAssets': ['GainLossOnSaleOfPropertyPlantEquipment', 'GainLossOnSaleOfAssets'],
    'GainLossOnInvestments': ['GainLossOnInvestments'],
    'DiscontinuedOperations': ['IncomeLossFromDiscontinuedOperationsNetOfTax'],

    # CASH FLOW STATEMENT - Operating Activities (10 metrics)
    'NetCashFromOperatingActivities': ['NetCashProvidedByUsedInOperatingActivities'],
    'DepreciationAndAmortization': ['DepreciationDepletionAndAmortization', 'Depreciation'],
    'StockBasedCompensationExpense': ['ShareBasedCompensation', 'StockBasedCompensationExpense'],
    'DeferredIncomeTaxExpenseBenefit': ['DeferredIncomeTaxExpenseBenefit'],
    'ChangeInAccountsReceivable': ['IncreaseDecreaseInAccountsReceivable'],
    'ChangeInInventories': ['IncreaseDecreaseInInventories'],
    'ChangeInPrepaidExpenses': ['IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets'],
    'ChangeInAccountsPayable': ['IncreaseDecreaseInAccountsPayable'],
    'ChangeInAccruedLiabilities': ['IncreaseDecreaseInAccruedLiabilities'],
    'ChangeInDeferredRevenue': ['IncreaseDecreaseInDeferredRevenue'],

    # CASH FLOW STATEMENT - Investing Activities (7 metrics)
    'NetCashFromInvestingActivities': ['NetCashProvidedByUsedInInvestingActivities'],
    'CapitalExpenditures': ['PaymentsToAcquirePropertyPlantAndEquipment'],
    'BusinessAcquisitions': ['PaymentsToAcquireBusinessesNetOfCashAcquired'],
    'InvestmentPurchases': ['PaymentsToAcquireInvestments'],
    'AssetSales': ['ProceedsFromSaleOfPropertyPlantAndEquipment'],
    'IntangibleAssetPurchases': ['PaymentsToAcquireIntangibleAssets'],
    'BusinessDivestitures': ['ProceedsFromDivestitureOfBusinesses'],

    # CASH FLOW STATEMENT - Financing Activities (10 metrics)
    'NetCashFromFinancingActivities': ['NetCashProvidedByUsedInFinancingActivities'],
    'CommonStockIssuance': ['ProceedsFromIssuanceOfCommonStock'],
    'ShareRepurchases': ['PaymentsForRepurchaseOfCommonStock'],
    'CommonDividendsPaid': ['PaymentsOfDividendsCommonStock'],
    'PreferredDividendsPaid': ['PaymentsOfDividendsPreferredStockAndPreferenceStock'],
    'LongTermDebtIssuance': ['ProceedsFromIssuanceOfLongTermDebt'],
    'LongTermDebtRepayments': ['RepaymentsOfLongTermDebt'],
    'ShortTermDebtProceeds': ['ProceedsFromShortTermDebt'],
    'ShortTermDebtRepayments': ['RepaymentsOfShortTermDebt'],
    'DebtIssuanceCosts': ['PaymentsOfDebtIssuanceCosts'],

    # SHARE DATA - Share Counts (7 metrics)
    'CommonStockSharesOutstanding': ['CommonStockSharesOutstanding'],
    'CommonStockSharesIssued': ['CommonStockSharesIssued'],
    'WeightedAverageSharesBasic': ['WeightedAverageNumberOfSharesOutstandingBasic'],
    'WeightedAverageSharesDiluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],
    'DilutionAdjustment': ['WeightedAverageNumberDilutedSharesOutstandingAdjustment'],
    'PreferredStockSharesOutstanding': ['PreferredStockSharesOutstanding'],
    'TreasuryStockShares': ['TreasuryStockShares'],

    # SHARE DATA - Stock-Related (1 metric)
    'StockBasedCompensationExpenseTotal': ['ShareBasedCompensation'],

    # MARKET DATA - External APIs (2 metrics)
    'MarketPriceUSD': None,  # From Financial Modeling Prep - current split-adjusted price
    'MarketPriceNonSplitAdjustedUSD': None  # From Financial Modeling Prep - raw historical price
}

# Verify we have exactly 104 metrics (102 SEC + 2 market price metrics)
assert len(SEC_METRICS) == 104, f"Expected 104 metric mappings, got {len(SEC_METRICS)}"

# DERIVED WARREN BUFFETT-STYLE METRICS (22 additional metrics)
DERIVED_METRICS = [
    # Valuation Metrics (4)
    'EarningsPerShare',
    'PriceToEarning',
    'BookValuePerShare', 
    'PriceToBook',
    
    # Growth Metrics (3)
    'RevenueGrowthRate',
    'NetIncomeGrowthRate',
    'BookValueGrowthRate',
    
    # Profitability Margins (4)
    'GrossMargin',
    'OperatingMargin', 
    'NetIncomeMargin',
    'FreeCashFlowMargin',
    
    # Cash Flow Metrics (2)
    'FreeCashFlow',
    'OwnerEarnings',
    
    # Financial Health Ratios (3)
    'CurrentRatio',
    'DebtToEquityRatio',
    'InterestCoverageRatio',
    
    # Return Metrics (3)
    'ReturnOnEquity',
    'ReturnOnAssets',
    'ReturnOnInvestedCapital',
    
    # Capital Allocation Metrics (3)
    'RetainedEarningsToNetIncome',
    'DividendPayoutRatio', 
    'CapitalExpenditureToDepreciation'
]

# Verify we have exactly 22 derived metrics
assert len(DERIVED_METRICS) == 22, f"Expected 22 derived metrics, got {len(DERIVED_METRICS)}"

class SECMetricsBuilder:
    """Main class for building comprehensive SEC financial metrics"""
    
    def __init__(self, rate_limit_delay: float = 0.5, fmp_api_key: Optional[str] = None):
        self.companies = {}
        self.load_companies_database()
        self.api_call_count = 0
        self.failed_companies = []
        self.failure_details = {}  # Track detailed failure reasons
        self.rate_limit_delay = rate_limit_delay  # Default 0.5 seconds between companies
        self.request_delay = 0.1  # 100ms between individual API requests
        
        # Load price cache
        self.load_price_cache()
        
        # Initialize FMP API keys (multiple for redundancy)
        self.fmp_api_keys = []
        for key_name in ['FMP_API_KEY', 'FMP_API_KEY2', 'FMP_API_KEY3']:
            key = fmp_api_key if key_name == 'FMP_API_KEY' and fmp_api_key else os.getenv(key_name)
            if key:
                self.fmp_api_keys.append(key)
        
        global FMP_API_KEY
        FMP_API_KEY = self.fmp_api_keys[0] if self.fmp_api_keys else None
        
        if not self.fmp_api_keys:
            logger.warning("⚠️  No FMP API keys found. Using cached prices only.")
        else:
            logger.info(f"📊 Loaded {len(self.fmp_api_keys)} FMP API keys for redundancy")
        
    def load_price_cache(self):
        """Load cached price database from JSON file"""
        global PRICE_CACHE
        try:
            if os.path.exists(PRICE_CACHE_FILE):
                with open(PRICE_CACHE_FILE, 'r') as f:
                    PRICE_CACHE = json.load(f)
                    logger.info(f"📦 Loaded price cache with {len(PRICE_CACHE)} companies")
            else:
                logger.warning(f"⚠️  Price cache not found: {PRICE_CACHE_FILE}")
                PRICE_CACHE = {}
        except Exception as e:
            logger.error(f"Failed to load price cache: {e}")
            PRICE_CACHE = {}
    
    def get_cached_price(self, ticker: str, year: int, price_type: str) -> Optional[float]:
        """Get cached price for ticker and year"""
        if not PRICE_CACHE:
            return None
        
        ticker_data = PRICE_CACHE.get(ticker)
        if not ticker_data:
            return None
            
        year_data = ticker_data.get('prices', {}).get(str(year))
        if not year_data:
            return None
            
        return year_data.get(price_type)

    def load_companies_database(self):
        """Load companies from sec_companies.json"""
        try:
            with open(SEC_COMPANIES_FILE, 'r') as f:
                data = json.load(f)
                self.companies = data['companies']
                logger.info(f"✅ Loaded {len(self.companies)} companies from {SEC_COMPANIES_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to load {SEC_COMPANIES_FILE}: {e}")
            raise
    
    def filter_companies_by_etf_status(self, tickers: List[str], skip_etf: bool = False, etf_only: bool = False) -> List[str]:
        """Filter companies based on ETF status"""
        if not skip_etf and not etf_only:
            return tickers  # No filtering
        
        filtered_tickers = []
        etf_count = 0
        non_etf_count = 0
        
        for ticker in tickers:
            if ticker in self.companies:
                company_title = self.companies[ticker].get('title', '').upper()
                # Check for common ETF/Fund indicators
                is_etf = (
                    'ETF' in company_title or
                    ('TRUST' in company_title and any(word in company_title for word in ['QQQ', 'SPY', 'VTI', 'SPDR', 'INVESCO'])) or
                    'FUND' in company_title or
                    'INDEX' in company_title
                )
                
                if is_etf:
                    etf_count += 1
                    if etf_only:
                        filtered_tickers.append(ticker)
                else:
                    non_etf_count += 1
                    if skip_etf:
                        filtered_tickers.append(ticker)
            else:
                # If ticker not found in database, include it (assume non-ETF)
                if skip_etf or not etf_only:
                    filtered_tickers.append(ticker)
                    non_etf_count += 1
        
        # Log filtering results
        if skip_etf:
            logger.info(f"🚫 ETF filtering: Skipped {etf_count} ETFs, processing {len(filtered_tickers)} non-ETF companies")
        elif etf_only:
            logger.info(f"📈 ETF filtering: Found {etf_count} ETFs, skipping {non_etf_count} non-ETF companies")
        
        return filtered_tickers
    
    def get_company_facts(self, cik: str) -> Optional[Dict]:
        """Get company facts from SEC API with respectful rate limiting"""
        try:
            url = f"{SEC_BASE_URL}/companyfacts/CIK{cik}.json"
            
            # Rate limiting before each request
            time.sleep(self.request_delay)
            
            response = session.get(url, timeout=30)
            
            if response.status_code == 429:  # Rate limited
                logger.warning(f"⚠️  Rate limited for CIK {cik}, waiting 3 seconds...")
                time.sleep(3)
                response = session.get(url, timeout=30)
            
            response.raise_for_status()
            self.api_call_count += 1
            
            return response.json(), None
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.warning(f"⚠️  No SEC data found for CIK {cik}")
                return None, "404_NOT_FOUND"
            else:
                logger.error(f"❌ HTTP error for CIK {cik}: {e}")
                return None, f"HTTP_ERROR_{response.status_code}"
        except Exception as e:
            logger.error(f"❌ API error for CIK {cik}: {e}")
            return None, f"API_ERROR_{str(e)[:50]}"
    
    def get_market_price_year_end(self, ticker: str, year: int) -> Optional[float]:
        """Get year-end market price (split-adjusted) - cached first, then FMP API"""
        # Try cache first
        cached_price = self.get_cached_price(ticker, year, 'MarketPriceUSD')
        if cached_price is not None:
            logger.debug(f"Using cached split-adjusted price for {ticker} {year}: ${cached_price:.2f}")
            return cached_price
        
        # Fall back to FMP API if available
        if not self.fmp_api_keys:
            return None
        
        # Try each API key until one works
        for api_key in self.fmp_api_keys:
            try:
                url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
                params = {
                    'apikey': api_key,
                    'from': f"{year}-12-01",
                    'to': f"{year}-12-31"
                }
                
                time.sleep(0.1)  # Rate limiting
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                if 'historical' in data and data['historical']:
                    # Get the last trading day of the year (closest to Dec 31)
                    last_price_entry = data['historical'][0]  # Data is sorted desc by date
                    price = float(last_price_entry['close'])
                    logger.debug(f"Fetched split-adjusted price for {ticker} {year}: ${price:.2f}")
                    return price
                
            except Exception as e:
                logger.debug(f"FMP API key failed for {ticker} {year}: {e}")
                continue
        
        logger.debug(f"All price sources failed for {ticker} {year}")
        return None
    
    def get_market_price_non_split_adjusted(self, ticker: str, year: int) -> Optional[float]:
        """Get year-end market price (non-split-adjusted/raw) - cached first, then FMP API"""
        # Try cache first
        cached_price = self.get_cached_price(ticker, year, 'MarketPriceNonSplitAdjustedUSD')
        if cached_price is not None:
            logger.debug(f"Using cached non-split-adjusted price for {ticker} {year}: ${cached_price:.2f}")
            return cached_price
        
        # Fall back to FMP API if available
        if not self.fmp_api_keys:
            return None
        
        # Try each API key until one works
        for api_key in self.fmp_api_keys:
            try:
                url = f"{FMP_BASE_URL}/historical-price-eod/non-split-adjusted"
                params = {
                    'apikey': api_key,
                    'symbol': ticker,
                    'from': f"{year}-12-01",
                    'to': f"{year}-12-31"
                }
                
                time.sleep(0.1)  # Rate limiting
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                if 'historical' in data and data['historical']:
                    # Get the last trading day of the year (closest to Dec 31)
                    last_price_entry = data['historical'][0]  # Data is sorted desc by date
                    price = float(last_price_entry['close'])
                    logger.debug(f"Fetched non-split-adjusted price for {ticker} {year}: ${price:.2f}")
                    return price
                
            except Exception as e:
                logger.debug(f"FMP API key failed for {ticker} {year}: {e}")
                continue
        
        logger.debug(f"All price sources failed for {ticker} {year}")
        return None
    
    def calculate_eps(self, net_income: Optional[float], shares_basic: Optional[float], 
                     shares_diluted: Optional[float], shares_outstanding: Optional[float]) -> Optional[float]:
        """Calculate Earnings Per Share (EPS)"""
        if net_income is None:
            return None
            
        # Primary: Weighted average basic shares (GAAP standard)
        if shares_basic and shares_basic > 0:
            return net_income / shares_basic
        # Fallback hierarchy: Diluted → Outstanding shares
        elif shares_diluted and shares_diluted > 0:
            return net_income / shares_diluted
        elif shares_outstanding and shares_outstanding > 0:
            return net_income / shares_outstanding
        return None
    
    def calculate_pe_ratio(self, market_price_non_split: Optional[float], eps: Optional[float]) -> Optional[float]:
        """Calculate Price-to-Earnings (P/E) ratio"""
        if market_price_non_split is None or eps is None:
            return None
        if eps > 0:  # Positive earnings only
            pe = market_price_non_split / eps
            return pe if pe <= 10000 else 10000  # Cap extreme values
        return None  # Negative earnings = undefined P/E
    
    def calculate_book_value_per_share(self, stockholders_equity: Optional[float], 
                                     shares_outstanding: Optional[float]) -> Optional[float]:
        """Calculate Book Value Per Share"""
        if stockholders_equity is None or shares_outstanding is None:
            return None
        if shares_outstanding > 0:
            return stockholders_equity / shares_outstanding
        return None
    
    def calculate_pb_ratio(self, market_price_non_split: Optional[float], 
                          book_value_per_share: Optional[float]) -> Optional[float]:
        """Calculate Price-to-Book (P/B) ratio"""
        if market_price_non_split is None or book_value_per_share is None:
            return None
        if book_value_per_share > 0:  # Positive book value only
            pb = market_price_non_split / book_value_per_share
            return pb if pb <= 1000 else 1000  # Cap extreme values
        return None  # Negative book value = undefined P/B

    def get_best_revenue_metric(self, year_data: Dict, year: int) -> tuple[Optional[float], Optional[str]]:
        """Get best revenue metric based on accounting standards"""
        revenues = year_data.get('Revenues')
        revenue_contracts = year_data.get('RevenueFromContracts')
        sales_net = year_data.get('SalesRevenueNet')
        
        # Post-2018: Prefer ASC 606 compliant metrics
        if year >= 2018:
            if revenue_contracts is not None:
                return revenue_contracts, 'RevenueFromContracts'
            elif revenues is not None:
                return revenues, 'Revenues'
            elif sales_net is not None:
                return sales_net, 'SalesRevenueNet'
        # Pre-2018: Prefer legacy metrics
        else:
            if revenues is not None:
                return revenues, 'Revenues'
            elif revenue_contracts is not None:
                return revenue_contracts, 'RevenueFromContracts'
            elif sales_net is not None:
                return sales_net, 'SalesRevenueNet'
        return None, None
    
    def calculate_revenue_growth_rate(self, current_year_data: Dict, prior_year_data: Dict, 
                                    current_year: int, prior_year: int) -> Optional[float]:
        """Calculate Revenue Growth Rate"""
        current_revenue, current_source = self.get_best_revenue_metric(current_year_data, current_year)
        prior_revenue, prior_source = self.get_best_revenue_metric(prior_year_data, prior_year)
        
        if current_revenue is not None and prior_revenue is not None and prior_revenue != 0:
            return (current_revenue - prior_revenue) / abs(prior_revenue) * 100
        return None
    
    def calculate_net_income_growth_rate(self, current_net_income: Optional[float], 
                                       prior_net_income: Optional[float]) -> Optional[float]:
        """Calculate Net Income Growth Rate"""
        if current_net_income is not None and prior_net_income is not None and prior_net_income != 0:
            return (current_net_income - prior_net_income) / abs(prior_net_income) * 100
        return None
    
    def calculate_book_value_growth_rate(self, current_stockholders_equity: Optional[float], 
                                       prior_stockholders_equity: Optional[float]) -> Optional[float]:
        """Calculate Book Value Growth Rate"""
        if (current_stockholders_equity is not None and prior_stockholders_equity is not None 
            and prior_stockholders_equity != 0):
            return (current_stockholders_equity - prior_stockholders_equity) / abs(prior_stockholders_equity) * 100
        return None

    def calculate_gross_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Gross Margin"""
        gross_profit = year_data.get('GrossProfit')
        revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
        
        if gross_profit is not None and revenue is not None and revenue > 0:
            return (gross_profit / revenue) * 100
        return None
    
    def calculate_operating_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Operating Margin"""
        operating_income = year_data.get('OperatingIncomeLoss')
        revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
        
        if operating_income is not None and revenue is not None and revenue > 0:
            return (operating_income / revenue) * 100
        return None
    
    def calculate_net_income_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Net Income Margin"""
        net_income = year_data.get('NetIncomeLoss')
        revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
        
        if net_income is not None and revenue is not None and revenue > 0:
            return (net_income / revenue) * 100
        return None
    
    def calculate_free_cash_flow_margin(self, year_data: Dict, year: int) -> Optional[float]:
        """Calculate Free Cash Flow Margin"""
        operating_cash_flow = year_data.get('NetCashFromOperatingActivities')
        capex = year_data.get('CapitalExpenditures')
        
        if operating_cash_flow is not None:
            free_cash_flow = operating_cash_flow - (capex or 0)
            revenue, revenue_source = self.get_best_revenue_metric(year_data, year)
            
            if revenue is not None and revenue > 0:
                return (free_cash_flow / revenue) * 100
        return None

    def calculate_free_cash_flow(self, operating_cash_flow: Optional[float], 
                               capex: Optional[float]) -> Optional[float]:
        """Calculate Free Cash Flow"""
        if operating_cash_flow is not None:
            capex_amount = capex if capex is not None else 0
            return operating_cash_flow - capex_amount
        return None
    
    def calculate_owner_earnings(self, year_data: Dict) -> Optional[float]:
        """Calculate Owner Earnings (Buffett's preferred metric)"""
        net_income = year_data.get('NetIncomeLoss')
        depreciation = year_data.get('DepreciationAndAmortization')
        capex = year_data.get('CapitalExpenditures')
        
        # Working capital changes
        change_ar = year_data.get('ChangeInAccountsReceivable', 0)
        change_inv = year_data.get('ChangeInInventories', 0)
        change_ap = year_data.get('ChangeInAccountsPayable', 0)
        
        if net_income is not None:
            owner_earnings = net_income
            if depreciation is not None:
                owner_earnings += depreciation
            if capex is not None:
                owner_earnings -= capex
            
            working_capital_change = (change_ar or 0) + (change_inv or 0) - (change_ap or 0)
            owner_earnings -= working_capital_change
            
            return owner_earnings
        return None

    def calculate_current_ratio(self, current_assets: Optional[float], 
                              current_liabilities: Optional[float]) -> Optional[float]:
        """Calculate Current Ratio"""
        if current_assets is None:
            return None
        if current_liabilities is None or current_liabilities == 0:
            return 999.99 if current_assets > 0 else None  # Infinite liquidity
        if current_liabilities > 0:
            ratio = current_assets / current_liabilities
            return min(ratio, 999.99)  # Cap at reasonable maximum
        return None
    
    def calculate_debt_to_equity_ratio(self, debt_current: Optional[float], 
                                     debt_noncurrent: Optional[float], 
                                     stockholders_equity: Optional[float]) -> Optional[float]:
        """Calculate Debt-to-Equity Ratio"""
        total_debt = (debt_current or 0) + (debt_noncurrent or 0)
        
        if stockholders_equity and stockholders_equity > 0:
            return total_debt / stockholders_equity
        return None  # Negative equity makes ratio meaningless
    
    def calculate_interest_coverage_ratio(self, operating_income: Optional[float], 
                                        interest_expense: Optional[float]) -> Optional[float]:
        """Calculate Interest Coverage Ratio"""
        if operating_income is None:
            return None
        if interest_expense and interest_expense > 0:
            return operating_income / interest_expense
        elif interest_expense == 0 and operating_income:
            return 999.99  # No interest expense = infinite coverage
        return None

    def calculate_roe(self, net_income: Optional[float], 
                     stockholders_equity: Optional[float]) -> Optional[float]:
        """Calculate Return on Equity (ROE)"""
        if net_income is None or stockholders_equity is None:
            return None
        if stockholders_equity > 0:
            return (net_income / stockholders_equity) * 100
        return None  # Negative equity makes ROE not meaningful
    
    def calculate_roa(self, net_income: Optional[float], 
                     total_assets: Optional[float]) -> Optional[float]:
        """Calculate Return on Assets (ROA)"""
        if net_income is None or total_assets is None:
            return None
        if total_assets > 0:
            return (net_income / total_assets) * 100
        return None
    
    def calculate_roic(self, net_income: Optional[float], interest_expense: Optional[float], 
                      tax_expense: Optional[float], stockholders_equity: Optional[float], 
                      debt_current: Optional[float], debt_noncurrent: Optional[float]) -> Optional[float]:
        """Calculate Return on Invested Capital (ROIC)"""
        if net_income is None:
            return None
            
        # Estimate tax rate
        if tax_expense is not None and net_income is not None:
            pre_tax_income = net_income + tax_expense
            if pre_tax_income > 0:
                tax_rate = min(tax_expense / pre_tax_income, 0.50)
            else:
                tax_rate = 0.25
        else:
            tax_rate = 0.25  # Default assumption
        
        # Calculate NOPAT (Net Operating Profit After Tax)
        interest_tax_shield = (interest_expense or 0) * (1 - tax_rate)
        nopat = (net_income or 0) + interest_tax_shield
        
        # Calculate invested capital
        total_debt = (debt_current or 0) + (debt_noncurrent or 0)
        invested_capital = (stockholders_equity or 0) + total_debt
        
        if invested_capital > 0:
            return (nopat / invested_capital) * 100
        return None

    def calculate_earnings_retention_efficiency(self, retained_earnings: Optional[float], 
                                              current_net_income: Optional[float]) -> Optional[float]:
        """Calculate Retained Earnings to Net Income ratio"""
        if retained_earnings is None or current_net_income is None:
            return None
        if current_net_income != 0:
            return (retained_earnings / abs(current_net_income)) * 100
        return None
    
    def calculate_dividend_payout_ratio(self, dividends_paid: Optional[float], 
                                      net_income: Optional[float]) -> Optional[float]:
        """Calculate Dividend Payout Ratio"""
        if net_income is None or net_income <= 0:
            return 0  # No positive earnings to pay dividends from
        if dividends_paid is None:
            return 0  # No dividends paid
        return abs(dividends_paid) / net_income * 100  # Dividends usually negative in cash flow
    
    def calculate_capex_to_depreciation_ratio(self, capex: Optional[float], 
                                            depreciation: Optional[float]) -> Optional[float]:
        """Calculate Capital Expenditure to Depreciation ratio"""
        if depreciation is None or depreciation <= 0:
            return None
        if capex is None:
            return 0  # No capex
        return abs(capex) / depreciation  # Capex usually negative in cash flow

    def calculate_derived_metrics(self, current_year_data: Dict, prior_year_data: Optional[Dict], 
                                year: int) -> Dict[str, Optional[float]]:
        """Calculate all 22 derived Warren Buffett-style metrics"""
        derived = {}
        
        # 1. EarningsPerShare
        derived['EarningsPerShare'] = self.calculate_eps(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('WeightedAverageSharesBasic'),
            current_year_data.get('WeightedAverageSharesDiluted'),
            current_year_data.get('CommonStockSharesOutstanding')
        )
        
        # 2. PriceToEarning
        derived['PriceToEarning'] = self.calculate_pe_ratio(
            current_year_data.get('MarketPriceNonSplitAdjustedUSD'),
            derived['EarningsPerShare']
        )
        
        # 3. BookValuePerShare
        derived['BookValuePerShare'] = self.calculate_book_value_per_share(
            current_year_data.get('StockholdersEquity'),
            current_year_data.get('CommonStockSharesOutstanding')
        )
        
        # 4. PriceToBook
        derived['PriceToBook'] = self.calculate_pb_ratio(
            current_year_data.get('MarketPriceNonSplitAdjustedUSD'),
            derived['BookValuePerShare']
        )
        
        # Growth metrics (require prior year data)
        if prior_year_data:
            # 5. RevenueGrowthRate
            derived['RevenueGrowthRate'] = self.calculate_revenue_growth_rate(
                current_year_data, prior_year_data, year, year - 1
            )
            
            # 6. NetIncomeGrowthRate
            derived['NetIncomeGrowthRate'] = self.calculate_net_income_growth_rate(
                current_year_data.get('NetIncomeLoss'),
                prior_year_data.get('NetIncomeLoss')
            )
            
            # 7. BookValueGrowthRate
            derived['BookValueGrowthRate'] = self.calculate_book_value_growth_rate(
                current_year_data.get('StockholdersEquity'),
                prior_year_data.get('StockholdersEquity')
            )
        else:
            derived['RevenueGrowthRate'] = None
            derived['NetIncomeGrowthRate'] = None
            derived['BookValueGrowthRate'] = None
        
        # 8-11. Profitability Margins
        derived['GrossMargin'] = self.calculate_gross_margin(current_year_data, year)
        derived['OperatingMargin'] = self.calculate_operating_margin(current_year_data, year)
        derived['NetIncomeMargin'] = self.calculate_net_income_margin(current_year_data, year)
        derived['FreeCashFlowMargin'] = self.calculate_free_cash_flow_margin(current_year_data, year)
        
        # 12-13. Cash Flow Metrics
        derived['FreeCashFlow'] = self.calculate_free_cash_flow(
            current_year_data.get('NetCashFromOperatingActivities'),
            current_year_data.get('CapitalExpenditures')
        )
        derived['OwnerEarnings'] = self.calculate_owner_earnings(current_year_data)
        
        # 14-16. Financial Health Ratios
        derived['CurrentRatio'] = self.calculate_current_ratio(
            current_year_data.get('AssetsCurrent'),
            current_year_data.get('LiabilitiesCurrent')
        )
        derived['DebtToEquityRatio'] = self.calculate_debt_to_equity_ratio(
            current_year_data.get('DebtCurrent'),
            current_year_data.get('DebtNoncurrent'),
            current_year_data.get('StockholdersEquity')
        )
        derived['InterestCoverageRatio'] = self.calculate_interest_coverage_ratio(
            current_year_data.get('OperatingIncomeLoss'),
            current_year_data.get('InterestExpense')
        )
        
        # 17-19. Return Metrics
        derived['ReturnOnEquity'] = self.calculate_roe(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('StockholdersEquity')
        )
        derived['ReturnOnAssets'] = self.calculate_roa(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('Assets')
        )
        derived['ReturnOnInvestedCapital'] = self.calculate_roic(
            current_year_data.get('NetIncomeLoss'),
            current_year_data.get('InterestExpense'),
            current_year_data.get('IncomeTaxExpenseBenefit'),
            current_year_data.get('StockholdersEquity'),
            current_year_data.get('DebtCurrent'),
            current_year_data.get('DebtNoncurrent')
        )
        
        # 20-22. Capital Allocation Metrics
        derived['RetainedEarningsToNetIncome'] = self.calculate_earnings_retention_efficiency(
            current_year_data.get('RetainedEarnings'),
            current_year_data.get('NetIncomeLoss')
        )
        derived['DividendPayoutRatio'] = self.calculate_dividend_payout_ratio(
            current_year_data.get('CommonDividendsPaid'),
            current_year_data.get('NetIncomeLoss')
        )
        derived['CapitalExpenditureToDepreciation'] = self.calculate_capex_to_depreciation_ratio(
            current_year_data.get('CapitalExpenditures'),
            current_year_data.get('DepreciationAndAmortization')
        )
        
        return derived

    def extract_metric_value(self, facts: Dict, metric_name: str, year: int, ticker: str = None) -> Optional[float]:
        """Extract a specific metric value for a given year"""
        try:
            # Handle market price metrics from external API
            if metric_name == 'MarketPriceUSD':
                return self.get_market_price_year_end(ticker, year) if ticker else None
            elif metric_name == 'MarketPriceNonSplitAdjustedUSD':
                return self.get_market_price_non_split_adjusted(ticker, year) if ticker else None
            
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            
            # Get all possible XBRL tags for this metric
            possible_tags = SEC_METRICS.get(metric_name, [])
            
            for tag in possible_tags:
                if tag in us_gaap:
                    units = us_gaap[tag].get('units', {})
                    
                    # Try USD first, then shares, then pure numbers
                    for unit_type in ['USD', 'shares', 'pure']:
                        if unit_type in units:
                            entries = units[unit_type]
                            
                            # Find best entry for the target year
                            best_value = None
                            best_form_priority = 0
                            
                            for entry in entries:
                                entry_date = entry.get('end', '')
                                entry_form = entry.get('form', '')
                                entry_fy = entry.get('fy')
                                
                                # Check if this entry is for our target year
                                if entry_date and str(year) in entry_date:
                                    # Prefer 10-K forms over 10-Q
                                    form_priority = 3 if entry_form == '10-K' else (2 if entry_form == '10-Q' else 1)
                                    
                                    # Also check fiscal year alignment
                                    fy_matches = (entry_fy == year) if entry_fy else True
                                    
                                    # Take best entry based on form priority and fiscal year match
                                    if (form_priority > best_form_priority) or (form_priority == best_form_priority and fy_matches):
                                        best_value = entry.get('val')
                                        best_form_priority = form_priority
                            
                            if best_value is not None:
                                return float(best_value)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting {metric_name} for {year}: {e}")
            return None
    
    def extract_all_metrics(self, facts: Dict, years: List[int], ticker: str = None) -> Dict[int, Dict[str, Optional[float]]]:
        """Extract all metrics for all specified years including derived metrics"""
        results = {}
        sorted_years = sorted(years)
        
        for i, year in enumerate(sorted_years):
            year_data = {}
            
            # Extract each SEC metric using SEC_METRICS keys in order
            for metric_name in SEC_METRICS.keys():
                value = self.extract_metric_value(facts, metric_name, year, ticker)
                year_data[metric_name] = value
            
            # Validate balance sheet equation if we have the core components
            assets = year_data.get('Assets')
            liabilities = year_data.get('Liabilities')  
            equity = year_data.get('StockholdersEquity')
            
            if assets and liabilities and equity:
                balance_diff = abs(assets - (liabilities + equity))
                tolerance = assets * 0.05  # 5% tolerance
                
                if balance_diff > tolerance:
                    logger.warning(f"⚠️  Balance sheet equation violation for {year}: "
                                 f"Assets={assets:,.0f}, L+E={liabilities + equity:,.0f}, "
                                 f"diff={balance_diff:,.0f}")
                    
                    # If liabilities suspiciously equal assets, recalculate
                    if abs(liabilities - assets) < (assets * 0.01):
                        logger.info(f"  🔧 Correcting liabilities: Assets - Equity")
                        year_data['Liabilities'] = assets - equity
            
            # Calculate derived Warren Buffett-style metrics
            prior_year_data = results.get(year - 1) if i > 0 else None
            derived_metrics = self.calculate_derived_metrics(year_data, prior_year_data, year)
            
            # Add derived metrics to year_data
            year_data.update(derived_metrics)
            
            results[year] = year_data
        
        return results
    
    def create_company_csv(self, ticker: str, company_data: Dict, output_dir: str) -> bool:
        """Create CSV file for a single company"""
        try:
            # Prepare data for CSV
            rows = []
            years = sorted(company_data.keys())
            
            # Add header row using SEC_METRICS keys + DERIVED_METRICS in order
            header = ['Year'] + list(SEC_METRICS.keys()) + DERIVED_METRICS
            rows.append(header)
            
            # Add data rows
            for year in years:
                year_metrics = company_data[year]
                row = [year]
                
                # Add SEC metrics
                for metric_name in SEC_METRICS.keys():
                    value = year_metrics.get(metric_name)
                    # Format values appropriately
                    if value is not None:
                        if abs(value) >= 1e6:  # Large numbers in millions
                            row.append(f"{value/1e6:.2f}")
                        elif abs(value) >= 1000:  # Thousands
                            row.append(f"{value:.0f}")
                        else:  # Small numbers with decimals
                            row.append(f"{value:.4f}")
                    else:
                        row.append('')  # Empty for missing values
                
                # Add derived metrics
                for metric_name in DERIVED_METRICS:
                    value = year_metrics.get(metric_name)
                    # Format derived metrics (mostly ratios and percentages)
                    if value is not None:
                        if metric_name in ['EarningsPerShare', 'BookValuePerShare']:
                            row.append(f"{value:.4f}")  # Per-share values with precision
                        elif metric_name.endswith('Ratio') or metric_name.endswith('Rate') or metric_name.endswith('Margin'):
                            row.append(f"{value:.2f}")  # Ratios and percentages with 2 decimals
                        elif abs(value) >= 1e6:  # Large absolute numbers
                            row.append(f"{value/1e6:.2f}")
                        else:
                            row.append(f"{value:.2f}")  # Default 2 decimal places for derived metrics
                    else:
                        row.append('')  # Empty for missing values
                
                rows.append(row)
            
            # Write CSV file
            csv_path = Path(output_dir) / f"{ticker}.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            logger.debug(f"✅ Created CSV: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create CSV for {ticker}: {e}")
            return False
    
    def validate_csv_format(self, csv_path: str) -> bool:
        """Validate CSV file format and structure"""
        try:
            # Read the CSV
            df = pd.read_csv(csv_path)
            
            # Check if we have the expected columns
            expected_columns = ['Year'] + list(SEC_METRICS.keys()) + DERIVED_METRICS
            if list(df.columns) != expected_columns:
                logger.error(f"❌ Column mismatch in {csv_path}")
                logger.error(f"   Expected: {len(expected_columns)} columns ({len(SEC_METRICS)} SEC + {len(DERIVED_METRICS)} derived)")
                logger.error(f"   Found: {len(df.columns)} columns")
                return False
            
            # Check if we have reasonable year range
            years = df['Year'].tolist()
            if not years or min(years) < 2015 or max(years) > 2025:
                logger.warning(f"⚠️  Unusual year range in {csv_path}: {min(years) if years else 'None'}-{max(years) if years else 'None'}")
            
            # Check for completely empty rows
            non_year_columns = [col for col in df.columns if col != 'Year']
            empty_rows = df[non_year_columns].isnull().all(axis=1).sum()
            
            if empty_rows == len(df):
                logger.warning(f"⚠️  All data rows empty in {csv_path}")
                return False
            
            logger.debug(f"✅ CSV validation passed: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ CSV validation failed for {csv_path}: {e}")
            return False
    
    def process_company(self, ticker: str, years: List[int], output_dir: str) -> bool:
        """Process a single company and create its CSV with respectful delays"""
        try:
            if ticker not in self.companies:
                logger.warning(f"⚠️  Ticker {ticker} not found in companies database")
                return False
            
            company_info = self.companies[ticker]
            cik = company_info['cik']
            company_name = company_info['title']
            
            logger.info(f"📊 Processing {ticker} ({company_name}) - CIK: {cik}")
            
            # Respectful delay before processing each company
            if self.api_call_count > 0:  # Skip delay for first company
                logger.debug(f"⏱️  Waiting {self.rate_limit_delay}s before next company...")
                time.sleep(self.rate_limit_delay)
            
            # Get company facts from SEC
            facts, error_reason = self.get_company_facts(cik)
            if not facts:
                logger.warning(f"⚠️  No SEC data available for {ticker}")
                self.failed_companies.append(ticker)
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': error_reason or 'UNKNOWN_ERROR',
                    'error_type': 'SEC_DATA_UNAVAILABLE'
                }
                return False
            
            # Extract all metrics for all years
            company_data = self.extract_all_metrics(facts, years, ticker)
            
            # Check if we got any useful data
            total_values = sum(
                sum(1 for v in year_data.values() if v is not None)
                for year_data in company_data.values()
            )
            
            if total_values == 0:
                logger.warning(f"⚠️  No financial data extracted for {ticker}")
                self.failed_companies.append(ticker)
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': 'NO_FINANCIAL_DATA_EXTRACTED',
                    'error_type': 'DATA_EXTRACTION_FAILED'
                }
                return False
            
            # Create CSV file
            success = self.create_company_csv(ticker, company_data, output_dir)
            if not success:
                self.failed_companies.append(ticker)
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': 'CSV_CREATION_FAILED',
                    'error_type': 'FILE_OPERATION_FAILED'
                }
                return False
            
            # Validate the created CSV
            csv_path = Path(output_dir) / f"{ticker}.csv"
            if not self.validate_csv_format(str(csv_path)):
                logger.warning(f"⚠️  CSV validation failed for {ticker}")
                self.failure_details[ticker] = {
                    'company_name': company_name,
                    'cik': cik,
                    'error_reason': 'CSV_VALIDATION_FAILED',
                    'error_type': 'DATA_VALIDATION_FAILED'
                }
                return False
            
            logger.info(f"✅ Successfully processed {ticker} ({total_values} data points)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process {ticker}: {e}")
            if ticker not in self.failed_companies:
                self.failed_companies.append(ticker)
            self.failure_details[ticker] = {
                'company_name': company_info.get('title', 'UNKNOWN') if 'company_info' in locals() else 'UNKNOWN',
                'cik': company_info.get('cik', 'UNKNOWN') if 'company_info' in locals() else 'UNKNOWN',
                'error_reason': f'EXCEPTION_{str(e)[:50]}',
                'error_type': 'PROCESSING_EXCEPTION'
            }
            return False
    
    def build_metrics(self, tickers: List[str], years: List[int], output_dir: str, 
                     max_companies: Optional[int] = None) -> Dict[str, Any]:
        """Build metrics for multiple companies with respectful rate limiting"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Limit companies if specified
        if max_companies:
            tickers = tickers[:max_companies]
        
        logger.info(f"🚀 Starting SEC metrics extraction")
        logger.info(f"   Companies: {len(tickers)}")
        logger.info(f"   Years: {min(years)}-{max(years)}")
        logger.info(f"   SEC Metrics: 102 (optimized - empty metrics removed)")
        
        # Market price status
        cache_count = len(PRICE_CACHE) if PRICE_CACHE else 0
        fmp_count = len(self.fmp_api_keys) if hasattr(self, 'fmp_api_keys') else 0
        price_status = []
        if cache_count > 0:
            price_status.append(f"cached: {cache_count} companies")
        if fmp_count > 0:
            price_status.append(f"FMP API: {fmp_count} keys")
        status_str = f"({', '.join(price_status)})" if price_status else "(unavailable)"
        
        logger.info(f"   Market Metrics: 2 {status_str}")
        logger.info(f"   Derived Metrics: {len(DERIVED_METRICS)} (Warren Buffett-style)")
        logger.info(f"   Total Metrics: {len(SEC_METRICS) + len(DERIVED_METRICS)} (104 base + 22 derived)")
        logger.info(f"   Rate limit: {self.rate_limit_delay}s between companies")
        logger.info(f"   Output: {output_dir}")
        
        start_time = time.time()
        successful_companies = []
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")
            
            success = self.process_company(ticker, years, output_dir)
            if success:
                successful_companies.append(ticker)
            
            # Progress update every 10 companies
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(tickers) - i) / rate if rate > 0 else 0
                
                logger.info(f"📈 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%) "
                           f"- Rate: {rate:.2f} companies/sec - ETA: {eta/60:.1f} min")
        
        # Create summary
        elapsed = time.time() - start_time
        
        summary = {
            'total_companies': len(tickers),
            'successful_companies': len(successful_companies),
            'failed_companies': len(self.failed_companies),
            'success_rate': len(successful_companies) / len(tickers) * 100,
            'elapsed_time': elapsed,
            'api_calls': self.api_call_count,
            'rate_limit_delay': self.rate_limit_delay,
            'output_directory': output_dir,
            'failed_tickers': self.failed_companies,
            'failure_details': self.failure_details
        }
        
        # Create ZIP file
        zip_path = Path(output_dir) / "sec_financial_metrics.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for ticker in successful_companies:
                csv_path = Path(output_dir) / f"{ticker}.csv"
                if csv_path.exists():
                    zipf.write(csv_path, f"{ticker}.csv")
        
        # Save summary
        summary_path = Path(output_dir) / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed failure report if there are failures
        if self.failed_companies:
            failure_report_path = Path(output_dir) / "failure_report.json"
            failure_report = {
                'extraction_date': datetime.now().isoformat(),
                'total_failed': len(self.failed_companies),
                'failure_breakdown': {},
                'detailed_failures': self.failure_details
            }
            
            # Group failures by error type
            for ticker, details in self.failure_details.items():
                error_type = details['error_type']
                if error_type not in failure_report['failure_breakdown']:
                    failure_report['failure_breakdown'][error_type] = []
                failure_report['failure_breakdown'][error_type].append({
                    'ticker': ticker,
                    'company_name': details['company_name'],
                    'error_reason': details['error_reason']
                })
            
            with open(failure_report_path, 'w') as f:
                json.dump(failure_report, f, indent=2)
            
            logger.info(f"   📋 Failure report: {failure_report_path}")
        
        logger.info(f"\n🎉 EXTRACTION COMPLETE!")
        logger.info(f"   ✅ Successful: {len(successful_companies)}/{len(tickers)} ({summary['success_rate']:.1f}%)")
        logger.info(f"   ❌ Failed: {len(self.failed_companies)}")
        logger.info(f"   ⏱️  Time: {elapsed/60:.1f} minutes")
        logger.info(f"   🌐 API calls: {self.api_call_count}")
        rate = len(tickers) / elapsed if elapsed > 0 else 0
        logger.info(f"   🚀 Rate: {rate:.2f} companies/sec (including delays)")
        logger.info(f"   📦 ZIP file: {zip_path}")
        
        # Create CSV failure summary if there are failures
        if self.failed_companies:
            failure_csv_path = Path(output_dir) / "failure_summary.csv"
            with open(failure_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(['Ticker', 'Company_Name', 'CIK', 'Error_Type', 'Error_Reason'])
                
                # Write failure details
                for ticker in self.failed_companies:
                    if ticker in self.failure_details:
                        details = self.failure_details[ticker]
                        writer.writerow([
                            ticker,
                            details['company_name'],
                            details['cik'],
                            details['error_type'],
                            details['error_reason']
                        ])
                    else:
                        writer.writerow([ticker, 'UNKNOWN', 'UNKNOWN', 'UNKNOWN_FAILURE', 'UNKNOWN_REASON'])
            
            logger.info(f"   📊 Failure summary CSV: {failure_csv_path}")
            
            # Brief console summary
            logger.info(f"\n❌ FAILURE SUMMARY ({len(self.failed_companies)} companies):")
            failure_types = {}
            for ticker in self.failed_companies:
                if ticker in self.failure_details:
                    error_type = self.failure_details[ticker]['error_type']
                    failure_types[error_type] = failure_types.get(error_type, 0) + 1
                else:
                    failure_types['UNKNOWN_FAILURE'] = failure_types.get('UNKNOWN_FAILURE', 0) + 1
            
            for error_type, count in failure_types.items():
                logger.info(f"   {error_type}: {count} companies")
        
        return summary

def main():
    """Main function with enhanced rate limiting options"""
    parser = argparse.ArgumentParser(description='SEC Comprehensive Financial Metrics Builder')
    
    # Company selection
    company_group = parser.add_mutually_exclusive_group(required=True)
    company_group.add_argument('--companies', choices=['all'], 
                              help='Process all companies in database')
    company_group.add_argument('--ticker', type=str, 
                              help='Process single ticker (e.g., AAPL)')
    company_group.add_argument('--tickers', nargs='+', 
                              help='Process multiple tickers')
    company_group.add_argument('--top', type=int, 
                              help='Process top N companies by market cap')
    
    # Year range
    parser.add_argument('--years', nargs=2, type=int, required=True,
                       help='Year range (e.g., 2015 2025)')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='sec_metrics_output',
                       help='Output directory for CSV files')
    
    # Rate limiting - more respectful defaults
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Delay between companies in seconds (default: 0.5s)')
    
    # Max companies (for testing)
    parser.add_argument('--max-companies', type=int,
                       help='Maximum number of companies to process (for testing)')
    
    # ETF filtering options
    etf_group = parser.add_mutually_exclusive_group()
    etf_group.add_argument('--skip-etf', action='store_true',
                          help='Skip ETF companies (companies with "ETF" in title)')
    etf_group.add_argument('--etf-only', action='store_true',
                          help='Process only ETF companies (companies with "ETF" in title)')
    
    args = parser.parse_args()
    
    # Initialize builder with rate limiting
    builder = SECMetricsBuilder(rate_limit_delay=args.rate_limit)
    
    # Determine which companies to process
    if args.companies == 'all':
        tickers = list(builder.companies.keys())
    elif args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.top:
        # For top N, we'll use the first N from our database
        # (in a real implementation, you'd sort by market cap)
        tickers = list(builder.companies.keys())[:args.top]
    
    # Apply ETF filtering
    original_count = len(tickers)
    tickers = builder.filter_companies_by_etf_status(tickers, args.skip_etf, args.etf_only)
    
    if args.skip_etf or args.etf_only:
        logger.info(f"📊 Company filtering: {original_count} → {len(tickers)} companies after ETF filtering")
    
    # Year range
    years = list(range(args.years[0], args.years[1] + 1))
    
    # Respectful API usage warning
    if len(tickers) > 100:
        estimated_time = len(tickers) * args.rate_limit / 60
        logger.info(f"⚠️  Processing {len(tickers)} companies with {args.rate_limit}s delays")
        logger.info(f"   Estimated minimum time: {estimated_time:.1f} minutes")
        logger.info(f"   This ensures respectful SEC API usage")
    
    # Build metrics
    summary = builder.build_metrics(
        tickers=tickers,
        years=years,
        output_dir=args.output_dir,
        max_companies=args.max_companies
    )
    
    # Exit with appropriate code
    if summary['success_rate'] < 50:
        logger.error("❌ Less than 50% success rate - check API connectivity")
        exit(1)
    else:
        logger.info("✅ Metrics extraction completed successfully")
        exit(0)

if __name__ == "__main__":
    main()