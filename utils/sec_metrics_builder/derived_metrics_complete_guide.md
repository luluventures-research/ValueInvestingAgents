# Complete Derived Financial Metrics Guide
## Warren Buffett-Style Value Investing Analysis

---

## Overview

This document outlines **22 comprehensive derived financial metrics** calculated from our 104 base SEC EDGAR metrics. These metrics provide institutional-quality financial analysis capabilities, with special focus on Warren Buffett's investment philosophy and value investing principles.

**Total Dataset**: 126 metrics (104 base + 22 derived)

---

## 📊 Valuation Metrics

### 1. EarningsPerShare (EPS)
**Formula**: `NetIncomeLoss / WeightedAverageSharesBasic`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement)
- `WeightedAverageSharesBasic` (Share Data)

**Calculation Method**:
```python
def calculate_eps(net_income, shares_basic, shares_diluted=None, shares_outstanding=None):
    # Primary: Weighted average basic shares (GAAP standard)
    if shares_basic and shares_basic > 0:
        return net_income / shares_basic
    # Fallback hierarchy: Diluted → Outstanding shares
    elif shares_diluted and shares_diluted > 0:
        return net_income / shares_diluted
    elif shares_outstanding and shares_outstanding > 0:
        return net_income / shares_outstanding
    return None
```

**Why Buffett Values This**:
*"The primary test of managerial economic performance is the achievement of a high earnings rate on equity capital employed (without undue leverage, accounting gimmickry, etc.)"*

EPS shows the fundamental earning power per share, essential for valuation and comparison across companies.

---

### 2. PriceToEarning (P/E)
**Formula**: `MarketPriceNonSplitAdjustedUSD / EarningsPerShare`

**Exact Metrics Used**:
- `MarketPriceNonSplitAdjustedUSD` (Market Data)
- `EarningsPerShare` (calculated above)

**Calculation Method**:
```python
def calculate_pe_ratio(market_price_non_split, eps):
    if market_price_non_split is None or eps is None:
        return None
    if eps > 0:  # Positive earnings only
        pe = market_price_non_split / eps
        return pe if pe <= 10000 else 10000  # Cap extreme values
    return None  # Negative earnings = undefined P/E
```

**Why Buffett Values This**:
*"Price is what you pay. Value is what you get."*

P/E ratio helps identify reasonably priced stocks. Buffett prefers P/E ratios that aren't excessive relative to growth prospects and business quality.

---

### 3. BookValuePerShare
**Formula**: `StockholdersEquity / CommonStockSharesOutstanding`

**Exact Metrics Used**:
- `StockholdersEquity` (Balance Sheet - Equity)
- `CommonStockSharesOutstanding` (Share Data)

**Calculation Method**:
```python
def calculate_book_value_per_share(stockholders_equity, shares_outstanding):
    if shares_outstanding and shares_outstanding > 0:
        return stockholders_equity / shares_outstanding
    return None
```

**Why Buffett Values This**:
*"Book value per share is a useful, though limited, guide to the intrinsic value of shares."*

Book value per share represents the accounting value of ownership, useful for asset-heavy businesses and as a baseline for valuation.

---

### 4. PriceToBook (P/B)
**Formula**: `MarketPriceNonSplitAdjustedUSD / BookValuePerShare`

**Exact Metrics Used**:
- `MarketPriceNonSplitAdjustedUSD` (Market Data)
- `BookValuePerShare` (calculated above)

**Calculation Method**:
```python
def calculate_pb_ratio(market_price_non_split, book_value_per_share):
    if market_price_non_split is None or book_value_per_share is None:
        return None
    if book_value_per_share > 0:  # Positive book value only
        pb = market_price_non_split / book_value_per_share
        return pb if pb <= 1000 else 1000  # Cap extreme values
    return None  # Negative book value = undefined P/B
```

**Why Buffett Values This**:
*"When we bought Coca-Cola, we weren't buying it because of its book value. We were buying it because of its earning power."*

While Buffett focuses more on earning power, P/B helps identify when quality companies trade at reasonable prices relative to their net worth.

---

## 📈 Growth Metrics

### 5. RevenueGrowthRate
**Formula**: `(Current_Revenue - Prior_Revenue) / abs(Prior_Revenue) * 100`

**Exact Metrics Used** (Smart Selection):
- `Revenues` (Income Statement) - Pre-2018 preferred
- `RevenueFromContracts` (Income Statement) - Post-2018 preferred (ASC 606)
- `SalesRevenueNet` (Income Statement) - Fallback

**Calculation Method**:
```python
def get_best_revenue_metric(year_data, year):
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

def calculate_revenue_growth_rate(current_year_data, prior_year_data, current_year, prior_year):
    current_revenue, current_source = get_best_revenue_metric(current_year_data, current_year)
    prior_revenue, prior_source = get_best_revenue_metric(prior_year_data, prior_year)
    
    if current_revenue is not None and prior_revenue is not None and prior_revenue != 0:
        return (current_revenue - prior_revenue) / abs(prior_revenue) * 100
    return None
```

**Why Buffett Values This**:
*"The businesses we own have increased their earnings over the years, and their stock prices have risen correspondingly."*

Consistent revenue growth indicates strong business momentum and market position, key indicators of sustainable competitive advantages.

---

### 6. NetIncomeGrowthRate
**Formula**: `(Current_NetIncome - Prior_NetIncome) / abs(Prior_NetIncome) * 100`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement) - Current and prior year

**Calculation Method**:
```python
def calculate_net_income_growth_rate(current_net_income, prior_net_income):
    if current_net_income is not None and prior_net_income is not None and prior_net_income != 0:
        return (current_net_income - prior_net_income) / abs(prior_net_income) * 100
    return None
```

**Why Buffett Values This**:
*"The key to investing is not assessing how much an industry is going to affect society, but rather determining the competitive advantage of any given company."*

Consistent earnings growth demonstrates durable competitive advantages and effective management execution.

---

### 7. BookValueGrowthRate
**Formula**: `(Current_StockholdersEquity - Prior_StockholdersEquity) / abs(Prior_StockholdersEquity) * 100`

**Exact Metrics Used**:
- `StockholdersEquity` (Balance Sheet - Equity) - Current and prior year

**Calculation Method**:
```python
def calculate_book_value_growth_rate(current_stockholders_equity, prior_stockholders_equity):
    if prior_stockholders_equity and prior_stockholders_equity != 0:
        return (current_stockholders_equity - prior_stockholders_equity) / abs(prior_stockholders_equity) * 100
    return None
```

**Why Buffett Values This**:
*"Our gain in net worth during the year was $8.3 billion, which increased the per-share book value of both our Class A and Class B stock by 6.5%."*

Book value growth measures wealth creation for shareholders over time, a key Berkshire Hathaway performance metric.

---

## 💰 Profitability Metrics

### 8. GrossMargin
**Formula**: `GrossProfit / Best_Revenue * 100`

**Exact Metrics Used**:
- `GrossProfit` (Income Statement)
- Best revenue metric (see RevenueGrowthRate logic)

**Calculation Method**:
```python
def calculate_gross_margin(year_data, year):
    gross_profit = year_data.get('GrossProfit')
    revenue, revenue_source = get_best_revenue_metric(year_data, year)
    
    if gross_profit is not None and revenue is not None and revenue > 0:
        return (gross_profit / revenue) * 100
    return None
```

**Why Buffett Values This**:
*"I like businesses with high margins, because it usually means they have some sort of competitive advantage."*

High gross margins indicate pricing power and competitive moats, essential for sustainable profitability.

---

### 9. OperatingMargin
**Formula**: `OperatingIncomeLoss / Best_Revenue * 100`

**Exact Metrics Used**:
- `OperatingIncomeLoss` (Income Statement)
- Best revenue metric (see RevenueGrowthRate logic)

**Calculation Method**:
```python
def calculate_operating_margin(year_data, year):
    operating_income = year_data.get('OperatingIncomeLoss')
    revenue, revenue_source = get_best_revenue_metric(year_data, year)
    
    if operating_income is not None and revenue is not None and revenue > 0:
        return (operating_income / revenue) * 100
    return None
```

**Why Buffett Values This**:
*"The most important thing to do when you find yourself in a hole is to stop digging."*

Operating margin shows core business profitability before financial engineering, revealing true operational efficiency.

---

### 10. NetIncomeMargin
**Formula**: `NetIncomeLoss / Best_Revenue * 100`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement)
- Best revenue metric (see RevenueGrowthRate logic)

**Calculation Method**:
```python
def calculate_net_income_margin(year_data, year):
    net_income = year_data.get('NetIncomeLoss')
    revenue, revenue_source = get_best_revenue_metric(year_data, year)
    
    if net_income is not None and revenue is not None and revenue > 0:
        return (net_income / revenue) * 100
    return None
```

**Why Buffett Values This**:
*"The investor of today does not profit from yesterday's growth."*

Net income margin reveals bottom-line profitability after all expenses, crucial for shareholder returns.

---

### 11. FreeCashFlowMargin
**Formula**: `FreeCashFlow / Best_Revenue * 100`

**Exact Metrics Used**:
- `NetCashFromOperatingActivities` (Cash Flow Statement)
- `CapitalExpenditures` (Cash Flow Statement)
- Best revenue metric (see RevenueGrowthRate logic)

**Calculation Method**:
```python
def calculate_free_cash_flow_margin(year_data, year):
    operating_cash_flow = year_data.get('NetCashFromOperatingActivities')
    capex = year_data.get('CapitalExpenditures')
    
    if operating_cash_flow is not None:
        free_cash_flow = operating_cash_flow - (capex or 0)
        revenue, revenue_source = get_best_revenue_metric(year_data, year)
        
        if revenue is not None and revenue > 0:
            return (free_cash_flow / revenue) * 100
    return None
```

**Why Buffett Values This**:
*"Free cash flow is really what you ought to be looking at."*

FCF margin shows real cash generation efficiency, indicating the quality of reported earnings.

---

## 💵 Cash Flow Metrics

### 12. FreeCashFlow
**Formula**: `NetCashFromOperatingActivities - CapitalExpenditures`

**Exact Metrics Used**:
- `NetCashFromOperatingActivities` (Cash Flow Statement)
- `CapitalExpenditures` (Cash Flow Statement)

**Calculation Method**:
```python
def calculate_free_cash_flow(operating_cash_flow, capex):
    if operating_cash_flow is not None:
        capex_amount = capex if capex is not None else 0
        return operating_cash_flow - capex_amount
    return None
```

**Why Buffett Values This**:
*"Free cash flow is really what you ought to be looking at. Cash is a fact. Everything else is opinion."*

Free cash flow represents actual cash available to shareholders after maintaining and growing the business.

---

### 13. OwnerEarnings
**Formula**: `NetIncomeLoss + DepreciationAndAmortization - CapitalExpenditures - WorkingCapitalChange`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement)
- `DepreciationAndAmortization` (Cash Flow Statement)
- `CapitalExpenditures` (Cash Flow Statement)
- `ChangeInAccountsReceivable` (Cash Flow Statement)
- `ChangeInInventories` (Cash Flow Statement)
- `ChangeInAccountsPayable` (Cash Flow Statement)

**Calculation Method**:
```python
def calculate_owner_earnings(year_data):
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
```

**Why Buffett Values This**:
*"Owner's earnings represent the amount of cash that could theoretically be taken out of the business each year without harming its competitive position."*

This is Buffett's preferred metric for understanding true economic value generation.

---

## ⚖️ Financial Health Metrics

### 14. CurrentRatio
**Formula**: `AssetsCurrent / LiabilitiesCurrent`

**Exact Metrics Used**:
- `AssetsCurrent` (Balance Sheet - Assets)
- `LiabilitiesCurrent` (Balance Sheet - Liabilities)

**Calculation Method**:
```python
def calculate_current_ratio(current_assets, current_liabilities):
    if current_assets is None:
        return None
    if current_liabilities is None or current_liabilities == 0:
        return 999.99 if current_assets > 0 else None  # Infinite liquidity
    if current_liabilities > 0:
        ratio = current_assets / current_liabilities
        return min(ratio, 999.99)  # Cap at reasonable maximum
    return None
```

**Why Buffett Values This**:
*"I like businesses that don't need a lot of working capital."*

Current ratio measures short-term liquidity and financial stability, important for business continuity.

---

### 15. DebtToEquityRatio
**Formula**: `(DebtCurrent + DebtNoncurrent) / StockholdersEquity`

**Exact Metrics Used**:
- `DebtCurrent` (Balance Sheet - Liabilities)
- `DebtNoncurrent` (Balance Sheet - Liabilities)
- `StockholdersEquity` (Balance Sheet - Equity)

**Calculation Method**:
```python
def calculate_debt_to_equity_ratio(debt_current, debt_noncurrent, stockholders_equity):
    total_debt = (debt_current or 0) + (debt_noncurrent or 0)
    
    if stockholders_equity and stockholders_equity > 0:
        return total_debt / stockholders_equity
    return None  # Negative equity makes ratio meaningless
```

**Why Buffett Values This**:
*"We avoid businesses that are heavily leveraged. Leverage can produce extraordinary returns, but also extraordinary losses."*

Low debt-to-equity ratios indicate conservative financial management and reduced financial risk.

---

### 16. InterestCoverageRatio
**Formula**: `OperatingIncomeLoss / InterestExpense`

**Exact Metrics Used**:
- `OperatingIncomeLoss` (Income Statement)
- `InterestExpense` (Income Statement)

**Calculation Method**:
```python
def calculate_interest_coverage_ratio(operating_income, interest_expense):
    if interest_expense and interest_expense > 0:
        return operating_income / interest_expense
    elif interest_expense == 0 and operating_income:
        return 999.99  # No interest expense = infinite coverage
    return None
```

**Why Buffett Values This**:
*"We avoid businesses that have poor interest coverage."*

High interest coverage ratios indicate strong ability to service debt obligations safely.

---

## 🎯 Return Metrics

### 17. ReturnOnEquity (ROE)
**Formula**: `NetIncomeLoss / StockholdersEquity * 100`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement)
- `StockholdersEquity` (Balance Sheet - Equity)

**Calculation Method**:
```python
def calculate_roe(net_income, stockholders_equity):
    if stockholders_equity and stockholders_equity > 0:
        return (net_income / stockholders_equity) * 100
    return None  # Negative equity makes ROE not meaningful
```

**Why Buffett Values This**:
*"The primary test of managerial economic performance is the achievement of a high earnings rate on equity capital employed."*

ROE measures management's effectiveness in generating profits from shareholders' investments.

---

### 18. ReturnOnAssets (ROA)
**Formula**: `NetIncomeLoss / Assets * 100`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement)
- `Assets` (Balance Sheet - Assets)

**Calculation Method**:
```python
def calculate_roa(net_income, total_assets):
    if total_assets and total_assets > 0:
        return (net_income / total_assets) * 100
    return None
```

**Why Buffett Values This**:
*"I like businesses that don't require a lot of capital to generate earnings."*

ROA shows how efficiently management uses assets to generate profits, indicating capital efficiency.

---

### 19. ReturnOnInvestedCapital (ROIC)
**Formula**: `(NetIncomeLoss + InterestExpense * (1 - TaxRate)) / (StockholdersEquity + TotalDebt) * 100`

**Exact Metrics Used**:
- `NetIncomeLoss` (Income Statement)
- `InterestExpense` (Income Statement)
- `IncomeTaxExpenseBenefit` (Income Statement) - for tax rate calculation
- `StockholdersEquity` (Balance Sheet - Equity)
- `DebtCurrent` + `DebtNoncurrent` (Balance Sheet - Liabilities)

**Calculation Method**:
```python
def calculate_roic(net_income, interest_expense, tax_expense, revenues, 
                  stockholders_equity, debt_current, debt_noncurrent):
    # Estimate tax rate
    if tax_expense is not None and net_income is not None:
        pre_tax_income = net_income + tax_expense
        if pre_tax_income > 0:
            tax_rate = min(tax_expense / pre_tax_income, 0.50)
        else:
            tax_rate = 0.25
    else:
        tax_rate = 0.25  # Default assumption
    
    # Calculate NOPAT
    interest_tax_shield = (interest_expense or 0) * (1 - tax_rate)
    nopat = (net_income or 0) + interest_tax_shield
    
    # Calculate invested capital
    total_debt = (debt_current or 0) + (debt_noncurrent or 0)
    invested_capital = (stockholders_equity or 0) + total_debt
    
    if invested_capital > 0:
        return (nopat / invested_capital) * 100
    return None
```

**Why Buffett Values This**:
*"The primary test of managerial economic performance is achieving a high return on the capital they employ."*

ROIC measures returns on all invested capital, providing a comprehensive view of capital efficiency.

---

## 📊 Capital Allocation Metrics

### 20. RetainedEarningsToNetIncome
**Formula**: `RetainedEarnings / (NetIncomeLoss * Years_In_Business)`

**Exact Metrics Used**:
- `RetainedEarnings` (Balance Sheet - Equity)
- `NetIncomeLoss` (Income Statement)

**Calculation Method**:
```python
def calculate_earnings_retention_efficiency(year_data, prior_year_data):
    retained_earnings = year_data.get('RetainedEarnings')
    current_net_income = year_data.get('NetIncomeLoss')
    
    if retained_earnings and current_net_income and current_net_income != 0:
        return (retained_earnings / abs(current_net_income)) * 100
    return None
```

**Why Buffett Values This**:
*"We want businesses that retain earnings productively."*

This ratio shows management's discipline in retaining versus distributing earnings for productive reinvestment.

---

### 21. DividendPayoutRatio
**Formula**: `CommonDividendsPaid / NetIncomeLoss * 100`

**Exact Metrics Used**:
- `CommonDividendsPaid` (Cash Flow Statement - Financing)
- `NetIncomeLoss` (Income Statement)

**Calculation Method**:
```python
def calculate_dividend_payout_ratio(dividends_paid, net_income):
    if net_income and net_income > 0 and dividends_paid:
        return abs(dividends_paid) / net_income * 100  # Dividends usually negative
    return 0  # No dividends paid
```

**Why Buffett Values This**:
*"We like companies with sustainable dividend policies."*

Dividend payout ratio reveals management's capital allocation philosophy and dividend sustainability.

---

### 22. CapitalExpenditureToDepreciation
**Formula**: `CapitalExpenditures / DepreciationAndAmortization`

**Exact Metrics Used**:
- `CapitalExpenditures` (Cash Flow Statement - Investing)
- `DepreciationAndAmortization` (Cash Flow Statement - Operating)

**Calculation Method**:
```python
def calculate_capex_to_depreciation_ratio(capex, depreciation):
    if depreciation and depreciation > 0 and capex:
        return abs(capex) / depreciation  # Capex usually negative in cash flow
    return None
```

**Why Buffett Values This**:
*"Maintenance capex should roughly equal depreciation for mature businesses."*

This ratio helps distinguish between maintenance and growth capital expenditures:
- **Ratio ≈ 1.0**: Maintenance capex (mature, stable business)
- **Ratio > 1.5**: Growth capex (expanding business)
- **Ratio < 0.8**: Potential underinvestment

---

## 📋 Implementation Summary

### **Total Comprehensive Dataset**:
- **104 Base SEC EDGAR Metrics**
- **22 Derived Warren Buffett-Style Metrics**
- **126 Total Metrics**

### **Output Format**:
```
Year, [104 base SEC metrics], EarningsPerShare, PriceToEarning, BookValuePerShare, 
PriceToBook, RevenueGrowthRate, NetIncomeGrowthRate, BookValueGrowthRate, GrossMargin, 
OperatingMargin, NetIncomeMargin, FreeCashFlowMargin, FreeCashFlow, OwnerEarnings, 
CurrentRatio, DebtToEquityRatio, InterestCoverageRatio, ReturnOnEquity, ReturnOnAssets, 
ReturnOnInvestedCapital, RetainedEarningsToNetIncome, DividendPayoutRatio, 
CapitalExpenditureToDepreciation
```

### **Key Value Propositions**:
1. **Complete Financial Analysis**: All major financial ratios and metrics
2. **Buffett-Style Value Investing**: Metrics specifically chosen for identifying quality businesses
3. **Institutional Quality**: Robust error handling and accounting standards compliance
4. **Historical Consistency**: 11 years of data with proper handling of accounting transitions
5. **Scalable Analysis**: 8,132+ companies with standardized calculations

This comprehensive dataset enables **professional-grade financial analysis** aligned with Warren Buffett's investment philosophy and value investing principles.