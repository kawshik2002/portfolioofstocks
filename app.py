import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import yfinance
# print(yfinance.__file__)



def calculate_benchmark_equity_curve(stock_prices, initial_investment):
    num_stocks = len(stock_prices.columns)
    portfolio_value = stock_prices.sum(axis=1)
    investment_per_stock = initial_investment / num_stocks
    benchmark_equity_curve = investment_per_stock * (portfolio_value / portfolio_value.iloc[0])
    return benchmark_equity_curve


def calculate_sample_strategy_equity_curve(stock_prices, initial_investment, measurement_period, num_stocks):
    returns = stock_prices.pct_change()
    returns = returns.iloc[-measurement_period:].mean()
    selected_stocks = returns.nlargest(num_stocks).index.tolist()
    selected_prices = stock_prices[selected_stocks]
    portfolio_value = selected_prices.sum(axis=1)
    sample_strategy_equity_curve = initial_investment * (portfolio_value / portfolio_value.iloc[0])
    return sample_strategy_equity_curve, selected_stocks


def calculate_nifty_index_equity_curve(nifty_prices, initial_investment):
    nifty_equity_curve = initial_investment * (nifty_prices / nifty_prices.iloc[0])
    return nifty_equity_curve


def calculate_cagr(returns):
    total_return = returns[-1]
    num_years = len(returns) / 252
    cagr = ((1 + total_return) ** (1 / num_years)) - 1
    return cagr * 100


def calculate_volatility(returns):
    volatility = returns.std() * np.sqrt(252)
    return volatility * 100


def calculate_sharpe_ratio(returns):
    cagr = calculate_cagr(returns)
    volatility = calculate_volatility(returns)
    sharpe_ratio = cagr / volatility
    return sharpe_ratio


def summarize_performance(equity_curves):
    performance = {}
    for name, equity_curve in equity_curves.items():
        returns = equity_curve.pct_change().dropna()
        cagr = calculate_cagr(returns)
        volatility = calculate_volatility(returns)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        performance[name] = {'CAGR (%)': cagr, 'Volatility (%)': volatility, 'Sharpe Ratio': sharpe_ratio}
    return performance


def app():
    st.title("Portfolio Out of Nifty50 Stocks")

    # Input parameters
    start_date = st.text_input("Enter the start date (YYYY-MM-DD):")
    end_date = st.text_input("Enter the end date (YYYY-MM-DD):")
    num_days = st.number_input("Enter the number of days for stock selection:", value=30)
    num_top_stocks = st.number_input("Enter the number of top stocks to select:", value=5)
    initial_equity = st.number_input("Enter the initial equity:", value=100000)

    if st.button("Submit"):
        try:
            # Fetch historical stock prices
            stock_symbols = ['RELIANCE.NS', 'HCLTECH.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS',
                            'BAJFINANCE.NS', 'APOLLOHOSP.NS', 'WIPRO.NS', 'ADANIENT.NS']
            stock_data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']

            # Fetch Nifty Index prices
            nifty_data = yf.download('^NSEI', start=start_date, end=end_date)['Adj Close']

            if stock_data.empty or nifty_data.empty:
                st.error("Error: No data available for the specified date range.")
                return

            # Objective 1: Benchmark Strategy
            initial_investment = initial_equity
            benchmark_equity_curve = calculate_benchmark_equity_curve(stock_data, initial_investment)
            st.write("Benchmark Equity Curve:")
            st.write(benchmark_equity_curve)

            # Objective 2: Sample Strategy
            measurement_period = num_days
            num_stocks = num_top_stocks
            sample_strategy_equity_curve, selected_stocks = calculate_sample_strategy_equity_curve(stock_data,
                                                                                                initial_investment,
                                                                                                measurement_period,
                                                                                                num_stocks)
            st.write("Sample Strategy Equity Curve:")
            st.write(sample_strategy_equity_curve)

            # Objective 3: Nifty Index Equity Curve
            nifty_equity_curve = calculate_nifty_index_equity_curve(nifty_data, initial_investment)
            st.write("Nifty Index Equity Curve:")
            st.write(nifty_equity_curve)

            # Objective 4: Performance Summary
            equity_curves = {
                'Nifty Index': nifty_equity_curve,
                'Benchmark Strategy': benchmark_equity_curve,
                'Sample Strategy': sample_strategy_equity_curve
            }
            performance_summary = summarize_performance(equity_curves)

            # Print the performance summary
            st.write("Performance Summary:")
            for name, metrics in performance_summary.items():
                st.write(f'{name}:')
                for metric, value in metrics.items():
                    st.write(f'{metric}: {value:.2f}')
                st.write()

            # Plot the equity curves
            plt.plot(benchmark_equity_curve.index, benchmark_equity_curve, label='Benchmark Strategy')
            plt.plot(sample_strategy_equity_curve.index, sample_strategy_equity_curve, label='Sample Strategy')
            plt.plot(nifty_equity_curve.index, nifty_equity_curve, label='Nifty Index')
            plt.xlabel('Date')
            plt.ylabel('Equity Curve')
            plt.title('Portfolio Performance')
            plt.legend()

            # Display the plot
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # st.set_option('deprecation.showPyplotGlobalUse', False)

        except ValueError:
            st.error("Error: No data available for the specified date range. Please check your inputs.")


if __name__ == '__main__':
    app()
