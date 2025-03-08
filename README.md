# Trading System Architectural Concepts

Our trading system is designed to support a broad range of strategies:

- **Day Trading & Swing Trading:**  
  These strategies last from a few minutes up to a few days. They typically 
  focus on futures or options to capture short-term market inefficiencies. 
  The system generates quick, actionable signals and executes trades with 
  pre-set stop losses.

- **Long-Term Strategies:**  
  This component can include approaches like buying deep in-the-money options 
  (or LEAPS) as a form of "equity replacement." While these strategies might 
  be managed separately, ideally they form a component of overall portfolio 
  optimization.

- **Recurring Options Income Strategies:**  
  For trades that need an underlying stock position (e.g., the wheel strategy),
  our platform is designed to accommodate and integrate these strategies, 
  effectively bridging short-term and long-term trading approaches.

## Day and Swing Trading

### Diagram 1: Overall Short-Term Trading System Architecture

This diagram shows the process flow from generating trading strategies based on
technical signals, through executing trades with fixed stop losses, and finally
ongoing monitoring positions with dynamic exits. New strategies are placed on a 
rolling basis within the hardcoded daily limits and availability of capital.

```mermaid
flowchart LR
    A[Technical Signals]
    B[Trading Strategy Generation]
    C[Trade Execution<br>with Fixed Stop Losses]
    D[Trade Monitoring<br>Technical Signals<br>with Dynamic Exits]

    A --> B
    B --> C
    C --> D
    D -- "New strategies restarted depending on daily limits" --> A
```

### Diagram 2: Short-Term Trading Signal Generation Pipeline

This diagram breaks down the process of generating trading signals by combining
a live data feed and historical data. Simple indicators (e.g., RSI, EMA
crossover) produce basic calculated states, while more complex processing
(via HMM and LSTM models) generates advanced states. Both feed into the overall
trading signal monitoring system.

```mermaid
flowchart TD
    A[Live Data Feed]
    B[Historical Data]
    C[Calculated States<br>RSI, EMA Crossover]
    D[Complex States<br>HMM, LSTM]
    E[Trading Signal Monitoring]

    A --> C
    B --> C
    A --> D
    C --> E
    D --> E
```

## Long-Term Portfolio Management and Optimization

This system combines classical portfolio optimization with advanced AI-based.
signal generation. The idea is to leverage historical data to calculate expected
returns and risk (covariance) while simultaneously incorporating real-time
signals to dynamically adjust the portfolio. This dual approach aims to improve
long-term portfolio construction and management.

### System Components

1. **Historical Data Pipeline:**
   - **Historical Price Data:** Ingest historical prices.
   - **Data Preprocessing & Cleaning:** Prepare the data for analysis.
   - **Expected Returns & Covariance:** Calculate statistical metrics for
     portfolio optimization.
   - **Mean-Variance Optimization:** Generate baseline portfolio weights.

2. **Real-Time AI Signals:**
   - **Live Data Feed:** Capture current market data.
   - **Real-Time Processing:** Clean and format live data.
   - **HMM Signal Generation:** Generate predictive signals using Hidden Markov
     Models.
   - **LSTM Signal Generation:** Generate predictive signals using LSTM networks.
   - **Aggregation:** Combine the AI signals into an overall adjustment factor.

3. **Portfolio Integration & Execution:**
   - **Integration:** Merge historical optimization outputs with AI signal
     adjustments.
   - **Adjusted Portfolio Weights:** Produce final portfolio allocations.
   - **Trade Execution:** Place trades based on the adjusted portfolio.
   - **Risk Management & Monitoring:** Apply stop losses, rebalance, and update
     the model based on performance feedback.

### System Flow Diagram

```mermaid
flowchart TD
    %% Historical Data Branch
    A[Historical Price Data]
    B[Data Preprocessing & Cleaning]
    C[Calculate Expected Returns]
    D[Calculate Covariance Matrix]
    E[Mean-Variance Optimization]
    F[Baseline Portfolio Weights]

    %% Real-Time AI Signals Branch
    G[Live Data Feed]
    H[Real-Time Data Processing]
    I[HMM Signal Generation]
    J[LSTM Signal Generation]
    K[Aggregate AI Signals]

    %% Integration
    L[Integrate Historical & AI Signals]
    M[Adjusted Portfolio Weights]

    %% Trade Execution & Monitoring
    N[Trade Execution & Order Placement]
    O[Risk Management & Constraint Checks]
    P[Portfolio Monitoring & Rebalancing]
    Q[Performance Feedback & Model Update]

    %% Flow Connections
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F

    G --> H
    H --> I
    H --> J
    I --> K
    J --> K

    F --> L
    K --> L
    L --> M

    M --> N
    N --> O
    O --> P
    P --> Q
```
