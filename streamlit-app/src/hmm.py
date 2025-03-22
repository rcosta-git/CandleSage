from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import numpy as np

def prepare_hmm_features(data, interval="1d"):
    print("\nPreparing HMM features...")
    print("Input data shape:", data.shape)
    print("Input columns:", data.columns)
    
    # Flatten the multi-index
    data.columns = data.columns.get_level_values(0)
    
    # Make a copy to avoid modifying the original data
    hmm_data = data.copy()
    
    # Check for suspicious price movements and outliers
    if 'Close' in hmm_data.columns:
        # Calculate percentage changes
        pct_changes = hmm_data['Close'].pct_change()
        
        # Detect extreme movements (e.g., > 20% in a single period)
        extreme_threshold = 0.20  # 20% price change
        extreme_moves = abs(pct_changes) > extreme_threshold
        
        extreme_count = extreme_moves.sum()
        if extreme_count > 0:
            print(f"WARNING: Detected {extreme_count} extreme price"
                  f" movements (>20%)")
            print("Extreme moves at:", hmm_data.index[extreme_moves].tolist())
            
            # Option 1: Remove these data points
            # hmm_data = hmm_data[~extreme_moves]
            
            # Option 2: Cap the extreme moves to a reasonable value
            # (winsorizing)
            max_reasonable_move = extreme_threshold
            pct_changes[pct_changes > max_reasonable_move] = max_reasonable_move
            pct_changes[pct_changes < -max_reasonable_move] = \
                -max_reasonable_move
            
            # Reconstruct prices with capped movements
            # This will help prevent the HMM from being distorted by outliers
            base_price = hmm_data['Close'].iloc[0]
            new_prices = [base_price]
            for pct in pct_changes.iloc[1:]:
                new_prices.append(new_prices[-1] * (1 + pct))
            
            # Replace the original prices with adjusted ones
            hmm_data['Close'] = pd.Series(new_prices, index=hmm_data.index)
    
    # Calculate returns based on cleaned data
    print("Calculating Returns...")
    hmm_data['Returns'] = hmm_data['Close'].diff()
    
    print("First few rows of Returns:\n", hmm_data['Returns'].head())
    
    # Prepare features
    features = hmm_data[['Returns']].dropna().values  # 2D array
    aligned_data = hmm_data.dropna(subset=['Returns'])  # Align with features
    
    print("Features shape:", features.shape)
    print("Aligned data shape:", aligned_data.shape)
    
    return aligned_data, features

def train_hmm(features, n_states):
    print("Training HMM with features shape:", features.shape)
    
    # Ensure we have enough data for the model
    if features.shape[0] < 10:  # Absolute minimum data points needed
        print("Not enough data points for HMM training")
        n_states = 2  # Force to simplest model
    
    try:
        # First attempt: standard training
        model = GaussianHMM(n_components=n_states, n_iter=1000, 
                          covariance_type="full", random_state=42)
        model.fit(features)
        
        # Direct fix for transition matrix issues
        # Ensure no row in transmat_ sums to 0
        fix_transition_matrix(model)
        
        hidden_states = model.predict(features)
        print(f"Successfully trained HMM with {n_states} states")
        
    except Exception as e:
        print(f"Error in HMM training: {e}")
        print("Falling back to simple model")
        
        # Create a simple deterministic model
        model = create_simple_hmm(features, n_states)
        
        # Use a basic state assignment based on return magnitude
        hidden_states = assign_states_by_returns(features, n_states)
    
    return model, hidden_states

def fix_transition_matrix(model):
    """Fix the transition matrix to ensure all rows sum to 1"""
    transmat = model.transmat_
    
    # Find rows that sum to 0 or are very close to 0
    zero_sum_rows = np.where(np.sum(transmat, axis=1) < 0.001)[0]
    
    if len(zero_sum_rows) > 0:
        print(f"Fixing {len(zero_sum_rows)} rows in transition matrix")
        
        for row in zero_sum_rows:
            # Set uniform transitions for this state
            transmat[row, :] = 1.0 / transmat.shape[1]
    
    # Normalize other rows to ensure they sum to 1
    row_sums = np.sum(transmat, axis=1, keepdims=True)
    model.transmat_ = transmat / np.maximum(row_sums, 1e-10)

def create_simple_hmm(features, n_states):
    """Create a simple HMM model with predefined parameters"""
    model = GaussianHMM(n_components=n_states)
    
    # Set uniform starting probabilities
    model.startprob_ = np.ones(n_states) / n_states
    
    # Set uniform transition matrix
    model.transmat_ = np.ones((n_states, n_states)) / n_states
    
    # Calculate simple means and covariances based on data statistics
    means = []
    covars = []
    
    # Get data statistics
    mean_return = np.mean(features)
    std_return = max(np.std(features), 0.0001)  # Ensure non-zero
    
    # Create means for each state, spaced proportionally
    for i in range(n_states):
        # Distribute states across the range of returns
        state_factor = -1.0 + (2.0 * i / max(1, n_states - 1))
        means.append([mean_return + state_factor * std_return])
        covars.append([[std_return * (0.5 + 0.5 * i)]])
    
    model.means_ = np.array(means)
    model.covars_ = np.array(covars)
    
    return model

def assign_states_by_returns(features, n_states):
    """Assign states based on returns magnitude"""
    hidden_states = np.zeros(len(features), dtype=int)
    
    if n_states == 2:
        # Simple bull/bear state assignment
        mean_return = np.mean(features)
        for i in range(len(features)):
            hidden_states[i] = 1 if features[i][0] > mean_return else 0
    else:
        # Multiple states based on percentiles
        boundaries = np.percentile(features, 
                                  np.linspace(0, 100, n_states + 1)[1:-1])
        
        for i in range(len(features)):
            state = 0
            for j, boundary in enumerate(boundaries):
                if features[i][0] > boundary:
                    state = j + 1
            hidden_states[i] = state
    
    return hidden_states

def create_hmm_plot(data, hidden_states, n_states):   
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        addplots = []
        
        # Add close price line
        addplots.append(mpf.make_addplot(
            data['Close'],
            color='blue',
            width=1,
            label='Close Price'
        ))
        
        # Prepare data for HMM state scatter plots
        state_colors = ['red', 'green', 'purple', 'orange', 'cyan']
        state_colors = state_colors[:n_states]
        
        # Create an addplot for each state
        for state in range(n_states):
            
            state_indices = (hidden_states == state)
            
            # Check if this state has any data points
            if not np.any(state_indices):
                print(f"State {state} has no data points, skipping plot")
                continue
            
            # Create a new Series with close prices only at matching state
            # positions, NaN elsewhere
            state_data = pd.Series(index=data.index, dtype=float)
            state_data.loc[data.index[state_indices]] = \
                data['Close'][state_indices]
            
            addplots.append(mpf.make_addplot(
                state_data,
                type='scatter',
                markersize=40,
                marker='o',
                color=state_colors[state],
                label=f'State {state}'
            ))
        
        # Set up style for the chart
        mc = mpf.make_marketcolors(
            up='green', down='red',
            wick={'up':'green', 'down':'red'},
            edge={'up':'green', 'down':'red'},
            volume={'up':'green', 'down':'red'}
        )

        s = mpf.make_mpf_style(
            marketcolors=mc, 
            gridstyle='--', 
            y_on_right=True,
            facecolor='white'
        )

        fig, axes = mpf.plot(
                data, 
                type='line',
                style=s,
                addplot=addplots,
                title='Close Price with HMM States',
                ylabel='Price (USD)',
                figsize=(15, 8),
                returnfig=True
        )
        return fig


def analyze_hidden_states(data, hidden_states, n_states, model, precision):
    state_info = []
    for state in range(n_states):
        # Round the average return and covariance to the specified precision
        avg_return = round(model.means_[state][0], precision)
        covariance = round(model.covars_[state][0][0], precision)
        state_info.append((state, avg_return, covariance))
    
    return state_info

def generate_hmm_analysis(df, interval="1d", n_states=3, precision=2):
    """Integrated function to generate all HMM analysis components"""
    print("\nStarting HMM analysis")
    print("Input dataframe shape:", df.shape)
    print("Input dataframe columns:", df.columns)

    # Prepare and train model
    data, features = prepare_hmm_features(df, interval) 

    if features.shape[0] < 5:
        print("ERROR: Not enough data points after preprocessing!")
        return None, pd.DataFrame(), pd.DataFrame()
    
    model, hidden_states = train_hmm(features, n_states)
    
    # Generate plot
    fig = create_hmm_plot(data, hidden_states, n_states)
    
    # Analyze states
    state_info = analyze_hidden_states(
        data, hidden_states, n_states, model, precision)
    
    # Create state analysis DataFrame
    state_df = pd.DataFrame(
        state_info, columns=['State', 'Average Return', 'Covariance'])
    state_df = state_df.set_index('State')
    
    # Get transition matrix
    transition_matrix = model.transmat_.round(precision)
    
    # Format the transition matrix to display with two decimal places
    transition_matrix_df = pd.DataFrame(transition_matrix)
    transition_matrix_df.columns = [
        f'State {i}' for i in range(transition_matrix_df.shape[1])]
    transition_matrix_df.index = [
        f'State {i}' for i in range(transition_matrix_df.shape[0])]

    return fig, state_df, transition_matrix_df
