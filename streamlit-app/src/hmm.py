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
    
    # Ensure Returns column exists
    if 'Returns' not in data.columns:
        print("Calculating Returns...")
        data['Returns'] = data['Close'].diff()
    
    print("First few rows of Returns:\n", data['Returns'].head())
    
    # Prepare features
    features = data[['Returns']].dropna().values  # 2D array
    aligned_data = data.dropna(subset=['Returns'])  # Align with features
    
    print("Features shape:", features.shape)
    print("Aligned data shape:", aligned_data.shape)
    
    return aligned_data, features

def train_hmm(features, n_states):
    print("Training HMM with features shape:", features.shape)
    model = GaussianHMM(n_components=n_states, n_iter=1000, random_state=42)
    model.fit(features)
    hidden_states = model.predict(features)
    
    return model, hidden_states

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
            
            # Create a new Series with close prices only at matching state positions, NaN elsewhere
            state_data = pd.Series(index=data.index, dtype=float)
            state_data.loc[data.index[state_indices]] = data['Close'][state_indices]
            
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
