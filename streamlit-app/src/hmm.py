from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import pandas as pd

def prepare_hmm_features(data):
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
    fig = plt.figure(figsize=(15, 8))
    plt.plot(data.index, data['Close'], color='gray', alpha=0.5, label="Price")
    state_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    state_colors = state_colors[:n_states]
    for state in range(n_states):
        state_indices = (hidden_states == state)
        plt.scatter(data.index[state_indices], data['Close'][state_indices], 
                    color=state_colors[state], label=f"State {state}", s=10)
    plt.title("Price by Hidden States")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    return fig

def analyze_hidden_states(data, hidden_states, n_states, model):
    state_info = []
    for state in range(n_states):
        avg_return = model.means_[state][0]  # Get mean from the model
        covariance = model.covars_[state][0]  # Get covariance from the model
        state_info.append((state, avg_return, covariance))  # Include mean and covariance in the result
    return state_info

def generate_hmm_analysis(df, n_states=3):
    """Integrated function to generate all HMM analysis components"""
    print("\nStarting HMM analysis")
    print("Input dataframe shape:", df.shape)
    print("Input dataframe columns:", df.columns)
    
    # Prepare and train model
    data, features = prepare_hmm_features(df)
    model, hidden_states = train_hmm(features, n_states)
    
    # Generate plot
    fig = create_hmm_plot(data, hidden_states, n_states)
    
    # Analyze states
    state_info = analyze_hidden_states(data, hidden_states, n_states, model)
    
    # Create state analysis DataFrame
    state_df = pd.DataFrame(
        state_info, columns=['State', 'Average Return', 'Covariance'])
    state_df = state_df.set_index('State')
    
    return fig, state_df
