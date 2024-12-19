import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

def load_and_validate_data(file) -> Tuple[bool, pd.DataFrame]:
    """Load and validate the uploaded dataset."""
    try:
        data = pd.read_csv(file)
        required_columns = {'user_id', 'item_id', 'rating'}
        if not all(col in data.columns for col in required_columns):
            return False, pd.DataFrame()
        return True, data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return False, pd.DataFrame()

def create_user_item_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Create and return the user-item matrix."""
    return data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating'
    ).fillna(0)

def get_recommendations(
    user_item_matrix: pd.DataFrame,
    user_id: int,
    n_recommendations: int = 10
) -> pd.Series:
    """Generate recommendations for a specific user."""
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(user_item_matrix)
    user_index = list(user_item_matrix.index).index(user_id)
    
    # Find similar users
    similar_users = np.argsort(-similarity_matrix[user_index])[1:6]
    
    # Generate recommendations
    recommendations = user_item_matrix.iloc[similar_users].mean(axis=0)
    current_user_items = user_item_matrix.loc[user_id]
    
    # Filter out already rated items
    recommendations = recommendations[current_user_items == 0]
    return recommendations.sort_values(ascending=False).head(n_recommendations)

def plot_rating_distribution(data: pd.DataFrame, user_id: int) -> go.Figure:
    """Create an interactive plot of user rating distribution."""
    user_ratings = data[data['user_id'] == user_id]['rating']
    fig = px.histogram(
        user_ratings,
        nbins=10,
        title=f"Rating Distribution for User {user_id}",
        labels={'value': 'Rating', 'count': 'Count'},
        color_discrete_sequence=['#3366cc']
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Rating",
        yaxis_title="Count"
    )
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Media Recommendation System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10
        )
    
    # Main content
    st.title("Media Content Recommendation System")
    st.markdown("""
    Get personalized content recommendations by uploading your dataset.
    
    Required columns in CSV:
    - user_id: User identifier
    - item_id: Content identifier
    - rating: User rating
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV format)",
        type="csv",
        help="Upload a CSV file with required columns"
    )
    
    if uploaded_file is not None:
        # Load and validate data
        is_valid, data = load_and_validate_data(uploaded_file)
        
        if is_valid:
            # Display dataset info
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Dataset Preview")
                st.write(data.head())
            with col2:
                st.write("### Dataset Statistics")
                st.write(f"Number of users: {data['user_id'].nunique():,}")
                st.write(f"Number of items: {data['item_id'].nunique():,}")
                st.write(f"Number of ratings: {len(data):,}")
                st.write(f"Average rating: {data['rating'].mean():.2f}")
            
            # User selection
            user_id = st.selectbox(
                "Select User ID:",
                data['user_id'].unique(),
                help="Select a user to get recommendations"
            )
            
            if st.button("Generate Recommendations"):
                with st.spinner("Generating recommendations..."):
                    # Create user-item matrix
                    user_item_matrix = create_user_item_matrix(data)
                    
                    # Get recommendations
                    recommendations = get_recommendations(
                        user_item_matrix,
                        user_id,
                        n_recommendations
                    )
                    
                    # Display recommendations
                    st.write("### Top Recommended Items")
                    recommendations_df = pd.DataFrame({
                        'Item ID': recommendations.index,
                        'Recommendation Score': recommendations.values.round(2)
                    })
                    st.write(recommendations_df)
                    
                    # User Analysis
                    st.write("### User Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        user_stats = data[data['user_id'] == user_id]['rating'].describe()
                        st.write("Rating Statistics")
                        st.write(user_stats)
                    
                    with col2:
                        st.write("Rating Distribution")
                        fig = plot_rating_distribution(data, user_id)
                        st.plotly_chart(fig)
        else:
            st.error("""
            Invalid dataset format. Please ensure your CSV file contains:
            - user_id: User identifier
            - item_id: Content identifier
            - rating: User rating
            """)

if __name__ == "__main__":
    main()
