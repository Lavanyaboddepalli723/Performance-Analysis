import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load the performance metrics CSV files
imdb_metrics_df = pd.read_csv('imdb_performance_metrics.csv')
twitter_metrics_df = pd.read_csv('twitter_performance_metrics.csv')

# Define the dictionary mapping model names to their corresponding confusion matrix file names
confusion_matrix_files = {
    'IMDB': {
        'SVM': 'imdb_SVM_cm.pkl',
        'Logistic Regression': 'imdb_Logistic Regression_cm.pkl',
        'Naive Bayes': 'imdb_Naive Bayes_cm.pkl',
        'Random Forest': 'imdb_Random Forest_cm.pkl',
        'Ensemble': 'imdb_Ensemble_cm.pkl'
    },
    'Twitter': {
        'SVM': 'twitter_SVM_cm.pkl',
        'Logistic Regression': 'twitter_Logistic Regression_cm.pkl',
        'Naive Bayes': 'twitter_Naive Bayes_cm.pkl',
        'Random Forest': 'twitter_Random Forest_cm.pkl',
        'Ensemble': 'twitter_Ensemble_cm.pkl'
    }
}

# Function to load confusion matrix from pickle file
def load_confusion_matrix(dataset, model_name):
    filename = confusion_matrix_files[dataset][model_name]
    return joblib.load(filename)

# Function to get performance metrics for selected models and dataset
def get_metrics(dataset_metrics_df, model_names):
    selected_metrics = dataset_metrics_df[dataset_metrics_df['Model'].isin(model_names)]
    return selected_metrics

# Function to load ROC curve data from joblib file
def load_roc_curve_data(dataset):
    filename = f'{dataset.lower()}_roc_curve_data.joblib'
    return joblib.load(filename)

#Function to generate comparative graphs for all algorithms based on different performance metrics and ROC curves
def generate_comparative_graphs(dataset_metrics_df, roc_curve_data):
    st.subheader('Comparative Graphs')
    
    #Plotting performance metrics
    performance_metrics = ['Training Time', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    for metric in performance_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=dataset_metrics_df, x='Model', y=metric, ax=ax)
        ax.set_title(f'{metric} Comparison of Algorithms')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    
    #Plotting ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    for model_name in dataset_metrics_df['Model']:
        fpr = roc_curve_data[model_name]['fpr']
        tpr = roc_curve_data[model_name]['tpr']
        ax_roc.plot(fpr, tpr, label=f'{model_name} ROC curve')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves for All Algorithms')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)


# Streamlit app
def main():
    
    # Add college logo with reduced size
    logo_col, name_col = st.columns([1, 3])
    with logo_col:
        st.image('Jntuk-logo.png', width=150)

    # Display the college name beside the logo
    with name_col:
        st.title('Univeristy College of Engineering Kakinada') 
    
    st.markdown('<hr style="margin: 5px;">', unsafe_allow_html=True)          
    st.markdown('<h3 style="color:#002D62;text-align:center;margin-bottom:0;">Peformance analysis of various classification algorithms using IMDB dataset and Twitter dataset</h3>', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 5px;">', unsafe_allow_html=True)

    # Sidebar for dataset selection
    st.sidebar.title('Choose Dataset')
    dataset = st.sidebar.selectbox('Select Dataset', ['IMDB', 'Twitter'])

    if dataset == 'IMDB':
        roc_curve_data = load_roc_curve_data('IMDB')
        selected_metrics_df = imdb_metrics_df
    else:
        roc_curve_data = load_roc_curve_data('Twitter')
        selected_metrics_df = twitter_metrics_df

    # Sidebar for model selection
    st.sidebar.title('Choose Models')
    model_1 = st.sidebar.selectbox('Select Model 1', list(roc_curve_data.keys()))
    model_2 = st.sidebar.selectbox('Select Model 2', list(roc_curve_data.keys()))

    # Display performance metrics for selected models and dataset
    st.subheader('Performance Metrics')
    selected_metrics = get_metrics(selected_metrics_df, [model_1, model_2])
    st.write(selected_metrics)

    # Determine best model
    st.subheader('Best Model')
    best_model = selected_metrics.loc[selected_metrics['Accuracy'].idxmax(), 'Model']
    st.write(f"The best model between {model_1} and {model_2} is: {best_model}")
    
    #Display confusion matrices for selected models and dataset
    st.subheader('Confusion Matrices')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figure size as needed
    for i, model_name in enumerate([model_1, model_2]):
        ax = axes[i]
        ax.set_title(f'Confusion Matrix for {model_name}')
        confusion_matrix_data = load_confusion_matrix(dataset, model_name)
        sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
    st.pyplot(fig)

    # Display ROC curves for the chosen models side by side
    st.subheader('ROC Curves')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create subplots
    for i, model_name in enumerate([model_1, model_2]):
        ax = axes[i]
        roc_curve_data = load_roc_curve_data(dataset)
        fpr = roc_curve_data[model_name]['fpr']
        tpr = roc_curve_data[model_name]['tpr']
        ax.plot(fpr, tpr, label=f'{model_name} ROC curve')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for {model_name}')
        ax.legend(loc='lower right')
    st.pyplot(fig)  # Display the subplots
    
    # Generate comparative graphs on a separate page if button is clicked
    if st.sidebar.button('Generate Comparative Graphs'):
        with st.sidebar.expander('Comparative Graphs'):
            generate_comparative_graphs(selected_metrics_df, roc_curve_data)
            
if __name__ == '__main__':
    main()


# Footer section
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 1500px;
    background-color: #f1f1f1;
    color: black;
    text-align: center;
    padding: 0.5px 0;
    font-size: 8px;
}
</style>

<div class="footer">
    Developed by: Lavanya Boddepalli, Pathan Althaf Vahid Khan, Thota Naga Vamsi, Syed Zubair Ahmad Bukhari
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
