"""
Employee Engagement & Sentiment Analytics Pipeline
Matching resume description: NLP (sentiment + keywords), Regression, Clustering, LLM
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

# Ensure NLTK data is available
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

print("="*70)
print("EMPLOYEE ENGAGEMENT & SENTIMENT ANALYTICS PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
def load_kaggle_data(filepath='WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """Load IBM HR Analytics dataset from Kaggle"""
    print("\n[STEP 1/7] Loading Kaggle Dataset...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} employee records from IBM HR Analytics")
    return df

# ============================================================================
# STEP 2: DATA TRANSFORMATION
# ============================================================================
def transform_data(df):
    """Transform Kaggle data to analytics format with synthetic comments"""
    print("\n[STEP 2/7] Transforming Data Structure...")
    
    transformed = pd.DataFrame()
    transformed['id'] = df.index + 1
    transformed['department'] = df['Department']
    transformed['tenure'] = df['YearsAtCompany']
    transformed['job_role'] = df['JobRole']
    transformed['monthly_income'] = df['MonthlyIncome']
    transformed['age'] = df['Age']
    transformed['overtime'] = df['OverTime']
    transformed['distance_from_home'] = df['DistanceFromHome']
    
    # Create engagement score from satisfaction metrics (weighted combination)
    transformed['engagementScore'] = (
        (df['JobSatisfaction'] * 25) * 0.4 +
        (df['EnvironmentSatisfaction'] * 25) * 0.3 +
        (df['WorkLifeBalance'] * 25) * 0.3
    ).round(0).astype(int)
    
    # Store raw satisfaction scores for regression
    transformed['job_satisfaction'] = df['JobSatisfaction']
    transformed['environment_satisfaction'] = df['EnvironmentSatisfaction']
    transformed['work_life_balance'] = df['WorkLifeBalance']
    
    # Actual attrition from dataset
    transformed['attrition'] = (df['Attrition'] == 'Yes').astype(int)
    
    # Generate realistic open-text comments based on actual data
    def generate_comment(row):
        templates = {
            'high_satisfaction': [
                f"I really appreciate the {row['department']} team culture. Work-life balance is excellent and management is supportive.",
                f"Great career growth opportunities in {row['job_role']}. The compensation package is competitive and benefits are strong.",
                f"Very satisfied with my role after {row['tenure']} years. The work environment is collaborative and resources are adequate.",
                f"Excellent leadership in {row['department']}. Flexibility with schedule and clear communication from management.",
            ],
            'medium_satisfaction': [
                f"The work in {row['job_role']} is okay. Some challenges with work-life balance especially with overtime requirements.",
                f"Decent experience in {row['department']} but career advancement opportunities could be better after {row['tenure']} years.",
                f"Average satisfaction overall. Compensation is fair but workload can be overwhelming at times.",
                f"Management is acceptable but communication could improve. Tools and resources are functional but dated.",
            ],
            'low_satisfaction': [
                f"Struggling with work-life balance in {row['department']}. Frequent overtime without adequate compensation adjustment.",
                f"Limited career growth in {row['job_role']} despite {row['tenure']} years tenure. Feeling undervalued and underpaid.",
                f"Poor management communication and support. High stress environment with unrealistic expectations and deadlines.",
                f"Considering leaving due to toxic work culture. Compensation below market rate and no recognition for contributions.",
            ]
        }
        
        if row['engagementScore'] >= 70:
            category = 'high_satisfaction'
        elif row['engagementScore'] >= 40:
            category = 'medium_satisfaction'
        else:
            category = 'low_satisfaction'
            
        return np.random.choice(templates[category])
    
    transformed['comment'] = transformed.apply(generate_comment, axis=1)
    
    print(f"✓ Transformed {len(transformed)} records with synthetic open-text comments")
    return transformed

# ============================================================================
# STEP 3: NLP ANALYSIS - SENTIMENT SCORING
# ============================================================================
def analyze_sentiment(df):
    """Perform sentiment analysis using VADER and TextBlob"""
    print("\n[STEP 3/7] NLP Analysis - Sentiment Scoring...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    # VADER sentiment analysis
    def get_vader_sentiment(text):
        scores = analyzer.polarity_scores(text)
        return scores['compound'], scores['pos'], scores['neu'], scores['neg']
    
    df[['sentiment_score', 'positive_score', 'neutral_score', 'negative_score']] = \
        df['comment'].apply(lambda x: pd.Series(get_vader_sentiment(x)))
    
    # Classify sentiment
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # TextBlob for additional polarity/subjectivity
    df['textblob_polarity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_subjectivity'] = df['comment'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    sentiment_dist = df['sentiment'].value_counts()
    print(f"✓ Sentiment Distribution: {sentiment_dist.to_dict()}")
    print(f"  Average Sentiment Score: {df['sentiment_score'].mean():.3f}")
    
    return df

# ============================================================================
# STEP 4: NLP ANALYSIS - KEYWORD EXTRACTION
# ============================================================================
def extract_keywords(df, top_n=20):
    """Extract key topics using keyword extraction from open-text responses"""
    print("\n[STEP 4/7] NLP Analysis - Keyword Extraction...")
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(['work', 'company', 'employee', 'year', 'years'])
    
    all_keywords = []
    
    for comment in df['comment']:
        # Tokenize and POS tagging
        tokens = word_tokenize(comment.lower())
        tagged = pos_tag(tokens)
        
        # Extract nouns and adjectives (most meaningful for sentiment)
        keywords = [word for word, pos in tagged 
                   if pos.startswith(('NN', 'JJ')) 
                   and word.isalpha() 
                   and word not in stop_words
                   and len(word) > 3]
        all_keywords.extend(keywords)
    
    # Top keywords
    keyword_freq = Counter(all_keywords)
    top_keywords = keyword_freq.most_common(top_n)
    
    # Store keywords per comment
    def get_comment_keywords(text):
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        keywords = [word for word, pos in tagged 
                   if pos.startswith(('NN', 'JJ')) 
                   and word.isalpha() 
                   and word not in stop_words
                   and len(word) > 3]
        return ', '.join(keywords[:5])  # Top 5 per comment
    
    df['extracted_keywords'] = df['comment'].apply(get_comment_keywords)
    
    print(f"✓ Extracted {len(keyword_freq)} unique keywords")
    print(f"  Top 10 Keywords: {[k[0] for k in top_keywords[:10]]}")
    
    return df, top_keywords

# ============================================================================
# STEP 5: PREDICTIVE MODEL - LINEAR REGRESSION
# ============================================================================
def build_regression_model(df):
    """Build regression model to predict engagement scores"""
    print("\n[STEP 5/7] Building Regression Model (Predict Engagement)...")
    
    # Features for regression
    feature_cols = [
        'tenure', 'monthly_income', 'age', 'distance_from_home',
        'sentiment_score', 'job_satisfaction', 'environment_satisfaction', 
        'work_life_balance'
    ]
    
    # Encode categorical
    df['overtime_encoded'] = (df['overtime'] == 'Yes').astype(int)
    feature_cols.append('overtime_encoded')
    
    X = df[feature_cols].fillna(0)
    y = df['engagementScore']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression
    reg_model = LinearRegression()
    reg_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = reg_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    # Store predictions
    df['predicted_engagement'] = reg_model.predict(scaler.transform(X))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': reg_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"✓ Linear Regression Model Trained")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Top 3 Predictors: {list(feature_importance.head(3)['feature'])}")
    
    return df, reg_model, r2

# ============================================================================
# STEP 6: PREDICTIVE MODEL - CLUSTERING
# ============================================================================
def build_clustering_model(df):
    """Build K-Means clustering to identify attrition risk segments"""
    print("\n[STEP 6/7] Building Clustering Model (Attrition Risk Segments)...")
    
    # Features for clustering
    cluster_features = [
        'engagementScore', 'tenure', 'sentiment_score', 
        'monthly_income', 'distance_from_home'
    ]
    
    X_cluster = df[cluster_features].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # K-Means clustering (3 segments: Low, Medium, High risk)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['risk_cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters and assign risk levels
    cluster_analysis = df.groupby('risk_cluster').agg({
        'engagementScore': 'mean',
        'attrition': 'mean',
        'id': 'count'
    }).round(2)
    cluster_analysis.columns = ['avg_engagement', 'attrition_rate', 'count']
    
    # Map clusters to risk levels based on attrition rate
    risk_mapping = {}
    for cluster in cluster_analysis.index:
        if cluster_analysis.loc[cluster, 'attrition_rate'] > 0.25:
            risk_mapping[cluster] = 'High'
        elif cluster_analysis.loc[cluster, 'attrition_rate'] > 0.15:
            risk_mapping[cluster] = 'Medium'
        else:
            risk_mapping[cluster] = 'Low'
    
    df['turnover_risk'] = df['risk_cluster'].map(risk_mapping)
    
    # Also build logistic regression for attrition prediction
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df['attrition'], test_size=0.2, random_state=42
    )
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    df['attrition_probability'] = lr_model.predict_proba(X_scaled)[:, 1]
    
    print(f"✓ K-Means Clustering Complete")
    print(f"  Identified 3 Risk Segments: {df['turnover_risk'].value_counts().to_dict()}")
    print(f"  Logistic Regression Accuracy: {accuracy:.3f}")
    
    return df, kmeans, cluster_analysis.to_dict('index')

# ============================================================================
# STEP 7: GENERATE ANALYTICS OUTPUT
# ============================================================================
def generate_analytics(df, top_keywords, regression_r2, cluster_analysis):
    """Generate comprehensive analytics for dashboard"""
    print("\n[STEP 7/7] Generating Analytics Output...")
    
    # Overall metrics
    overall_metrics = {
        'total_records': len(df),
        'avg_engagement': round(df['engagementScore'].mean(), 2),
        'avg_sentiment_score': round(df['sentiment_score'].mean(), 3),
        'sentiment_breakdown': df['sentiment'].value_counts().to_dict(),
        'attrition_rate': round(df['attrition'].mean() * 100, 2),
        'high_risk_count': int((df['turnover_risk'] == 'High').sum()),
        'regression_r2': round(regression_r2, 3),
        'top_keywords': [{'keyword': k, 'count': c} for k, c in top_keywords[:15]]
    }
    
    # Department analytics
    dept_analytics = df.groupby('department').agg({
        'engagementScore': 'mean',
        'sentiment_score': 'mean',
        'attrition': lambda x: (x == 1).sum(),
        'id': 'count'
    }).round(2).to_dict('index')
    
    # Sentiment by department
    sentiment_by_dept = df.groupby(['department', 'sentiment']).size().unstack(fill_value=0).to_dict('index')
    
    # Risk segment analysis
    risk_distribution = df['turnover_risk'].value_counts().to_dict()
    
    # Keyword sentiment correlation
    keyword_sentiment = {}
    for keyword, _ in top_keywords[:10]:
        mask = df['extracted_keywords'].str.contains(keyword, na=False)
        if mask.sum() > 0:
            keyword_sentiment[keyword] = {
                'avg_sentiment': round(df[mask]['sentiment_score'].mean(), 3),
                'count': int(mask.sum())
            }
    
    # Export comprehensive JSON
    output = {
        'overall_metrics': overall_metrics,
        'department_analytics': dept_analytics,
        'sentiment_by_department': sentiment_by_dept,
        'cluster_analysis': cluster_analysis,
        'risk_distribution': risk_distribution,
        'keyword_sentiment': keyword_sentiment,
        'sample_data': df.head(100).to_dict('records')
    }
    
    with open('analytics_output.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Export full processed data
    df.to_csv('processed_employee_data.csv', index=False)
    
    print(f"✓ Analytics Complete!")
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total Records Analyzed: {len(df)}")
    print(f"Average Engagement Score: {overall_metrics['avg_engagement']}/100")
    print(f"Sentiment Distribution: {overall_metrics['sentiment_breakdown']}")
    print(f"Attrition Rate: {overall_metrics['attrition_rate']}%")
    print(f"High-Risk Employees: {overall_metrics['high_risk_count']}")
    print(f"Regression Model R²: {overall_metrics['regression_r2']}")
    print(f"Top Keywords: {[k['keyword'] for k in overall_metrics['top_keywords'][:5]]}")
    print(f"\nOutput Files:")
    print(f"  ✓ analytics_output.json")
    print(f"  ✓ processed_employee_data.csv")
    print(f"{'='*70}\n")
    
    return output

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Execute complete analytics pipeline"""
    
    # Step 1: Load data
    df_raw = load_kaggle_data()
    
    # Step 2: Transform
    df = transform_data(df_raw)
    
    # Step 3: Sentiment analysis
    df = analyze_sentiment(df)
    
    # Step 4: Keyword extraction
    df, top_keywords = extract_keywords(df)
    
    # Step 5: Regression model
    df, reg_model, r2 = build_regression_model(df)
    
    # Step 6: Clustering model
    df, kmeans_model, cluster_analysis = build_clustering_model(df)
    
    # Step 7: Generate analytics
    output = generate_analytics(df, top_keywords, r2, cluster_analysis)
    
    return output

if __name__ == "__main__":
    results = run_pipeline()