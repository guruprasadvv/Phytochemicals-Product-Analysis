import os
import re
import string
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# NLP / ML imports
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)

# HuggingFace classifiers
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
    print("HuggingFace transformers detected — will use DistilBERT classifier.")
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace not found — using TF-IDF + Logistic Regression classifier.")

# SentenceTransformers for Semantic ABSA
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    ST_AVAILABLE = True
    print("SentenceTransformers detected — will use semantic aspect extraction.")
except ImportError:
    ST_AVAILABLE = False
    raise ImportError("Please install sentence-transformers (pip install sentence-transformers torch).")

# BERTopic for Advanced Topic Modeling
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
    print("BERTopic detected — will use advanced transformer-based topic modeling.")
except ImportError:
    BERTOPIC_AVAILABLE = False
    raise ImportError("Please install bertopic (pip install bertopic) to run Step 4.")

# ── Download NLTK assets ───────────────────────────────────────────────────────
for resource in ['corpora/stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

# ── Output directory 
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ==============================================================================
# STEP 0 — Load Dataset
#
CSV_PATH = "combine_data.csv"

REQUIRED_COLUMNS = {
    'review': ['Review Body', 'review_text', 'reviewText', 'review',
               'text', 'comment', 'Review'],
    'label':  ['Rating', 'label', 'sentiment', 'rating',
               'score', 'Label', 'overall', 'stars'],
    'helpful': ['Helpful', 'helpful_votes', 'helpfulVotes',
                'helpful_count', 'HelpfulCount'],
    'product': ['Product Name', 'product_name', 'ProductName',
                'product', 'asin', 'ASIN'],
}

def detect_column(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

print(f"Loading dataset from: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"  → Loaded {len(df):,} rows, {len(df.columns)} columns")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find '{CSV_PATH}'. "
        "Place the CSV in the same directory as this script."
    )

review_col  = detect_column(df.columns, REQUIRED_COLUMNS['review'])
label_col   = detect_column(df.columns, REQUIRED_COLUMNS['label'])
helpful_col = detect_column(df.columns, REQUIRED_COLUMNS['helpful'])
product_col = detect_column(df.columns, REQUIRED_COLUMNS['product'])

if review_col is None:
    raise ValueError(f"No review text column found. Expected one of {REQUIRED_COLUMNS['review']}.")

# Standardise column names
df = df.rename(columns={review_col: 'review_text'})
if label_col and label_col != 'label':
    df = df.rename(columns={label_col: 'label_raw'})
elif label_col:
    df = df.rename(columns={'label': 'label_raw'})

if helpful_col and helpful_col != 'helpful_votes':
    df = df.rename(columns={helpful_col: 'helpful_votes'})
if product_col and product_col != 'product_name':
    df = df.rename(columns={product_col: 'product_name'})

# Drop missing reviews
df = df.dropna(subset=['review_text']).reset_index(drop=True)
df['review_text'] = df['review_text'].astype(str)
df = df[df['review_text'].str.strip().str.len() > 3].reset_index(drop=True)

# Derive BOTH binary and 5-class labels 
if 'label_raw' in df.columns:
    df['label_raw'] = pd.to_numeric(df['label_raw'], errors='coerce')
    max_val = df['label_raw'].max()

    if max_val > 1:
        df['label_5class'] = (df['label_raw'].fillna(3) - 1).clip(0, 4).astype(int)
        df['label'] = (df['label_raw'] >= 4).astype(int)
    else:
        df['label'] = df['label_raw'].fillna(0).astype(int)
        df['label_5class'] = df['label'] * 4 
else:
    df['label'] = df['review_text'].apply(
        lambda x: int(TextBlob(str(x)).sentiment.polarity > 0)
    )
    df['label_5class'] = df['label'] * 4

# Helpful votes default
if 'helpful_votes' in df.columns:
    df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0)
else:
    df['helpful_votes'] = 0.0

print(f"  → {len(df):,} reviews after cleaning.\n")


# ==============================================================================
# STEP 1 — Overall Sentiment Score
#
print("STEP 1: Computing TextBlob sentiment scores …")

df['sentiment_score'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda s: 'positive' if s > 0.05 else ('negative' if s < -0.05 else 'neutral')
)

sent_dist = df['sentiment_label'].value_counts()
print(f"  → Sentiment distribution:\n{sent_dist.to_string()}\n")


# ==============================================================================
# STEP 2 — Semantic Aspect-Based Sentiment Analysis
#
print("STEP 2: Running Semantic Aspect-Based Sentiment Analysis ...")

aspect_queries = {
    'effectiveness': 'product effectiveness, pain relief, works well',
    'price': 'price, cost, expensive, value for money',
    'packaging': 'packaging, bottle, seal, broken container',
    'taste': 'taste, flavor, smell, hard to swallow',
    'delivery': 'shipping, delivery time, tracking',
    'side_effects': 'side effects, nausea, headache, stomach pain',
    'authenticity': 'fake product, counterfeit, not authentic',
    'health_impact': 'health impact, immunity, energy, wellness'
}

aspect_names = list(aspect_queries.keys())
aspect_prompts = list(aspect_queries.values())

st_model = SentenceTransformer('all-MiniLM-L6-v2')
aspect_embeddings = st_model.encode(aspect_prompts, convert_to_tensor=True)

NEGATION_WORDS = {'not', 'no', "n't", 'never', 'without', 'hardly', 'barely'}

def get_negation_aware_polarity(sentence):
    tokens = sentence.lower().split()
    for i, tok in enumerate(tokens):
        window = set(tokens[max(0, i - 3): i])
        if window & NEGATION_WORDS:
            sentence = "not " + sentence
            break
    return TextBlob(sentence).sentiment.polarity

aspect_rows = []
SIMILARITY_THRESHOLD = 0.35 

for idx, row in df.iterrows():
    review_raw = row['review_text']
    helpful    = row.get('helpful_votes', 0)
    
    sentences = [s.strip() for s in re.split(r'[.,!;\n]', review_raw) if len(s.strip()) > 5]
    if not sentences:
        continue
        
    sentence_embeddings = st_model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(sentence_embeddings, aspect_embeddings)
    
    for col_idx, aspect in enumerate(aspect_names):
        max_score_idx = torch.argmax(cosine_scores[:, col_idx]).item()
        max_score = cosine_scores[max_score_idx, col_idx].item()
        
        if max_score >= SIMILARITY_THRESHOLD:
            relevant_sentence = sentences[max_score_idx]
            polarity = get_negation_aware_polarity(relevant_sentence)
            sentiment = ('positive' if polarity > 0.05 else ('negative' if polarity < -0.05 else 'neutral'))
            
            aspect_rows.append({
                'Review':        review_raw,
                'Aspect':        aspect,
                'Sentiment':     sentiment,
                'Score':         round(polarity, 4),
                'Helpful_Votes': helpful,
            })

df_aspect = pd.DataFrame(aspect_rows)
df_aspect.to_csv(out("Deliverable1_Aspect_Sentiment.csv"), index=False)
print(f"  → {len(df_aspect):,} aspect-sentiment records saved.\n")


# ==============================================================================
# STEP 3 — Text Cleaning (Kept for downstream TF-IDF / Baseline Classifier)
print("STEP 3: Cleaning review text …")

stop_words        = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r'\d+', '', text)
    text  = text.translate(punctuation_table)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

df['clean_review'] = df['review_text'].apply(clean_text)
print("  → Text cleaning complete.\n")


# ==============================================================================
# STEP 4 — Topic Modeling: BERTopic
print("STEP 4: Running BERTopic Modeling …")

# Using raw text because transformers handle punctuation and stop words internally
docs = df['review_text'].tolist()

# Limit to 6 topics (5 core + 1 outlier class) to maintain chart readability
topic_model = BERTopic(language="english", calculate_probabilities=False, nr_topics=6)
topics, probs = topic_model.fit_transform(docs)

df['dominant_topic_id'] = topics
topic_info = topic_model.get_topic_info()

topic_data = []
for _, row in topic_info.iterrows():
    t_id = row['Topic']
    count = row['Count']
    name = row['Name']
    
    if t_id == -1:
        business_label = "Uncategorized / Outliers"
        keywords = "noise, mixed, unassigned"
    else:
        # Extract top 10 words for the topic
        top_words = [word for word, _ in topic_model.get_topic(t_id)[:10]]
        keywords = ", ".join(top_words)
        business_label = f"Key Topic {t_id + 1}"
        
    topic_data.append({
        'Topic_ID':       t_id,
        'Business_Label': business_label,
        'Keywords':       keywords,
        'Review_Count':   count,
    })

df_topics = pd.DataFrame(topic_data)
df_topics.to_csv(out("Deliverable2_Topic_Table.csv"), index=False)
print("  → BERTopic table saved.\n")


# ==============================================================================
# STEP 5 — Market Opportunity Quantification
# ==============================================================================
print("STEP 5: Quantifying market opportunities …")

opp_rows = []
for aspect in aspect_names:
    subset    = df_aspect[df_aspect['Aspect'] == aspect]
    total     = len(subset)
    neg       = len(subset[subset['Sentiment'] == 'negative'])
    pos       = len(subset[subset['Sentiment'] == 'positive'])
    avg_score = subset['Score'].mean() if total > 0 else 0.0
    neg_ratio = neg / total if total > 0 else 0.0
    
    avg_helpful = max(subset['Helpful_Votes'].mean(), 1.0) if total > 0 else 1.0
    opp_score = round(neg_ratio * np.log1p(total) * np.log1p(avg_helpful) * 10, 2)

    opp_rows.append({
        'Aspect':              aspect,
        'Total_Mentions':      total,
        'Positive_Count':      pos,
        'Negative_Count':      neg,
        'Avg_Sentiment_Score': round(avg_score, 4),
        'Negative_Ratio_%':    round(neg_ratio * 100, 1),
        'Avg_Helpful_Votes':   round(avg_helpful, 2),
        'Opportunity_Score':   opp_score,
    })

df_opp = pd.DataFrame(opp_rows).sort_values('Opportunity_Score', ascending=False)
df_opp.to_csv(out("Deliverable4_Market_Opportunity.csv"), index=False)
print("  → Market opportunity table saved.\n")


# ==============================================================================
# STEP 6 — Authenticity & Side-Effect Flags 
# ==============================================================================
print("STEP 6: Flagging authenticity concerns and side effects …")

auth_keywords = ['fake', 'counterfeit', 'adulterated', 'spurious', 'imitation']
side_effect_keywords = ['nausea', 'headache', 'allergy', 'reaction', 'rash', 'dizziness']

df['authenticity_flag'] = df['review_text'].apply(
    lambda x: int(any(kw in x.lower() for kw in auth_keywords))
)
df['side_effect_flag'] = df['review_text'].apply(
    lambda x: int(any(kw in x.lower() for kw in side_effect_keywords))
)

flagged_df = df[['review_text', 'sentiment_label', 'sentiment_score',
                 'authenticity_flag', 'side_effect_flag']].copy()
flagged_df.to_csv(out("Deliverable5_Flagged_Reviews.csv"), index=False)

auth_count        = df['authenticity_flag'].sum()
side_effect_count = df['side_effect_flag'].sum()
print(f"  → {auth_count} reviews flagged for authenticity concerns.")
print(f"  → {side_effect_count} reviews flagged for side effects.\n")


# ==============================================================================
# STEP 7 — Product Comparison Table
if 'product_name' in df.columns:
    print("STEP 7: Building Product Comparison Table …")
    rating_col_src = 'label_raw' if 'label_raw' in df.columns else 'label'

    prod_table = (
        df.groupby('product_name')
        .agg(
            Avg_Rating      =(rating_col_src, lambda x: round(pd.to_numeric(x, errors='coerce').mean(), 2)),
            Review_Count    =('review_text', 'count'),
            Positive_Pct    =('label', lambda x: round(x.mean() * 100, 1)),
            Avg_Sentiment   =('sentiment_score', lambda x: round(x.mean(), 4)),
            Side_Effect_Pct =('side_effect_flag', lambda x: round(x.mean() * 100, 1)),
            Auth_Flag_Pct   =('authenticity_flag', lambda x: round(x.mean() * 100, 1)),
        )
        .query('Review_Count >= 5')
        .sort_values('Avg_Rating', ascending=False)
        .reset_index()
        .rename(columns={'product_name': 'Product_Name'})
    )

    def sentiment_tier(pct):
        if pct >= 70: return 'Positive'
        if pct >= 50: return 'Mixed'
        return 'Negative'

    prod_table['Sentiment_Tier'] = prod_table['Positive_Pct'].apply(sentiment_tier)
    prod_table.to_csv(out("Deliverable6_Product_Comparison.csv"), index=False)
    print(f"  → Product comparison table: {len(prod_table)} products (min 5 reviews).\n")


# ==============================================================================
# STEP 8 — Business Insights Report
print("STEP 8: Generating Business Insights Report …")

total_reviews  = len(df)
pos_pct        = round(df[df['sentiment_label'] == 'positive'].shape[0] / total_reviews * 100, 1)
neg_pct        = round(df[df['sentiment_label'] == 'negative'].shape[0] / total_reviews * 100, 1)
neu_pct        = round(100 - pos_pct - neg_pct, 1)

# Find top topic excluding outliers (-1)
valid_topics = df_topics[df_topics['Topic_ID'] != -1]
if not valid_topics.empty:
    top_topic = valid_topics.sort_values('Review_Count', ascending=False).iloc[0]['Keywords']
else:
    top_topic = "N/A"
    
top_opp_aspect = df_opp.iloc[0]['Aspect'] if len(df_opp) > 0 else "N/A"
products = df['product_name'].nunique() if 'product_name' in df.columns else "N/A"
pkg_neg = len(df_aspect[(df_aspect['Aspect'] == 'packaging') & (df_aspect['Sentiment'] == 'negative')])
tot_neg = len(df_aspect[df_aspect['Sentiment'] == 'negative'])
pkg_pct = round(pkg_neg / tot_neg * 100, 1) if tot_neg > 0 else 0.0

insights = f"""
==============================================================================
DELIVERABLE 3: BUSINESS INSIGHTS — PHYTOCHEMICAL WELLNESS PRODUCTS

DATASET OVERVIEW
----------------
  Total Reviews Analysed : {total_reviews:,}
  Unique Products        : {products}
  Positive Sentiment     : {pos_pct}%
  Neutral  Sentiment     : {neu_pct}%
  Negative Sentiment     : {neg_pct}%

TOPIC LANDSCAPE (BERTopic)
--------------------------
  Dominant Concept       : {top_topic}
  Note: Refer to Deliverable2_Topic_Table.csv for the full cluster mapping.

ASPECT SENTIMENT HIGHLIGHTS
----------------------------
  Highest Market Opportunity Aspect : '{top_opp_aspect}'
  Packaging Negative Share          : {pkg_pct:.1f}% of all negative aspect mentions

AUTHENTICITY & SAFETY FLAGS
----------------------------
  Authenticity Concerns  : {auth_count} reviews ({round(auth_count/total_reviews*100,1)}%)
  Side-Effect Mentions   : {side_effect_count} reviews ({round(side_effect_count/total_reviews*100,1)}%)
==============================================================================
"""
with open(out("Deliverable3_Business_Insights.txt"), "w") as f:
    f.write(insights)
print("  → Insights generated.\n")


# ==============================================================================
# STEP 9 — Visualisations
# ==============================================================================
print("STEP 9: Generating charts …")
COLORS = {'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'}

# Chart 1
sentiment_counts = df['sentiment_label'].value_counts()
ordered_labels   = [l for l in ['positive', 'neutral', 'negative'] if l in sentiment_counts.index]
colors_ordered   = [COLORS[l] for l in ordered_labels]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(ordered_labels, [sentiment_counts[l] for l in ordered_labels], color=colors_ordered, edgecolor='white')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 5, f'{h:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Overall Sentiment Distribution', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(out("Chart1_Sentiment_Distribution.png"), dpi=150)
plt.close()

# Chart 2
aspect_pivot = df_aspect.groupby(['Aspect', 'Sentiment']).size().unstack(fill_value=0).reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(aspect_pivot.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(aspect_pivot.columns)))
ax.set_xticklabels(aspect_pivot.columns, fontsize=11)
ax.set_yticks(range(len(aspect_pivot.index)))
ax.set_yticklabels(aspect_pivot.index, fontsize=10)
for i in range(aspect_pivot.shape[0]):
    for j in range(aspect_pivot.shape[1]):
        val = aspect_pivot.values[i, j]
        ax.text(j, i, f'{val:,}', ha='center', va='center', fontsize=9, fontweight='bold', color='black' if val < aspect_pivot.values.max() * 0.7 else 'white')
plt.colorbar(im, ax=ax, label='Review Count')
ax.set_title('Aspect-Sentiment Heatmap', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(out("Chart2_Aspect_Sentiment_Heatmap.png"), dpi=150)
plt.close()

# Chart 3
fig, ax = plt.subplots(figsize=(9, 5))
df_opp_plot = df_opp.sort_values('Opportunity_Score')
bar_colors  = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_opp_plot)))
ax.barh(df_opp_plot['Aspect'], df_opp_plot['Opportunity_Score'], color=bar_colors, edgecolor='white')
ax.set_title('Market Opportunity by Aspect', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(out("Chart3_Market_Opportunity.png"), dpi=150)
plt.close()

# Chart 4 (Updated for BERTopic)
fig, ax = plt.subplots(figsize=(10, 5))
# Generate enough colors for the dynamic number of topics
cmap = plt.get_cmap('tab10')
bar_colors2 = [cmap(i % 10) for i in range(len(df_topics))]

ax.bar(df_topics['Business_Label'], df_topics['Review_Count'], color=bar_colors2, edgecolor='white')
for i, v in enumerate(df_topics['Review_Count']):
    ax.text(i, v + (df_topics['Review_Count'].max() * 0.02), str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title(f'BERTopic Distribution', fontsize=12, fontweight='bold', pad=12)
plt.xticks(rotation=25, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(out("Chart4_Topic_Distribution.png"), dpi=150)
plt.close()

print("  → Visualisations saved.\n")


# ==============================================================================
# STEP 10 — Sentiment Classifier
print("STEP 10: Training Sentiment Classifier …\n")

tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=2)
X_all = tfidf.fit_transform(df['clean_review'])
y_all = df['label'].values

X_train, X_val, y_train, y_val, text_train, text_val = train_test_split(
    X_all, y_all, df['review_text'], test_size=0.2, random_state=42, stratify=y_all
)

if HF_AVAILABLE:
    print("  → Loading DistilBERT pipeline …")
    clf_pipeline = hf_pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    
    val_texts  = text_val.tolist()
    hf_results = clf_pipeline(val_texts, batch_size=32)
    y_pred     = np.array([1 if r['label'] == 'POSITIVE' else 0 for r in hf_results])
    model_name = "DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)"

else:
    print("  → Training TF-IDF + Logistic Regression classifier …")
    clf = LogisticRegression(C=1.0, max_iter=1_000, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)
    y_pred     = clf.predict(X_val)
    model_name = "TF-IDF + Logistic Regression"

acc = accuracy_score(y_val, y_pred)
print(f"\n  ─── {model_name} ───")
print(f"  Validation Accuracy : {acc*100:.2f}%\n")
print(classification_report(y_val, y_pred, target_names=['Negative (0)', 'Positive (1)']))

cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix\n{model_name}', fontsize=11, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(out("Chart6_Confusion_Matrix.png"), dpi=150)
plt.close()

# ── End of Pipeline Output Summary ──
print("\n─── Pipeline complete. ───\n")
print("Output files created:")

outputs = [
    "Deliverable1_Aspect_Sentiment.csv",
    "Deliverable2_Topic_Table.csv",
    "Deliverable3_Business_Insights.txt", 
    "Deliverable4_Market_Opportunity.csv",
    "Deliverable5_Flagged_Reviews.csv",
    "Deliverable6_Product_Comparison.csv",
    "Chart1_Sentiment_Distribution.png",
    "Chart2_Aspect_Sentiment_Heatmap.png",
    "Chart3_Market_Opportunity.png",
    "Chart4_Topic_Distribution.png",
    "Chart6_Confusion_Matrix.png",
]

for i, fname in enumerate(outputs, 1):
    path   = out(fname)
    status = "✓" if os.path.exists(path) else "○ (skipped)"
    print(f"  {status}  {i:2}. {path}")