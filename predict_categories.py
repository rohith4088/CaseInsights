import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Parse HTML Categories
html_path = 'support_categories_table.html'
with open(html_path, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

valid_categories = []
for td in soup.find_all("td"):
    span = td.find("span", class_="code")
    if span:
        cat_name = td.text.replace(span.text, "").strip()
        if cat_name and cat_name != "—":
            valid_categories.append(cat_name)

print(f"Extracted {len(valid_categories)} valid categories from HTML.")
cat_lower_map = {c.lower(): c for c in valid_categories}

# 2. Read Excel File
excel_path = 'Closed cases 1.xlsx'
df = pd.read_excel(excel_path)
print(f"Loaded {len(df)} rows from Excel.")

# 3. Preprocess Text Columns
def safe_str(x):
    if pd.isna(x): return ""
    return str(x).strip()

df['Combined_Text'] = df['Subject'].apply(safe_str) + " " + \
                      df['Issue Plain Text'].apply(safe_str) + " " + \
                      df['Other Issue Category Description'].apply(safe_str)

# 4. TF-IDF Setup
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
cat_tfidf = vectorizer.fit_transform(valid_categories)

predicted_categories = []
confidence_scores = []

for idx, row in df.iterrows():
    desc = safe_str(row.get('Other Issue Category Description', '')).lower()
    cat = safe_str(row.get('Issue Category', '')).lower()
    
    matched = False
    for valid_cat, orig_cat in cat_lower_map.items():
        if valid_cat in desc:
            predicted_categories.append(orig_cat)
            confidence_scores.append(1.0)
            matched = True
            break
            
    if matched: continue
    
    for valid_cat, orig_cat in cat_lower_map.items():
        if valid_cat == cat:
            predicted_categories.append(orig_cat)
            confidence_scores.append(1.0)
            matched = True
            break
            
    if matched: continue

    text = row['Combined_Text']
    if not text.strip():
        predicted_categories.append("Others")
        confidence_scores.append(0.0)
        continue
        
    text_tfidf = vectorizer.transform([text])
    sims = cosine_similarity(text_tfidf, cat_tfidf)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    
    if best_score > 0.05:
        predicted_categories.append(valid_categories[best_idx])
        confidence_scores.append(float(best_score))
    else:
        predicted_categories.append("Others")
        confidence_scores.append(float(best_score))

df['Predicted_Category'] = predicted_categories
df['Prediction_Confidence'] = confidence_scores

# 5. Export to Excel
out_path = 'Categorized_Cases.xlsx'
df.to_excel(out_path, index=False)
print(f"Saved categorized cases to {out_path}.")

# 6. Visualization
artifact_dir = "/Users/rohithr/.gemini/antigravity/brain/b9d746db-2702-4f9f-a0b1-3c49cd5ecfa5"
os.makedirs(artifact_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
top_cats = df['Predicted_Category'].value_counts().head(15)
sns.barplot(x=top_cats.values, y=top_cats.index, palette="viridis")
plt.title("Top 15 Predicted Issue Categories")
plt.xlabel("Number of Cases")
plt.ylabel("Category")
plt.tight_layout()
bar_path = os.path.join(artifact_dir, "top_predictions.png")
plt.savefig(bar_path, dpi=150)
plt.close()

main_headers = []
for th in soup.find_all("th"):
    span = th.find("span", class_="cat-num")
    if span:
        hdr_name = th.text.replace(span.text, "").strip()
        main_headers.append(hdr_name)

cat_to_main = {}
for td in soup.find_all("td"):
    span = td.find("span", class_="code")
    if span:
        cat_name = td.text.replace(span.text, "").strip()
        code = span.text.strip()
        main_idx = int(code.split('.')[0])
        if main_idx < len(main_headers):
            cat_to_main[cat_name] = main_headers[main_idx]

df['Main_Category'] = df['Predicted_Category'].map(lambda x: cat_to_main.get(x, "Others"))

plt.figure(figsize=(10, 8))
main_counts = df['Main_Category'].value_counts()
plt.pie(main_counts, labels=main_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3", len(main_counts)))
plt.title("Distribution of Cases by Main Category")
plt.axis('equal')
plt.tight_layout()
pie_path = os.path.join(artifact_dir, "category_distribution.png")
plt.savefig(pie_path, dpi=150)
plt.close()

print(f"Generated visualizations at {artifact_dir}")
