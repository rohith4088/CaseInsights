import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import os, json, hashlib, joblib

# Highest-priority keyword rules (domain-specific)
KEYWORD_BOOSTER = {
    "onboarding": "Unable to onboard device",
    "onboard": "Unable to onboard device",
    "sso": "SSO / SAML Configuration",
    "saml": "SSO / SAML Configuration",
    "single sign-on": "SSO / SAML Configuration",
    "access denied": "Access Denied",
    "login fail": "Unable to login to platform",
    "unable to login": "Unable to login to platform",
    "cannot login": "Unable to login to platform",
    "forgot password": "Invalid / Forgot Password",
    "reset password": "Unable to reset password",
    "workspace creation": "Workspace creation, editing, access error",
    "workspace access": "Workspace creation, editing, access error",
    "assign device": "Unable to assign device",
    "subscription key": "Requires subscription key",
    "license extension": "License Extension",
    "subscription transfer": "Subscription Transfer",
    "device transfer": "Subscription Transfer",
    "invite user": "Unable to invite user",
    "no roles": "No Roles Assigned",
    "network issue": "Network Issue",
    "browser issue": "Browser issue",
    "local gateway": "Local gateway VM connection failure",
    "gateway vm": "Unable to access local gateway VM",
    "ova": "Unable to access or download OVA",
    "registration token": "Registration Token missing or mismatch",
    "merge account": "Merge accounts",
    "delete account": "Account Deletion",
    "account blocked": "Action Not Permitted / Account Blocked",
    "delete mfa": "Delete MFA",
    "firmware": "Application / Patch / Firmware issue",
    "user sync": "User sync issue",
    "partner portal": "Unable to login to partner portal",
    "outage": "Outage, Maintenance",
    "fail prov": "Fail Prov No Rule",
    "export devices": "Unable to export devices",
    "create tags": "Unable to create tags",
    "heartbeat": "Activation, Register, Logs, Heartbeat failure",
    "initialize device": "Unable to initialize device",
    "setup device": "Unable to setup device",
    "aruba central": "Unable to access Aruba central",
    "location mapping": "Location mapping failure",
    "tenant sync": "Tenant sync issue",
    "system time": "System time issue",
    "invalid url": "Invalid / Incorrect URL",
    "glcs": "GLCS",
    "no business account": "No business account",
    "unable to create account": "Unable to create account",
    "activate role": "Activate role not available",
    "product number": "Product number mis-match",
    "model mismatch": "Device model mis-match",
    "acc email": "ACC email notification failure / missing data",
    "auto case": "Auto case creation failure",
    "misroute": "Misroute",
    "enhancement": "Enhancement",
    "missing information": "Missing information in dashboard",
    "export data": "Export data failure",
    "gtc service": "Local gateway GTC service error",
    "install": "Install / Configuration / Registration error",
    "configuration error": "Install / Configuration / Registration error",
    "ui error": "UI Error",
    "new user registration": "New User Registration",
    "unknown device": "Unknown devices",
}


class CategoryPredictor:
    def __init__(self, html_path=None):
        # Resolve path relative to this file so it works in Docker and locally
        if html_path is None:
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'support_categories_table.html')
        self.html_path = html_path
        self.valid_categories = []
        self.cat_lower_map = {}
        self.main_headers = []
        self.cat_to_main = {}
        self.supervised_classifier = None

        print("Loading Semantic Embedding Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cat_embeddings = None
        self._load_taxonomy()

    def _load_taxonomy(self):
        with open(self.html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        for td in soup.find_all("td"):
            span = td.find("span", class_="code")
            if span:
                cat_name = td.text.replace(span.text, "").strip()
                if cat_name and cat_name != "—":
                    self.valid_categories.append(cat_name)

        self.cat_lower_map = {c.lower(): c for c in self.valid_categories}

        if self.valid_categories:
            print(f"Pre-computing embeddings for {len(self.valid_categories)} categories...")
            self.cat_embeddings = self.model.encode(self.valid_categories)

        for th in soup.find_all("th"):
            span = th.find("span", class_="cat-num")
            if span:
                self.main_headers.append(th.text.replace(span.text, "").strip())

        for td in soup.find_all("td"):
            span = td.find("span", class_="code")
            if span:
                cat_name = td.text.replace(span.text, "").strip()
                code = span.text.strip()
                main_idx = int(code.split('.')[0])
                if main_idx < len(self.main_headers):
                    self.cat_to_main[cat_name] = self.main_headers[main_idx]

    def _safe_str(self, x):
        if pd.isna(x): return ""
        return str(x).strip()

    def _make_text(self, row):
        """Combine all relevant columns for rich context."""
        fields = ['Subject', 'Issue Plain Text', 'Other Issue Category Description',
                  'Cause', 'Resolution', 'Resolution Type', 'Resolution Code']
        parts = []
        for f in fields:
            val = self._safe_str(row.get(f, ''))
            if val:
                parts.append(val)
        return " ".join(parts)

    def _normalize_label(self, text):
        """Map a raw data label to a valid taxonomy category using multi-strategy fuzzy matching."""
        import re, difflib
        if not text or str(text).strip().lower() in ("", "other", "nan", "others", "general query"):
            return None
        t = str(text).strip()

        # Strip code prefixes like '4.7 Unable to...' or '4.18 Device Transfer'
        t = re.sub(r'^\d+\.\d+\s+', '', t)

        # Normalize punctuation differences (slashes/commas/spaces)
        def norm(s):
            return re.sub(r'[\s/,;]+', ' ', s).strip().lower()

        t_norm = norm(t)

        # 1. Exact normalized match
        for vl, vo in self.cat_lower_map.items():
            if norm(vl) == t_norm:
                return vo

        # 2. Substring match
        for vl, vo in self.cat_lower_map.items():
            n_vl = norm(vl)
            if n_vl in t_norm or t_norm in n_vl:
                return vo

        # 3. Fuzzy match with difflib (handles typos, word order variants)
        candidates = list(self.cat_lower_map.keys())
        norm_candidates = [norm(c) for c in candidates]
        matches = difflib.get_close_matches(t_norm, norm_candidates, n=1, cutoff=0.72)
        if matches:
            idx = norm_candidates.index(matches[0])
            return self.cat_lower_map[candidates[idx]]

        return None

    def _train_supervised(self, df, extra_texts=None, extra_labels=None):
        """Train a CalibratedLinearSVC. Caches model to disk; reloads if data fingerprint matches."""
        texts, labels = [], []
        for _, row in df.iterrows():
            label = None
            for col in ['Other Issue Category Description', 'Issue Category']:
                val = self._safe_str(row.get(col, ''))
                norm = self._normalize_label(val)
                if norm:
                    label = norm
                    break
            if label:
                text = self._make_text(row)
                if text:
                    texts.append(text)
                    labels.append(label)

        # Inject active learning corrections
        if extra_texts and extra_labels:
            texts.extend(extra_texts)
            labels.extend(extra_labels)
            print(f"Active learning: injecting {len(extra_texts)} corrections into training.")

        if len(texts) < 50 or len(set(labels)) < 5:
            print(f"Insufficient labeled data ({len(texts)} rows). Skipping supervised training.")
            return None

        from collections import Counter
        label_counts = Counter(labels)
        filtered = [(t, l) for t, l in zip(texts, labels) if label_counts[l] >= 3]
        if len(filtered) < 50:
            print("Too few usable labeled rows after filtering rare classes. Skipping.")
            return None
        texts, labels = zip(*filtered)
        texts, labels = list(texts), list(labels)

        # Cache fingerprint: hash of sorted unique labels + count
        fingerprint = hashlib.md5(",".join(sorted(set(labels))).encode()).hexdigest()[:8]
        cache_path = os.path.join(os.path.dirname(__file__), f"model_cache_{fingerprint}.pkl")

        if os.path.exists(cache_path) and not (extra_texts):
            print(f"Loading cached model ({fingerprint})...")
            return joblib.load(cache_path)

        min_class_count = min(Counter(labels).values())
        cv = min(3, min_class_count)
        print(f"Training supervised classifier on {len(texts)} rows, {len(set(labels))} categories (cv={cv})...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=8000,
                                      sublinear_tf=True, min_df=2)),
            ('clf', CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0, class_weight='balanced'), cv=cv))
        ])
        pipeline.fit(texts, labels)
        sample = min(500, len(texts))
        acc = sum(p == l for p, l in zip(pipeline.predict(texts[:sample]), labels[:sample])) / sample
        print(f"Training accuracy (sample): {acc * 100:.1f}%")
        joblib.dump(pipeline, cache_path)
        return pipeline

    def retrain_with_feedback(self, feedback_path, current_df=None):
        """Active learning: incorporate human corrections and retrain."""
        if not os.path.exists(feedback_path):
            return {"status": "no_feedback"}
        with open(feedback_path) as f:
            feedback = json.load(f)
        extra_texts = [fb['text'] for fb in feedback if fb.get('text') and fb.get('corrected_category')]
        extra_labels = [fb['corrected_category'] for fb in feedback if fb.get('text') and fb.get('corrected_category')]
        if not extra_texts:
            return {"status": "no_text_in_feedback"}
        if current_df is not None:
            self.supervised_classifier = self._train_supervised(current_df, extra_texts, extra_labels)
        else:
            # Minimal retrain on just the feedback
            from collections import Counter
            if len(extra_texts) >= 5:
                label_counts = Counter(extra_labels)
                ft = [(t, l) for t, l in zip(extra_texts, extra_labels) if label_counts[l] >= 1]
                if ft:
                    t_list, l_list = zip(*ft)
                    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(stop_words='english')),
                        ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=min(2, min(Counter(l_list).values()))))
                    ])
                    try:
                        pipeline.fit(list(t_list), list(l_list))
                        self.supervised_classifier = pipeline
                    except Exception as e:
                        print(f"Feedback retrain error: {e}")
        return {"status": "retrained", "feedback_used": len(extra_texts)}

    def predict(self, df):
        # Train supervised classifier from this file's labeled data
        self.supervised_classifier = self._train_supervised(df)

        df['Combined_Text'] = df.apply(self._make_text, axis=1)
        n = len(df)
        predicted_categories = ["Others"] * n
        confidence_scores = [0.0] * n
        semantic_indices, semantic_texts = [], []

        # Store top-3 predictions per row (from supervised classifier)
        top3_list = [[]] * n

        for i, (_, row) in enumerate(df.iterrows()):
            text_l = row['Combined_Text'].lower()
            desc_l = self._safe_str(row.get('Other Issue Category Description', '')).lower()

            # Tier 1: Keyword booster
            matched = False
            for kw, cat in KEYWORD_BOOSTER.items():
                if kw in desc_l or kw in text_l:
                    if cat in self.cat_lower_map.values():
                        predicted_categories[i] = cat
                        confidence_scores[i] = 0.98
                        matched = True
                        break
            if matched:
                continue

            # Tier 2: Exact substring match in description
            for vl, vo in self.cat_lower_map.items():
                if vl in desc_l:
                    predicted_categories[i] = vo
                    confidence_scores[i] = 1.0
                    matched = True
                    break
            if matched:
                continue

            # Tier 3: Supervised classifier — capture full top-3
            if self.supervised_classifier and row['Combined_Text'].strip():
                proba = self.supervised_classifier.predict_proba([row['Combined_Text']])[0]
                best_i = int(np.argmax(proba))
                best_score = float(proba[best_i])
                # Build top-3
                top_idx = np.argsort(proba)[::-1][:3]
                top3_list[i] = [{"category": self.supervised_classifier.classes_[ti],
                                  "confidence": round(float(proba[ti]), 3)} for ti in top_idx]
                if best_score >= 0.35:
                    predicted_categories[i] = self.supervised_classifier.classes_[best_i]
                    confidence_scores[i] = best_score
                    continue

            # Tier 4: Semantic similarity fallback
            if row['Combined_Text'].strip():
                semantic_indices.append(i)
                semantic_texts.append(row['Combined_Text'])

        if semantic_texts:
            print(f"Semantic fallback on {len(semantic_texts)} cases...")
            embs = self.model.encode(semantic_texts, batch_size=64, show_progress_bar=False)
            sims = cosine_similarity(embs, self.cat_embeddings)
            for j, ri in enumerate(semantic_indices):
                score = float(np.max(sims[j]))
                if score > 0.15:
                    predicted_categories[ri] = self.valid_categories[int(np.argmax(sims[j]))]
                    confidence_scores[ri] = score
                else:
                    predicted_categories[ri] = "Others"
                    confidence_scores[ri] = score

        df['Predicted_Category'] = predicted_categories
        df['Prediction_Confidence'] = [round(c, 3) for c in confidence_scores]
        df['Main_Category'] = df['Predicted_Category'].map(lambda x: self.cat_to_main.get(x, "Others"))
        df['Combined_Text'] = df['Combined_Text']  # keep for active learning
        df = df.fillna("")

        # --- Summary with MoM % change ---
        top_cats = df['Predicted_Category'].value_counts().head(10)
        main_counts = df['Main_Category'].value_counts()

        # Month-over-Month change per main category
        mom_changes = {}
        if 'Opened Date' in df.columns:
            try:
                df['_month'] = pd.to_datetime(df['Opened Date'], errors='coerce').dt.to_period('M').astype(str)
                valid_months = sorted([m for m in df['_month'].unique() if m != 'NaT'])
                if len(valid_months) >= 2:
                    last_m, prev_m = valid_months[-1], valid_months[-2]
                    last_counts = df[df['_month'] == last_m]['Main_Category'].value_counts()
                    prev_counts = df[df['_month'] == prev_m]['Main_Category'].value_counts()
                    for cat in df['Main_Category'].unique():
                        lc = int(last_counts.get(cat, 0))
                        pc = int(prev_counts.get(cat, 0))
                        if pc > 0:
                            change_pct = round((lc - pc) / pc * 100, 1)
                            mom_changes[cat] = {"change_pct": change_pct, "last": lc, "prev": pc}
            except Exception as e:
                print("MoM error:", e)

        summary = {
            "top_categories": {"labels": top_cats.index.tolist(), "values": top_cats.values.tolist()},
            "main_categories": {"labels": main_counts.index.tolist(), "values": main_counts.values.tolist()},
            "total_cases": n,
            "high_confidence": sum(1 for c in confidence_scores if c >= 0.8),
            "low_confidence": sum(1 for c in confidence_scores if c < 0.5),
            "mom_changes": mom_changes
        }

        # --- Analytics: Trends (monthly) ---
        trends = {}
        if 'Opened Date' in df.columns:
            try:
                df['_month'] = pd.to_datetime(df['Opened Date'], errors='coerce').dt.to_period('M').astype(str)
                tdf = df[df['_month'] != 'NaT'].groupby(['_month', 'Main_Category']).size().reset_index(name='count')
                months = sorted(tdf['_month'].unique().tolist())
                cats = df['Main_Category'].value_counts().head(5).index.tolist()
                datasets = []
                colours = ['#6366f1', '#a855f7', '#ec4899', '#10b981', '#f59e0b']
                for ci, cat in enumerate(cats):
                    sub = tdf[tdf['Main_Category'] == cat].set_index('_month')['count']
                    datasets.append({
                        "label": cat,
                        "data": [int(sub.get(m, 0)) for m in months],
                        "borderColor": colours[ci % len(colours)],
                        "backgroundColor": colours[ci % len(colours)] + "33",
                        "tension": 0.4, "fill": True
                    })
                trends = {"labels": months[-24:], "datasets": [{"label": d["label"], "data": d["data"][-24:], "borderColor": d["borderColor"], "backgroundColor": d["backgroundColor"], "tension": d["tension"], "fill": d["fill"]} for d in datasets]}
            except Exception as e:
                print("Trends error:", e)

        # --- Analytics: Severity ---
        severity = {}
        if 'Severity' in df.columns:
            try:
                svdf = df.groupby(['Main_Category', 'Severity']).size().reset_index(name='count')
                cats_list = df['Main_Category'].value_counts().head(8).index.tolist()
                sevs = [s for s in df['Severity'].unique().tolist() if s != ""]
                sev_colours = ['#ef4444', '#f59e0b', '#10b981', '#6366f1', '#a855f7']
                sev_datasets = []
                for si, sev in enumerate(sevs):
                    sub = svdf[svdf['Severity'] == sev].set_index('Main_Category')['count']
                    sev_datasets.append({
                        "label": str(sev),
                        "data": [int(sub.get(c, 0)) for c in cats_list],
                        "backgroundColor": sev_colours[si % len(sev_colours)] + "cc"
                    })
                severity = {"labels": cats_list, "datasets": sev_datasets}
            except Exception as e:
                print("Severity error:", e)

        # --- Analytics: Agent leaderboard ---
        agents = []
        if 'Case Owner' in df.columns:
            try:
                ag = df.groupby('Case Owner').agg(
                    count=('Case Number', 'count'),
                    top_cat=('Predicted_Category', lambda x: x.value_counts().index[0] if len(x) > 0 else "")
                ).sort_values('count', ascending=False).head(15)
                agents = [{"name": str(idx), "count": int(row['count']), "top_category": str(row['top_cat'])} for idx, row in ag.iterrows()]
            except Exception as e:
                print("Agents error:", e)

        # --- Analytics: Resolution Time ---
        resolution_time = []
        age_col = None
        for c in ['Case Age', 'Age (Days)', 'Age']:  # try common column names
            if c in df.columns:
                age_col = c
                break
        if age_col is None and 'Opened Date' in df.columns and 'Closed Date' in df.columns:
            try:
                df['_age_days'] = (pd.to_datetime(df['Closed Date'], errors='coerce') -
                                   pd.to_datetime(df['Opened Date'], errors='coerce')).dt.days
                age_col = '_age_days'
            except Exception:
                pass
        if age_col:
            try:
                df['_age_num'] = pd.to_numeric(df[age_col], errors='coerce')
                res = df.groupby('Predicted_Category')['_age_num'].agg(['mean', 'median', 'count']).dropna()
                res = res[res['count'] >= 5].sort_values('mean', ascending=False).head(15)
                resolution_time = [{"category": str(idx), "avg_days": round(float(r['mean']), 1),
                                     "median_days": round(float(r['median']), 1), "count": int(r['count'])}
                                    for idx, r in res.iterrows()]
            except Exception as e:
                print("Resolution time error:", e)

        # --- Benchmark (fuzzy comparison) ---
        benchmark = None
        actual_col = None
        for col in ['Other Issue Category Description', 'Issue Category']:
            if col in df.columns:
                actual_col = col
                break
        if actual_col:
            df['_actual_taxonomy'] = df[actual_col].apply(self._normalize_label)
            eval_mask = df['_actual_taxonomy'].notna()
            edf = df[eval_mask]
            if len(edf) > 0:
                correct_mask = edf['_actual_taxonomy'] == edf['Predicted_Category']
                n_correct = int(correct_mask.sum())
                n_total = len(edf)
                accuracy = round((n_correct / n_total) * 100, 1)

                # All mismatches sorted by lowest confidence — send up to 200
                wrong = edf[~correct_mask].copy()
                wrong['Prediction_Confidence'] = df.loc[wrong.index, 'Prediction_Confidence']
                wrong = wrong.sort_values('Prediction_Confidence', ascending=True).head(200)
                top_mismatches = [{
                    "case_number": str(r.get('Case Number', '')),
                    "subject": str(r.get('Subject', ''))[:90],
                    "actual": str(r['_actual_taxonomy'])[:60],
                    "predicted": str(r['Predicted_Category'])[:60],
                    "severity": str(r.get('Severity', '')),
                    "confidence_pct": int(r['Prediction_Confidence'] * 100)
                } for _, r in wrong.iterrows()]

                # Confidence histogram buckets (0-10, 10-20, ..., 90-100)
                all_confidences = [int(c * 100) for c in confidence_scores]
                hist_buckets = {f"{i*10}-{(i+1)*10}%": sum(1 for c in all_confidences if i*10 <= c < (i+1)*10)
                                for i in range(10)}
                hist_buckets["100%"] = sum(1 for c in all_confidences if c == 100)

                benchmark = {
                    "evaluated": n_total, "correct": n_correct,
                    "incorrect": n_total - n_correct, "accuracy_pct": accuracy,
                    "top_mismatches": top_mismatches,
                    "confidence_histogram": hist_buckets
                }

        # --- Table (with top-3) ---
        cols_to_return = ['Case Number', 'Subject', 'Severity', 'Priority', 'Region',
                          'Predicted_Category', 'Prediction_Confidence', 'Main_Category',
                          'Case Owner', 'Account Name', 'Opened Date', 'Status', 'Combined_Text']
        cols = [c for c in cols_to_return if c in df.columns]
        table_data = df[cols].to_dict(orient='records')
        # Inject top-3
        for i, row in enumerate(table_data):
            row['top_3'] = top3_list[i] if i < len(top3_list) else []
            row.pop('Combined_Text', None)  # strip from table, keep in current_dataset

        return {
            "table": table_data,
            "summary": summary,
            "benchmark": benchmark,
            "analytics": {"trends": trends, "severity": severity, "agents": agents,
                          "resolution_time": resolution_time},
            "valid_categories": self.valid_categories
        }
