<h1 align="center">🏠 Airbnb Price Prediction</h1>
<h4 align="center">Predicting the nightly price of Airbnb listings from their features — an end-to-end machine-learning regression pipeline </h4>

---

## 🎯 Objective

Predict the **`log_price`** (natural log of the nightly price) of an Airbnb listing from its characteristics — room type, city, capacity, amenities, location, reviews, host info, and more.

We predict the *log* of the price rather than the raw price because the raw price is heavily right-skewed (a few listings above \$700/night), whereas `log_price` is close to a normal distribution (mean ≈ 4.78, median ≈ 4.70) — a much friendlier target for linear models.

## 📊 Dataset

Airbnb listings across **6 US cities** (New York, Los Angeles, San Francisco, Washington DC, Boston, Chicago), with features covering capacity (`accommodates`, `bedrooms`, `beds`, `bathrooms`), `room_type` / `property_type`, `amenities`, host information, reviews and geolocation. A train/test split is provided; the goal is to generate price predictions for the test set.

## 🔧 The pipeline

1. **Exploratory data analysis** — inspecting scales, distributions and the target variable.
2. **Missing values** — a per-case strategy: median imputation for features with <1% missing; **frequency encoding** for `neighbourhood` (too many categories for one-hot); and a `has_reviews` flag plus median imputation for the review columns (20%+ missing = listings with no reviews). *No rows dropped*, since the test set shows the same patterns.
3. **Feature engineering** — ordinal encoding of ordered categories (`room_type`, `city`), binary host flags, parsing of `amenities` into counts and key-amenity flags, host tenure, a **Haversine distance-to-city-center** feature, `beds_per_person`, and neighbourhood frequency encoding.
4. **Standardization & PCA** — `StandardScaler` fit on train only (no leakage); PCA confirms strong redundancy among size features (3 components ≈ 82.9% of variance), which justifies using **Ridge** regularization.
5. **Modeling** — a from-scratch gradient-descent baseline, a regularized polynomial regression, and an SVR for comparison.
6. **Final prediction** — retrain on the full training set, predict the test set, clip to the observed range, and export `predictions.csv`.

## 🤖 Models & results

| Model | R² (validation) | Notes |
|---|:---:|---|
| Linear regression — gradient descent *(from scratch, NumPy)* | 0.575 | Baseline |
| **Polynomial (deg 2) + Ridge** | **0.644** | ✅ **Retained** — best trade-off |
| SVR (RBF kernel) | — | Overfits (train/val gap ≈ 0.26) |

The **degree-2 polynomial + Ridge** model was chosen: it lifts R² by **+0.069** over the linear baseline by capturing non-linear interactions between features, while keeping a small train/val gap (≈ 0.028) thanks to Ridge. Its robustness was confirmed with **5-fold cross-validation**, and its residuals are centered on zero with a near-normal spread (no systematic bias).

<p align="center">
<img width="1398" height="536" alt="image" src="https://github.com/user-attachments/assets/84974c50-8b8d-4e1e-a5f9-aa05d11226ea" />
</p>

## 🔑 Key findings

- **Room type is the strongest driver** after location: an entire home (~\$160/night) costs about **3.5×** a shared room (~\$45/night).
- **Distance to the city center** correlates negatively with price in most cities (Boston −0.44, NYC −0.39, Chicago −0.31) — but **not in Los Angeles** (+0.05), a polycentric city with no single dominant center, where the model relies on raw latitude/longitude instead.
- **Size features** (`room_type`, `accommodates`, `bedrooms`, `beds`) are highly correlated with each other (r > 0.7) — multicollinearity that Ridge handles by penalizing large coefficients.

<p align="center">
  <img width="1719" height="884" alt="image" src="https://github.com/user-attachments/assets/878b186d-e654-4698-9de4-f222630d2646" />
</p>

## 🛠️ Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

## ▶️ How to run

1. Place `airbnb_train.csv` and `airbnb_test.csv` next to the notebook.
2. Open `projet_airbnb.ipynb` and run all cells.
3. The notebook writes the predictions to `predictions.csv`.

---

<p align="center"><a href="https://github.com/Jiboti">⬅️ Back to profile</a></p>
