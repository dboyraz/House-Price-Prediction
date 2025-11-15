

This project uses "uv" as Python package/project manager.

Please install uv first and then run: "uv sync" so it can create the virtual environment and install all dependencies automatically.

# House Price Prediction using Ames Housing dataset

## 1. Problem Definition

**What we’re predicting**

This project builds a regression model to predict house sale prices in Ames, Iowa.
The target variable in the raw data is `SalePrice`; in the modeling pipeline we use its log transformed version:

* `saleprice_log = log1p(SalePrice)`

**Who benefits**

* **Home buyers / sellers** – get a data-driven price estimate for a property with given characteristics.
* **Real-estate agents / analysts** – can benchmark listing prices, identify under/over-priced properties.
* **Developers / investors** – can run “what-if” scenarios (e.g. “What happens if I add a garage or finish the basement?”).

**How the model will be used**

* The final model is a **RandomForestRegressor** 
* It takes in ~30 engineered numeric features (size, quality, age, bathrooms, amenities, etc.) and returns a predicted log price, which is then converted back to dollars with `np.expm1`.
* This model is intended to be served in a web app.

---

## 2. Why This Problem Matters

Accurate house price estimation is important because:

* Housing is typically the largest asset most people ever buy or sell.
* Overpricing can lead to longer time on market; underpricing can mean lost money for sellers.
* Lenders and investors rely on valuation models for risk assessment and portfolio decisions.
* A transparent, data driven model helps reduce guesswork and provides a consistent baseline for negotiations.

From a machine learning perspective, this is also a classic regression problem that showcases:

* Handling skewed targets (log transforms)
* Managing missing data
* Doing feature engineering for better model performance
* Comparing and tuning multiple model families

---

## 3. Data Source and Retrieval

**Source**

The dataset is the **Ames Housing** data hosted on Kaggle:

 [Ames Iowa Housing Data](https://www.kaggle.com/datasets/marcopale/housing/data)

This dataset contains detailed information about residential properties sold in Ames, Iowa, including lot characteristics, building type and style, quality and condition ratings, basement and garage details, and sale information.

**Retrieval steps (as used in the notebook)**

1. Download `AmesHousing.csv` from the Kaggle dataset page.

2. Place it in the working directory of the notebook.

After this, the notebook proceeds with preprocessing, EDA, feature engineering, and model training.

---

## 4. Exploratory Data Analysis (EDA)

This section summarizes the EDA performed in the notebook:

### 4.1 Distributions and Target Transform

**Raw `SalePrice`**

* We examined the distribution of `SalePrice`.

* The distribution is heavily right skewed (long tail of expensive houses).

* This motivates using a log transform of the target.

**Log-transformed `SalePrice`**

* We applied `log1p` to reduce skew.

* After transformation, the target distribution is much closer to symmetric.

**Other numeric features / skewness**

* We computed skewness for all numeric columns.

* Then identified highly skewed features (`|skew| > 1`).

* Many features such as `misc_val`, `pool_area`, `lot_area`, `low_qual_fin_sf`, `3ssn_porch`, `bsmtfin_sf_2`, `enclosed_porch`, `screen_porch`, `bsmt_half_bath`, `mas_vnr_area`, `open_porch_sf`, `wood_deck_sf`, `1st_flr_sf`, `bsmtfin_sf_1`, `gr_liv_area`, `total_bsmt_sf` showed strong right skew.

**Treatment of skewed features**

* For key skewed continuous features, we created log-transformed versions.

* These log features are later candidates for the model and improve linearity / reduce the impact of extreme values.

---

### 4.2 Missing Values


* **Numeric columns**: missing values replaced with 0 (often representing “absence” of the feature, e.g. no basement area).

* **Categorical columns**: missing values replaced with the string 'None', explicitly indicating “no category / not present”.

For model training with multiple models, the final pipeline also includes:

* `SimpleImputer(strategy="median")` inside the model pipeline for safety and to ensure consistent handling of missing values during cross-validation and in production.

---

### 4.3 Correlations

We explored correlations between numeric features and the log-transformed target `saleprice_log` (created during feature engineering).

**Key correlation insights:**

* **Strong positive correlations**:

  * `overall_qual` – overall material/finish quality (strongest single predictor).
  * Size-related / log-transformed features: `total_sf_log`, `gr_liv_area_log`, `1st_flr_sf_log1p`, `total_bsmt_sf`.
  * Quality encodings: `exter_qual_num`, `kitchen_qual_num`, `bsmt_qual_num`, `fireplace_qu_num`, `heating_qc_num`.
  * Garage and bathroom features: `garage_cars`, `garage_area`, `total_bath`, `total_full_bath`.

* **Moderate correlations**:

  * Age/timing features: `house_age`, `since_remodel`, `garage_age_log1p`.
  * Outdoor/amenity features: `total_outdoor_sf`, `open_porch_sf_log1p`, `wood_deck_sf_log1p`, `has_fireplace`, `has_deck`, `has_porch`.

These correlations were used to select a subset of features for modeling:

* We kept engineered features with **|corr| ≥ 0.3** and additional important raw numerics.
* Final selected features include 30 numeric variables such as:

  * `total_sf_log`, `1st_flr_sf_log1p`, `total_bsmt_sf_log1p`
  * `exter_qual_num`, `kitchen_qual_num`, `bsmt_qual_num`, `fireplace_qu_num`, `heating_qc_num`
  * `overall_score` (interaction of overall quality and condition)
  * `house_age`, `since_remodel`, `garage_age_log1p`
  * `total_bath`, `bath_per_bedroom`
  * `lot_area_log1p`, `mas_vnr_area_log1p`
  * `bsmt_finished_sf`, `total_outdoor_sf`, `open_porch_sf_log1p`, `wood_deck_sf_log1p`
  * `has_fireplace`, `has_deck`, `has_porch`
  * and key raw numerics: `overall_qual`, `garage_cars`, `garage_area`, `total_bsmt_sf`, `totrms_abvgrd`.


---

### 4.4 Outliers and Extreme Values

1. **Visualization of target distribution**

   * The histogram of `SalePrice` clearly shows a long right tail (expensive outlier houses).
   * Log-transforming `SalePrice` (`saleprice_log`) reduces the impact of these extremes and makes the distribution more symmetric.

2. **Log-transforming strongly skewed features**

   * For highly skewed numeric features (e.g. `lot_area`, `gr_liv_area`, `total_bsmt_sf`, `mas_vnr_area`, `wood_deck_sf`), log1p transforms compress extreme values and stabilize variance.

3. **Dropping some highly skewed, mostly-zero features**

   * For each highly skewed feature, we computed:

     * percentage of zeros
     * correlation with `log_saleprice`

     ```python
     columns_to_drop = []
     for col_name in highly_skewed.index:
         zero_pct = (df[col_name] == 0).sum() / len(df) * 100
         correlation = abs(df[col_name].corr(log_saleprice))
         if zero_pct > 90 or correlation < 0.1:
             columns_to_drop.append(col_name)
     ```

   * Features with >90% zeros or very low correlation were candidates for dropping.

4. **Context aware decisions**

   * Some extreme but semantically important features (e.g. large `lot_area`, high `gr_liv_area`, amenity areas) were kept and transformed rather than dropped, because they meaningfully influence price even if they’re rare.

---

### Summary of EDA

* **Distributions**: `SalePrice` and several numeric features are right skewed; applied log transforms.
* **Missing values**: numeric NaNs → 0; categoricals → `"None"`, plus median imputation inside model pipelines.
* **Correlations**: size, overall quality, and certain quality ratings showed strongest relationships with price; used to guide feature selection.
* **Outliers**: managed primarily through log transforms and dropping extremely sparse features, rather than hard deletion.