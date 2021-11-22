# Goodreads-app-Heroku
Predicting multi tags of quotes based on Goodreads quotes.

# Goodreads
<!-- Scraping data from goodreads and makes an EDA and modeling on data. -->

### What i did in the project:
* Start with scraping all quotes in Goodreads that are `82460 quotes` with `27 label`, that each label have `2945 quote`.
* Makes all preprocessing pipeline for cleaning data.
* Makes some of EDA 'Exploratory Data Analysis' for each words appear with all tags and alos word cloud for visualization, feature engineering for knowing lenght fo each quote and number of words in each one.
* Showing most frequent n-grams "one, two" words appear in each tag.
* Makes frequent tags which is appear in data, and customize the tags by the top 20 tags appear.
* Modeling as a ML models for multi class classification and also DL model by RoBERTa.

-----
### Publishing data.
* Shared the data on [kaggle](https://www.kaggle.com/abdokamr/good-reads-quotes)
* Makes some notebooks like:
  * [EDA | Feature Engineering For Multi-Class](https://www.kaggle.com/abdokamr/eda-feature-engineering-for-multi-class).
  * [Multi-label tags classification](https://www.kaggle.com/abdokamr/multi-class-tags-classification)
  *  Modling by pretrained model [`RoBERTa` for `multi-class classification`](https://www.kaggle.com/abdokamr/goodreads-modeling-by-roberta-for-multi-class) porblem to predicting a quote is `love` or `motivation` or `wisdom`.
