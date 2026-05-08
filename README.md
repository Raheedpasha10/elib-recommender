# elib-recommender 📚🤖

An ML-powered book recommendation engine built as an extension for the E-Library app.

## Project Overview
This project trains a book recommendation model on the Goodreads public dataset and exposes it as a lightweight API that integrates with the E-Library application.

## Structure
```
elib-recommender/
├── data/           # Raw and processed datasets
├── notebooks/      # Jupyter/Colab notebooks (EDA + training)
├── models/         # Trained model artifacts (.pkl files)
├── api/            # Flask API to serve the model
└── README.md
```

## Tech Stack
- **Python** — core language
- **Pandas / NumPy** — data processing
- **Scikit-learn / Surprise** — ML models
- **Flask** — model serving API
- **Google Colab** — training environment

## Dataset
[Goodreads Dataset](https://mengtingwan.github.io/data/goodreads.html) — millions of real user ratings and interactions.

## Features
- Collaborative filtering based book recommendations
- Content-based filtering using book metadata
- Hybrid recommendation approach
- REST API endpoint for E-Library integration
