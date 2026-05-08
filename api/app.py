from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allow E-Library frontend to call this API

# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'recommender.pkl')
BOOKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'books_clean.csv')

model = None
books_df = None

def load_artifacts():
    global model, books_df
    try:
        model = joblib.load(MODEL_PATH)
        books_df = pd.read_csv(BOOKS_PATH)
        print("✅ Model and book data loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠️  Artifacts not found: {e}. Run the notebook first.")

load_artifacts()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "elib-recommender API is running 🚀",
        "model_loaded": model is not None,
        "books_loaded": books_df is not None
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Accepts a user_id and returns top N book recommendations.

    Request body (JSON):
    {
        "user_id": 12345,
        "top_n": 10
    }
    """
    if model is None or books_df is None:
        return jsonify({"error": "Model not loaded. Run the training notebook first."}), 503

    data = request.get_json()

    if not data or 'user_id' not in data:
        return jsonify({"error": "user_id is required"}), 400

    user_id = data['user_id']
    top_n = data.get('top_n', 10)

    try:
        # Get all book IDs
        all_book_ids = books_df['book_id'].unique()

        # Predict ratings for all books for this user
        predictions = [
            (book_id, model.predict(user_id, book_id).est)
            for book_id in all_book_ids
        ]

        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_books = predictions[:top_n]

        # Enrich with book metadata
        results = []
        for book_id, predicted_rating in top_books:
            book_info = books_df[books_df['book_id'] == book_id].iloc[0]
            results.append({
                "book_id": int(book_id),
                "title": book_info.get('title', 'Unknown'),
                "authors": book_info.get('authors', 'Unknown'),
                "average_rating": float(book_info.get('average_rating', 0)),
                "predicted_rating": round(float(predicted_rating), 2),
                "image_url": book_info.get('image_url', ''),
            })

        return jsonify({
            "user_id": user_id,
            "recommendations": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend/book', methods=['POST'])
def recommend_similar():
    """
    Given a book_id, returns similar books (content-based).

    Request body (JSON):
    {
        "book_id": 42,
        "top_n": 5
    }
    """
    if books_df is None:
        return jsonify({"error": "Book data not loaded."}), 503

    data = request.get_json()
    if not data or 'book_id' not in data:
        return jsonify({"error": "book_id is required"}), 400

    book_id = data['book_id']
    top_n = data.get('top_n', 5)

    try:
        target = books_df[books_df['book_id'] == book_id]
        if target.empty:
            return jsonify({"error": "Book not found"}), 404

        # Simple content filter: same author or similar average rating range
        target_authors = target.iloc[0].get('authors', '')
        target_rating = float(target.iloc[0].get('average_rating', 3.5))

        similar = books_df[
            (books_df['book_id'] != book_id) &
            (
                (books_df['authors'] == target_authors) |
                (books_df['average_rating'].between(target_rating - 0.3, target_rating + 0.3))
            )
        ].head(top_n)

        results = similar[['book_id', 'title', 'authors', 'average_rating', 'image_url']].to_dict(orient='records')

        return jsonify({
            "based_on_book_id": book_id,
            "similar_books": results,
            "count": len(results)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
