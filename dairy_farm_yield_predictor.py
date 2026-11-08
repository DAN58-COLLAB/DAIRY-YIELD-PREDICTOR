import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Illustrative training data: [feed_kg_per_day, lactation_day].
# Label: 1 = above 15 litres/day, 0 = below the target.
X = np.array([
    [6.5, 30], [7.0, 45], [5.5, 20], [7.5, 60], [6.0, 25],
    [8.0, 75], [5.0, 15], [7.2, 50], [6.8, 40], [7.8, 70],
    [5.8, 22], [7.3, 55], [6.2, 28], [8.1, 80], [5.3, 18],
    [7.1, 48], [6.6, 35], [7.6, 65], [5.7, 21], [7.9, 72],
])
y = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1])

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)


def predict_cow_yield(feed_kg, lactation_day):
    """Return a predicted yield class and its model probability."""
    features = np.array([[feed_kg, lactation_day]])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])
    status = "Above 15L target" if prediction == 1 else "Below 15L target"
    return status, round(confidence * 100, 1)


if __name__ == "__main__":
    sample_cows = [
        ("Cow 1 (Kamau farm)", 7.2, 50),
        ("Cow 2 (Wanjiku farm)", 5.5, 20),
        ("Cow 3 (Mwangi farm)", 7.8, 70),
    ]
    print("DAIRY FARM: ILLUSTRATIVE DAILY YIELD PREDICTIONS\n")
    for name, feed, lactation_day in sample_cows:
        status, probability = predict_cow_yield(feed, lactation_day)
        print(f"{name}")
        print(f"  Feed: {feed} kg/day | Lactation day: {lactation_day}")
        print(f"  Prediction: {status} (model probability: {probability:.1f}%)\n")
