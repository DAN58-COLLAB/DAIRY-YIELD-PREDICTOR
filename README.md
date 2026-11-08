# Dairy Yield Predictor

A small scikit-learn classification exercise that predicts whether an illustrative cow record is above or below a 15-litre daily-yield target using daily feed quantity and lactation day.

## Requirements
Python 3.9+, NumPy and scikit-learn.

## Run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python dairy_farm_yield_predictor.py
```

## Inputs and output
- Input features: feed in kg/day and lactation day.
- Output class: above or below the illustrative 15 L/day target.
- The script also prints the model's predicted class probability for three sample records.

## Limitations
The training set contains only 20 hand-written illustrative rows. It is not a validated dairy-production model. The displayed probability is not a guarantee of accuracy. Do not use this exercise to make feeding, veterinary or farm-management decisions.
