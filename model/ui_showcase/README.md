# UI showcase data: 12 real meals with model input and output

Real meals from 4 study participants: #001 (healthy), #011 and #022 (prediabetes), and #036 (type 2 diabetes). All meals
come from days 8 and later, which the model never saw during training. For each person we kept the 3 meals where the
predicted glucose curve came closest to the real CGM curve. `showcase_best_meals.png` shows all 12 at a glance.

The JSON follows the live API, so screens built on these files work with the real service unchanged.

## Files

| Path | What it is | Use it for |
|---|---|---|
| `showcase_best_meals.png` | Overview figure: photo, predicted curve with its 80% band, and the real CGM curve for each meal | Quick look |
| `index.json` | List of people and meals with their file paths and headline numbers | Navigation |
| `meals/<meal>.json` → `input` | What the app sends for the meal (photo, time, meal type) and the CGM and heart rate from the 2 hours before | "Take a photo" flow |
| `meals/<meal>.json` → `output` | The API response: food tags, estimated nutrition, predicted curve with its band, peak, chance of going above 140 or 180, explanation, confidence, what-if scenarios, similar past meals | Meal result screen |
| `meals/<meal>.json` → `observed` | **Demo only**: the real CGM curve and the participant's own food log. The live app will not have this | "Prediction vs reality" demo |
| `people/<id>/input_week1.json` | First-week data: CGM and heart rate every 5 min, and every meal with its photo and logged nutrients | History or onboarding screen |
| `people/<id>/output_profile.json` | Profile response: personal averages, diet guide (carb budget, biggest spikes with suggested swaps), metabolic profile (estimated HbA1c, risk tier) | Profile screen |
| `photos/` | Meal photos, 640 px wide | Images |

## Notes for design

- In these files `photo_key` holds a path into `photos/`. In the live API it is an opaque key that the app uses to fetch the image.
- Glucose is always in mg/dL (divide by 18 for mmol/L). Every curve uses the same grid, 0 to 120 minutes in steps of 5.
- Show the band, not only the line. About 80% of real curves fall inside it.
- `confidence.flags` and the `note` fields are written to be shown to users as they are. The diet guide and risk numbers are estimates and must not be presented as a diagnosis.
- These 12 are best cases, picked on purpose. For a typical meal the curve is off by about 15 mg/dL on average, so the design should handle predictions that miss.
- Field-by-field reference: `model/docs/api_contract.md`.
- To regenerate: `scripts/best_subjects.py`, then `scripts/export_ui_package.py`.

## Data source and license

The photos, CGM and food logs come from CGMacros: Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. *CGMacros: a scientific
dataset for personalized nutrition and diet monitoring* (version 1.0.0). PhysioNet, 2025. doi:10.13026/3z8q-x658.
The license is **CC BY-NC-SA 4.0**: keep this attribution, do not use the data commercially, and share anything derived
from it under the same license.
