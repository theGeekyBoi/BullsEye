# BullsEye team branch handoff

Everyone should use [**the BullsEye repository**](https://github.com/theGeekyBoi/BullsEye) and start from [**`integration/current-50-price-news`**](https://github.com/theGeekyBoi/BullsEye/tree/integration/current-50-price-news).

An *integration branch* is an ordinary Git branch where we brought the price-model work and qualitative preparation together. It gives the team one common starting point: the current 50 companies, shared rules, price code, news tables, and existing checks. It is **not a finished model**. News coverage still needs work, and the price models still need improvement. I checked GitHub: `main` is currently behind this branch, so teammates should not start new work from `main` or from either older source branch.

### How each person should work

Create a **separate branch from the integration branch** for each workstream. When a piece is ready, open a pull request **into `integration/current-50-price-news`**. This lets the team review and combine changes without four people editing the shared branch directly.

| Person | Branch to create | Work now | Dependency |
| --- | --- | --- | --- |
| **Adam** | `adam/news-coverage` | Tasks 1 and 2: explain the FNSPID gaps, then recover usable records already in FNSPID. | Give Dhruv the gap findings and recovered records. |
| **Dhruv** | `dhruv/news-tables` | Tasks 4 and 5: prepare the cleaning rules, rebuild the three news tables, and define the shared comparison keys. | Can set up the schema and checks now; completes the rebuild using Adam’s findings. |
| **Bryce** | `bryce/simulator-dashboard` | Build the long-only simulator with made-up predictions, the dashboard shell, and the review sheet. | Ask Aaditya for the clean daily price table when the simulator is ready for real replay. |
| **Aaditya** | `aaditya/price-refinement` | Diagnose and refine the price-only models; prepare the daily price table for Bryce. | Save predictions in the shared output format so Bryce and Dhruv can use them. |

For a fresh clone, each teammate can start with:

```bash
git clone --branch integration/current-50-price-news https://github.com/theGeekyBoi/BullsEye.git
cd BullsEye
git lfs install
git lfs pull
git switch -c your-name/your-task
```

The [team setup guide](https://github.com/theGeekyBoi/BullsEye/blob/integration/current-50-price-news/INTEGRATION_GUIDE.md) covers the files and checks. The [shared experiment contract](https://github.com/theGeekyBoi/BullsEye/blob/integration/current-50-price-news/EXPERIMENT_CONTRACT.md) defines the company list, dates, news tables, and prediction format; everyone should code against it.

**Two later assignments remain:** Task 3, testing an additional news source, should follow Adam’s gap report so the team knows which gaps actually need another source. Tasks 6 and 7—training the news models and combining them with price predictions—follow the cleaned news tables and agreed comparison keys.
