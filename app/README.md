# Demo app

Two pages: recommendations (pick products or load a real user, see live top-k) and a pipeline monitor (reads the newest
pipeline run record). Full description, deployment notes and limits: `docs/demo.md`.

```sh
pip install -r app/requirements.txt
python -m src.serving.recommend --processed data/processed --out app/artifacts   # about 3 MB
streamlit run app/Recommendations.py
python -m unittest tests.test_recommend tests.test_monitor tests.test_pipeline tests.test_app
```
