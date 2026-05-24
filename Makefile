PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install lint format test train-t5 train-bert evaluate serve ui clean

install:
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src tests scripts
	$(PYTHON) -m black --check src tests scripts
	$(PYTHON) -m isort --check src tests scripts

format:
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m isort src tests scripts

test:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

train-t5:
	$(PYTHON) scripts/train_t5.py

train-bert:
	$(PYTHON) scripts/train_bert.py

evaluate:
	$(PYTHON) scripts/evaluate.py

serve:
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) src/ui/gradio_app.py

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in (Path('.pytest_cache'), Path('htmlcov'), Path('build'), Path('dist')) if p.exists()]; [p.unlink() for p in Path('.').glob('*.coverage*') if p.is_file()]"
