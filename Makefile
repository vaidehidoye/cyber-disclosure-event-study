PYTHON := .venv/bin/python

.PHONY: all data analysis test setup

all: data analysis

setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.edgar
	$(PYTHON) -m src.events
	$(PYTHON) -m src.prices
	$(PYTHON) -m src.features

analysis:
	$(PYTHON) -m src.analysis

test:
	$(PYTHON) -m pytest -q
