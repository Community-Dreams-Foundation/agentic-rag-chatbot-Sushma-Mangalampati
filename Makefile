.PHONY: sanity run install

install:
	pip install -r requirements.txt

sanity:
	@python3 sanity_run.py

run:
	@streamlit run src/app.py --server.port 8501