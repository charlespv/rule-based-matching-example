# Rule Based Matching Example

## Getting started

1. Install Docker Desktop
2. Read install Giskard instructions https://docs.giskard.ai/start/guides/installation
3. git clone https://github.com/Giskard-AI/giskard.git
4. cd giskard
5. docker compose pull && docker compose up -d --force-recreate --no-build
6. cd ..
7. poetry install
8. poetry run giskard worker start
9. Go to http://localhost:19000
10. poetry run python project.py