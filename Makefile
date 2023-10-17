.PHONY: all setup

VENV_NAME?=APMLvenv
VENV_ACTIVATE=. ./$(VENV_NAME)/bin/activate
PYTHON=./${VENV_NAME}/bin/python3

all: setup fonts

setup: ./$(VENV_NAME)/bin/activate

./$(VENV_NAME)/bin/activate: requirements.txt
	test -d ./$(VENV_NAME) || python3 -m venv ./$(VENV_NAME)
	${PYTHON} -m pip install --upgrade pip 
	${PYTHON} -m pip install -r requirements.txt --upgrade
	touch ./$(VENV_NAME)/bin/activate

fonts:
	sudo apt install msttcorefonts -qq
	rm ~/.cache/matplotlib -rf

activate:
	@echo "To activate the virtual environment, execute:"
	@echo "source ./$(VENV_NAME)/bin/activate"