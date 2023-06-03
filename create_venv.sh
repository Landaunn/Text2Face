python3 -m venv env
. env/bin/activate
pip3 install -U pip && pip3 install -U setuptools
pip3 install -r requirements.txt
pip3 install huggingface-hub==0.14.1