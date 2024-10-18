#!/bin/sh

python3 -m venv env
. env/bin/activate
git clone git@github.com:rieder/amuse.git
cd amuse
git switch libphantom
pip install -r requirements.txt
./configure
pip install -e .
make framework
make phantom.code fi.code
