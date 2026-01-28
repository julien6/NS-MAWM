#!/usr/bin/env bash

if command -v python3.10 >/dev/null 2>&1; then
	echo "✅ Python 3.10 was found."
else
	echo "❌ Python 3.10 is not installed. Please install it (ex: sudo apt install python3.10 python3.10-venv)."
	exit 1
fi

echo "🛠️  Creating virtual Python 3.10 environment..."
python3.10 -m venv ./venv

echo "Activation of virtual environment..."
source ./venv/bin/activate

pip install -e .

wget https://anonymous.4open.science/api/repo/Gridcraft-006A/zip
mkdir Gridcraft
mv zip Gridcraft
cd Gridcraft
unzip zip
rm -rf zip
pip install -e .

cd ..

git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai
pip install -e .

cd ..

git clone https://github.com/oxwhirl/smac.git
pip install -e smac/
