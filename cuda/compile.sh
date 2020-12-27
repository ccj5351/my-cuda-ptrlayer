MY_PYTHON=python3.7
TORCH=$(${MY_PYTHON} -c "import os; import torch; print(os.path.dirname(torch.__file__))")
echo $TORCH

$MY_PYTHON setup.py clean
rm -rf build

$MY_PYTHON setup.py build
cp -r build/lib* build/lib
