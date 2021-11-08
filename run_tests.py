'''required entrypoint for coveralls'''
import os
import sys

import pytest

EXIT_CODE = pytest.main(['-s','./bayesian_finetuning/tests'])
os.system('rm -rf ./temp')

sys.exit(EXIT_CODE)
