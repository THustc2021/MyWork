@REM if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )
set PYTHONPATH=.
call conda activate rlearn
python troy_env_pettingzoo/train_troy_marl.py
