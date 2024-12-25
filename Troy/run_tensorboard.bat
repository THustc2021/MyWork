@REM if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )
cd results
call conda activate rlearn
tensorboard --logdir .
pause
