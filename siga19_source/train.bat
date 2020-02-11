SET /A SAMPLE_NUM=200000000
SET /A DIMENSION=11
SET DATA_ROOT=../training_data/
SET /A FIX_NUM=12
SET TRAIN_RATIO="0.9"

::Generate training data
python origin_parameter_generator.py %DATA_ROOT% %SAMPLE_NUM% %TRAIN_RATIO% %FIX_NUM%

::Train here
python tame.py %DATA_ROOT%