:: python tf_ggx_render.py 0 test_data/
:: python tf_ggx_render.py 1 test_data/
:: python tf_ggx_render.py 2 test_data/
:: python tf_ggx_render.py 3 test_data/
:: python tf_ggx_render.py 4 test_data/
:: python tf_ggx_render.py 5 test_data/
:: python tf_ggx_render.py 6 test_data/
:: python tf_ggx_render.py 7 test_data/
:: python tf_ggx_render.py 8 test_data/
:: python tf_ggx_render.py 9 test_data/

:: SET /A THREAD = 0
:: SET /A BLOCK_SIZE = 1000

:: set kai=%time:~0,-3%
:: echo\&echo 批处理开始运行时间：%kai%

:: for /L %%i in (0,1,9) do (
::     START /B python tf_ggx_fittinger.py %%i test_data/ logs/ %BLOCK_SIZE%
:: )

python tf_ggx_fittinger.py 0 test_data/test_fitting_3channel/ logs/ 1000 1 1 cam00_data.bin