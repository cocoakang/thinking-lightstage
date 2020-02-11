SET DATA_ROOT="G:/2019_fresh_meat/3_11_beer_can/pattern/"
SET DATA_FILE_NAME_BASE="cam00_data_"
SET /A IF_DUMP=1
SET /A IF_ORIGIN_COLORFUL=1
SET /A IF_USE_GUESSED_PARAM=1

REM for /L %%i in (0,1,4) do (
REM     START /B python tf_ggx_fittinger.py %%i %DATA_ROOT% %IF_ORIGIN_COLORFUL% %IF_DUMP% %DATA_FILE_NAME_BASE%
REM )
python tf_ggx_fittinger.py 0 %DATA_ROOT% %IF_ORIGIN_COLORFUL% %IF_DUMP% %DATA_FILE_NAME_BASE% %IF_USE_GUESSED_PARAM%

REM python tf_ggx_render_visualizer.py 0 %DATA_ROOT% %IF_ORIGIN_COLORFUL%

SET /A THREAD_NUM= 1
SET WANTED_LIGHT_PAN="G:/2019_fresh_meat/3_11_beer_can/pattern/wanted_lights.txt"
REM python fitted_results_render.py %THREAD_NUM% %DATA_ROOT% %IF_ORIGIN_COLORFUL% %WANTED_LIGHT_PAN%
REM python fitted_results_combiner.py %THREAD_NUM% %DATA_ROOT%