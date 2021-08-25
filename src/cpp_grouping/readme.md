Library holding custom c++ code to speed some things up in python!

<!-- 
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Relese -G Ninja
cmake --build .

cd ..
python setup.py build_ext --inplace
cp .\cpp_grouping.cp37-win_amd64.pyd ..\..\..\..\miniconda3\envs\hand_decision_trees\Lib\site-packages\.
python grouping_test.py -->