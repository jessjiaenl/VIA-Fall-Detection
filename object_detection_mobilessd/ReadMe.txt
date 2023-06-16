=========================
= Build sample program  =
=========================
1. Set toolchain path
   - Edit the CMakeLists.txt
   - Change All NDK_STANDALONE_TOOLCHAIN to correct path
2. Change working directory to build
   Choose to build 32-bit/64-bit sample program
   - 32-bit
     cmake -DBENV="P" -DTARGET=arm ../
   - 64-bit
     cmake -DBENV="P" -DTARGET=arm ../
3. Build sample
   make

=========================
= Run sample program    =
=========================
Please follow the execution flow
- Execute 0_python_convert_input_2_bin.bat
- Execute 2_push_model.bat
- Execute 3_push_input.bat
- Execute 4_run.bat
