rm -rf build
mkdir build
cd build
cmake -DBENV="P" -DTARGET=aarch64 ../
make
cp MobilenetSSDDemo ../
cd ..
rm -rf build
