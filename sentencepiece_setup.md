git clone --recurse-submodules https://github.com/cimeister/unimixlm.git
# OR if you already cloned:
git submodule update --init --recursive

pip install sentencepiece

cd sentencepiece 
mkdir build
cd build
make -j $(nproc)
make install
