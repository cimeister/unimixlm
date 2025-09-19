git clone --recurse-submodules https://gitlab.ethz.ch/meistecl/unimixlm.git
# OR if you already cloned:
git submodule update --init --recursive

pip install sentencepiece

cd sentencepiece 
mkdir build
cd build
make -j $(nproc)
make install
