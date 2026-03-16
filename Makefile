.PHONY: build test bench clean

build:
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev 2>/dev/null
	@cmake --build build

test: build
	@cd build && ctest --output-on-failure

bench: build
	@./build/benchmark_gemm

clean:
	@rm -rf build
