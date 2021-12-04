# LU Decomposition in parallel
This is a program I made to parallelise a the LU decomposition algorithm. Written in Rust, using rayon, ndarray and whatever dependencies they have.

# Running
1. Clone it from git and build it in release mode
	```sh
	cargo build --release
	```
2. Run the executable or just cargo run in release mode
	```sh
	cargo run --release ...
	```
3. Just execute the program without giving it command line arguments and it will explain what to do.
4. It can either generate a number of tests of specified sizes to check execution speed
	```sh
	cargo run --release gen "./example_folder/example_file" 10 100 200 400 800 #...
	```
5. Or run those tests with a specified number of threads
	```sh
	cargo run --release run "./example_folder/example_file" 5 8 #(5 tests 8 threads)
	```